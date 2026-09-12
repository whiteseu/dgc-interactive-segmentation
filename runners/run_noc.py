"""Closed-loop NoC evaluation: simulated interactive segmentation.

Protocol. The first click is the distance-transform maximum of the ground-truth mask.
Each later round then:
  1) re-segments from all accumulated clicks plus the previous round's low-resolution
     logits (mask_input) in one multimask pass, taking the highest predicted-IoU
     candidate as the current mask -- identical for every policy, so the comparison
     is fair;
  2) asks the policy for the next click location (only the oracle sees ground truth;
     the others use the model's own outputs);
  3) reads the polarity of that location from the ground truth, standing in for an
     annotator's label.
At most T=20 clicks. Records the IoU trajectory and whether each proposed click landed
inside the true residual error.

Policies:
  risk      candidate disagreement map -> weighted centroid of its largest component
            (training-free, no ground truth)
  risk_dt   same support, but the deepest interior point instead of the centroid
  entropy   binary entropy of the current low-resolution logits -> same centroid rule
  boundary  random point in a band around the current mask boundary
  random    uniform over the image (floor)
  oracle    the true error map, same interior-point rule (privileged reference)

Output CSV: dataset,id,strategy,n_clicks_85,n_clicks_90,final_iou,iou_traj(json),hit_rate

Usage:
  python run_noc.py --backend sam --model-type vit_b --checkpoint ... \
      --strategy risk --dataset grabcut --manifest ... --data-root ... \
      --out results/noc/noc_sam_vit_b_risk_grabcut.csv
"""
import argparse
import csv
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from run_sam import load_instances, first_click, align_gt_to_image  # noqa: E402

T_MAX = 20
THRESH = (0.85, 0.90)


def build_predictor(args, device):
    if args.backend == "sam":
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
        return SamPredictor(sam)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(args.config, args.checkpoint, device=device))


def iou(a, b, valid=None):
    if valid is not None:
        a, b = a & valid, b & valid
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def blob_centroid(score, region, q=0.9, rng=None):
    """Threshold score at its q-quantile inside region, then take the weighted centroid
    of the largest connected component; fall back to argmax when degenerate. Returns (x, y)."""
    s = np.where(region, score, 0.0)
    if not region.any() or float(s.max()) <= 0:
        if rng is not None:
            ys, xs = np.nonzero(region) if region.any() else (None, None)
            if ys is not None and len(ys):
                k = rng.integers(len(ys))
                return int(xs[k]), int(ys[k])
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        return int(x), int(y)
    thr = np.quantile(s[region], q)
    hot = s >= max(thr, 1e-9)
    lab, n = ndimage.label(hot)
    if n == 0:
        y, x = np.unravel_index(int(np.argmax(s)), s.shape)
        return int(x), int(y)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    k = int(np.argmax(sizes)) + 1
    m = lab == k
    w = s * m
    ys, xs = np.nonzero(m)
    cy = float((ys * w[ys, xs]).sum() / w[ys, xs].sum())
    cx = float((xs * w[ys, xs]).sum() / w[ys, xs].sum())
    # The centroid can fall outside a non-convex component: snap to the nearest point in it
    d2 = (ys - cy) ** 2 + (xs - cx) ** 2
    j = int(np.argmin(d2))
    return int(xs[j]), int(ys[j])


def blob_dt(score, region, q=0.9, rng=None):
    """Same thresholding as blob_centroid, but returns the distance-transform maximum of
    the largest component -- the same selection operator risk_dt uses. Applying it to the
    continuous entropy/MI maps too separates the contribution of the signal from that of
    the selection rule."""
    s = np.where(region, score, 0.0)
    if not region.any() or float(s.max()) <= 0:
        if rng is not None and region.any():
            ys, xs = np.nonzero(region)
            k = rng.integers(len(ys))
            return int(xs[k]), int(ys[k])
        y, x = np.unravel_index(int(np.argmax(score)), score.shape)
        return int(x), int(y)
    thr = np.quantile(s[region], q)
    lab, n = ndimage.label(s >= max(thr, 1e-9))
    if n == 0:
        y, x = np.unravel_index(int(np.argmax(s)), s.shape)
        return int(x), int(y)
    sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
    comp = lab == (int(np.argmax(sizes)) + 1)
    dt = ndimage.distance_transform_edt(comp)
    y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
    return int(x), int(y)


def upsample_lowres(low, shape):
    """Resize a 256x256 low-resolution map up to the image size."""
    img = Image.fromarray(low.astype(np.float32), mode="F")
    return np.asarray(img.resize((shape[1], shape[0]), Image.BILINEAR))


def bald_next(predictor, pts, lbs, base, mask_input, K, rng, dt=False, rho=0.1):
    """Training-free BALD acquisition: jitter the most recent click K times to obtain
    posterior samples, score mutual information as H[mean_k p_k] - mean_k H[p_k], and take
    the centroid of the highest-MI component. Uses no ground truth.

    rho scales the jitter radius by the equivalent-circle radius. It is exposed as a
    parameter so this competing baseline can be re-run at a stronger setting, ruling out
    the possibility that our margin comes from a weakened baseline."""
    H, W = base.shape
    area = max(int(base.sum()), 1)
    r = max(2.0, rho * np.sqrt(area / np.pi))
    probs = []
    for _ in range(K):
        jp = list(pts)
        lx, ly = jp[-1]
        ang = rng.uniform(0, 2 * np.pi)
        rad = rng.uniform(0, r)
        jp[-1] = (int(np.clip(lx + rad * np.cos(ang), 0, W - 1)),
                  int(np.clip(ly + rad * np.sin(ang), 0, H - 1)))
        kw = dict(point_coords=np.array(jp, dtype=np.float32),
                  point_labels=np.array(lbs, dtype=np.int64),
                  multimask_output=True, return_logits=True)
        if mask_input is not None:
            kw["mask_input"] = mask_input[None]
        masks, ious_p, low = predictor.predict(**kw)
        sel = int(np.argmax(ious_p))
        p = 1.0 / (1.0 + np.exp(-upsample_lowres(np.asarray(low[sel]), base.shape)))
        # 1e-6 rather than 1e-12 for the same float32 reason as the entropy branch; this
        # is a no-op for the result because ent() below already re-clips at 1e-6.
        probs.append(np.clip(p, 1e-6, 1 - 1e-6))
    P = np.stack(probs, 0)

    def ent(p):
        p = np.clip(p, 1e-6, 1.0 - 1e-6)  # float32-safe: 1-1e-12 collapses to 1.0
        return -(p * np.log(p) + (1 - p) * np.log(1 - p))

    bald = np.clip(ent(P.mean(0)) - ent(P).mean(0), 0, None).astype(np.float32)
    region = base | (bald > 0.02)
    return (blob_dt if dt else blob_centroid)(bald, region, q=0.9, rng=rng)


def _disag(cands):
    return (((cands[0] ^ cands[1]).astype(np.float32)
             + (cands[0] ^ cands[2]) + (cands[1] ^ cands[2])) / 3)


def _rand_in(mask, base, rng, W, H):
    """Uniform point inside mask; fall back to base, then to the whole image."""
    for m in (mask, base):
        if m is not None and m.any():
            ys, xs = np.nonzero(m)
            k = rng.integers(len(ys))
            return int(xs[k]), int(ys[k])
    return int(rng.integers(W)), int(rng.integers(H))


def _boundary_pt(base, rng, W, H):
    if not base.any():
        return int(rng.integers(W)), int(rng.integers(H))
    band = ndimage.binary_dilation(base, iterations=3) ^ ndimage.binary_erosion(
        base, iterations=3)
    ys, xs = np.nonzero(band)
    if not len(ys):
        return int(rng.integers(W)), int(rng.integers(H))
    k = rng.integers(len(ys))
    return int(xs[k]), int(ys[k])


def next_click(strategy, base, cands, low_sel, gt, rng, dis_frozen=None, excl=None, pts=None):
    """Next click for a given policy. Variants of the disagreement family:
       risk         disagreement support recomputed each round, uniform fallback in b
       risk_dt      same support, deepest interior point of its largest component (DGC)
       risk_nofb    no fallback: return None (stop) when the support is empty
       risk_bndfb   boundary-band fallback instead of a uniform one
       risk_frozen  freeze the first round's support instead of recomputing it
       random_in_d  uniform inside the support, isolating it from the selection rule
    excl: boolean mask of radii around already-clicked points (True = disallowed), set
          only under --no-repeat; None reproduces the default behaviour."""
    H, W = base.shape
    rm = (~excl) if excl is not None else None  # selectable region

    def _rand(region):  # uniform in region & rm; fall back to region, then to the image
        for m in ((region & rm) if rm is not None else region, region):
            if m is not None and m.any():
                ys, xs = np.nonzero(m); k = rng.integers(len(ys))
                return int(xs[k]), int(ys[k])
        return int(rng.integers(W)), int(rng.integers(H))

    def _band():
        return (ndimage.binary_dilation(base, iterations=3)
                ^ ndimage.binary_erosion(base, iterations=3))

    def _dt_center(reg):  # distance-transform maximum of the largest component of reg
        lab, n = ndimage.label(reg)
        if n == 0:
            return None
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        comp = lab == (int(np.argmax(sizes)) + 1)
        dt = ndimage.distance_transform_edt(comp)
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        return int(x), int(y)

    if strategy == "oracle":
        err = base ^ gt
        if rm is not None and (err & rm).any():
            err = err & rm
        if not err.any():
            return None
        return _dt_center(err)
    if strategy == "random":
        return _rand(np.ones((H, W), bool)) if rm is not None else (
            int(rng.integers(W)), int(rng.integers(H)))
    if strategy == "boundary":
        b = _band()
        return _rand(b) if (rm is not None) else _boundary_pt(base, rng, W, H)
    if strategy in ("risk", "risk_nofb", "risk_bndfb", "risk_frozen"):
        dis = dis_frozen if (strategy == "risk_frozen" and dis_frozen is not None) \
            else _disag(cands)
        if (dis > 0).any():
            reg = base | (dis > 0)
            return blob_centroid(dis, (reg & rm) if rm is not None else reg, q=0.9, rng=rng)
        if strategy == "risk_nofb":
            return None
        if strategy == "risk_bndfb":
            return _rand(_band()) if rm is not None else _boundary_pt(base, rng, W, H)
        return _rand(base) if rm is not None else _rand_in(base, base, rng, W, H)
    if strategy == "risk_dt":
        # Deterministic rule: the point of the largest disagreement component farthest
        # from its boundary, mirroring the oracle's interior-point heuristic
        dis = _disag(cands)
        reg = (dis > 0)
        if rm is not None:
            reg = reg & rm
        if reg.any():
            pt = _dt_center(reg)
            if pt is not None:
                return pt
        return _rand(base) if rm is not None else _rand_in(base, base, rng, W, H)
    if strategy == "random_in_d":
        return _rand(_disag(cands) > 0) if rm is not None else \
            _rand_in(_disag(cands) > 0, base, rng, W, H)
    if strategy == "rand_mask":  # stronger random baseline: uniform inside the prediction
        return _rand(base) if rm is not None else _rand_in(base, base, rng, W, H)
    if strategy in ("entropy", "entropy_dt"):
        p = 1.0 / (1.0 + np.exp(-upsample_lowres(low_sel, base.shape)))
        # float32-safe clip: upsample_lowres returns float32, in which 1-1e-12 rounds to
        # exactly 1.0, so a 1e-12 upper bound is a no-op and p can hit 1.0 -> (1-p)*log(1-p)
        # = 0*(-inf) = nan. Measured: nan appeared in the entropy map on 28% of images,
        # and a nan makes np.quantile/argmax below select the artefact pixel, i.e. the
        # entropy baseline clicked noise rather than its high-entropy region. Same 1e-6
        # bound bald_next() already uses.
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        ent = (-(p * np.log(p) + (1 - p) * np.log(1 - p))).astype(np.float32)
        region = base | (ent > 0.3)
        if rm is not None:
            region = region & rm
        op = blob_dt if strategy == "entropy_dt" else blob_centroid
        return op(ent, region, q=0.9, rng=rng)
    if strategy == "fps":  # farthest-point exploration: label-free, uses no uncertainty
        # signal at all -- just the point of the dilated prediction farthest from every
        # previous click
        region = (ndimage.binary_dilation(base, iterations=5) if base.any()
                  else np.ones((H, W), bool))
        if rm is not None:
            region = region & rm
        if not region.any():
            region = rm if (rm is not None and rm.any()) else np.ones((H, W), bool)
        if pts:
            pm = np.ones((H, W), bool)
            for (px, py) in pts:
                pm[int(py), int(px)] = False
            dist = ndimage.distance_transform_edt(pm)
        else:
            dist = ndimage.distance_transform_edt(region)
        dist = np.where(region, dist, -1.0)
        y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
        return int(x), int(y)
    raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["sam", "sam2"])
    ap.add_argument("--model-type", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--strategy", required=True,
                    choices=["risk", "entropy", "boundary", "random", "oracle", "bald",
                             "random_in_d", "risk_nofb", "risk_bndfb", "risk_frozen",
                             "risk_dt", "entropy_dt", "bald_dt", "rand_mask", "fps"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--k-jitter", type=int, default=3)
    ap.add_argument("--bald-rho", type=float, default=0.1,
                    help="jitter radius coefficient for the perturbation-MI baseline "
                         "(0.1 reproduces the original setting); exposed so the baseline "
                         "can be re-run at a stronger setting as a fairness check.")
    ap.add_argument("--rand-first", action="store_true",
                    help="draw the first click uniformly inside the GT mask instead of "
                         "at its distance-transform maximum")
    ap.add_argument("--n-first", type=int, default=1,
                    help="run this many random first clicks per instance, one row each")
    ap.add_argument("--no-repeat", action="store_true",
                    help="forbid clicking again within --excl-rad of a previous click")
    ap.add_argument("--excl-rad", type=float, default=10.0,
                    help="exclusion radius in pixels for --no-repeat")
    ap.add_argument("--click-jitter", type=float, default=0.0,
                    help="noisy interaction: Gaussian positional noise, in pixels, on "
                         "every guided click (t>=2)")
    ap.add_argument("--wrong-pol", type=float, default=0.0,
                    help="noisy interaction: probability of flipping a click's polarity; "
                         "by default polarity is read from the ground truth")
    ap.add_argument("--single-mask", action="store_true",
                    help="protocol ablation: rounds t>=2 decode with multimask_output=False "
                         "(SAM default for multi-point prompts); round 1 unchanged. Only for "
                         "policies that do not read the candidate set (oracle, fps, ...).")
    ap.add_argument("--max-gpu-frac", type=float, default=0.0,
                    help="cap this process at the given fraction of total GPU memory "
                         "(set_per_process_memory_fraction); 0 = no cap. Better to fail "
                         "with OOM here than to crash another job on a shared card.")
    args = ap.parse_args()
    if args.single_mask and (args.strategy.startswith("risk") or args.strategy == "random_in_d"):
        ap.error("--single-mask needs three candidates-free policy; DGC/random_in_d are defined on multimask")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.max_gpu_frac > 0 and device == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.max_gpu_frac)
        print(f"[gpu] capped to {args.max_gpu_frac:.3f} of total "
              f"({args.max_gpu_frac * torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)",
              flush=True)
    predictor = build_predictor(args, device)
    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done = set()
    if args.out.exists():  # resume, keyed on id#first-click-index
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(f"{r['id']}#{r.get('first_seed', 0)}")
    mode = "a" if done else "w"
    fout = open(args.out, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=[
        "dataset", "id", "strategy", "first_seed", "n_clicks_85", "n_clicks_90",
        "final_iou", "hit_rate", "first_d_empty", "round_d1", "iou_traj"])
    if mode == "w":
        writer.writeheader()

    n_done = 0
    for inst_id, img_path, gt, void in load_instances(args.data_root, args.manifest):
        if args.limit and n_done >= args.limit:
            break
        n_done += 1
        if all(f"{inst_id}#{fi}" in done for fi in range(args.n_first)):
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        gt, void = align_gt_to_image(gt, void, image)
        valid = ~void if void is not None else None
        predictor.set_image(image)

        for fi in range(args.n_first):
            if f"{inst_id}#{fi}" in done:
                continue
            crng = np.random.default_rng((args.seed * 100 + fi) * 99_991
                                         + zlib.crc32(str(inst_id).encode()))
            if args.rand_first:
                ys0, xs0 = np.nonzero(gt)
                if len(ys0):
                    kk = crng.integers(len(ys0))
                    cx, cy = int(xs0[kk]), int(ys0[kk])
                else:
                    cx, cy = first_click(gt)
            else:
                cx, cy = first_click(gt)
            pts, lbs = [(cx, cy)], [1]
            mask_input, dis_frozen = None, None
            traj, hits, n85, n90 = [], [], None, None
            first_d_empty, round_d1 = "", -1
            for t in range(1, T_MAX + 1):
                coords = np.array(pts, dtype=np.float32)
                labels = np.array(lbs, dtype=np.int64)
                # Protocol ablation: with --single-mask, rounds t>=2 use SAM's default
                # single-mask decoding for multi-point prompts; round 1 stays multimask so
                # the first prediction is identical to every Table 1 policy.
                single = bool(args.single_mask and t > 1)
                kw = dict(point_coords=coords, point_labels=labels,
                          multimask_output=not single, return_logits=True)
                if mask_input is not None:
                    kw["mask_input"] = mask_input[None]
                masks, ious_p, low = predictor.predict(**kw)
                sel = int(np.argmax(ious_p))
                base = np.asarray(masks[sel]) > 0.0
                mask_input = np.asarray(low[sel])
                cands = np.asarray(masks) > 0.0
                cur = iou(base, gt, valid)
                traj.append(round(cur, 4))
                if args.strategy.startswith("risk"):  # diagnostic: is the support empty?
                    nonempty = bool((_disag(cands) > 0).any())
                    if t == 1:
                        first_d_empty = int(not nonempty)
                        if args.strategy == "risk_frozen":
                            dis_frozen = _disag(cands)
                    if nonempty and round_d1 < 0:
                        round_d1 = t
                if n85 is None and cur >= THRESH[0]:
                    n85 = t
                if n90 is None and cur >= THRESH[1]:
                    n90 = t
                if n90 is not None or t == T_MAX:
                    break
                excl = None
                if args.no_repeat and len(pts):
                    pm = np.zeros(gt.shape, bool)
                    for (px, py) in pts:
                        pm[py, px] = True
                    excl = ndimage.distance_transform_edt(~pm) < args.excl_rad
                if args.strategy in ("bald", "bald_dt"):
                    nxt = bald_next(predictor, pts, lbs, base, mask_input,
                                    args.k_jitter, crng, dt=(args.strategy == "bald_dt"),
                                    rho=args.bald_rho)
                else:
                    nxt = next_click(args.strategy, base, cands, mask_input, gt,
                                     crng, dis_frozen, excl=excl, pts=pts)
                if nxt is None:
                    break
                nx, ny = int(np.clip(nxt[0], 0, gt.shape[1] - 1)), int(
                    np.clip(nxt[1], 0, gt.shape[0] - 1))
                if args.click_jitter > 0:  # noisy interaction: positional noise
                    nx = int(np.clip(nx + round(float(crng.normal(0, args.click_jitter))),
                                     0, gt.shape[1] - 1))
                    ny = int(np.clip(ny + round(float(crng.normal(0, args.click_jitter))),
                                     0, gt.shape[0] - 1))
                hits.append(bool((base ^ gt)[ny, nx]))
                pts.append((nx, ny))
                lab = int(gt[ny, nx])  # polarity is read from the ground truth
                if args.wrong_pol > 0 and crng.random() < args.wrong_pol:
                    lab = 1 - lab  # noisy interaction: wrong polarity
                lbs.append(lab)

            writer.writerow(dict(
                dataset=args.dataset, id=inst_id, strategy=args.strategy,
                first_seed=fi,
                n_clicks_85=n85 if n85 is not None else T_MAX + 1,
                n_clicks_90=n90 if n90 is not None else T_MAX + 1,
                final_iou=traj[-1],
                hit_rate=round(float(np.mean(hits)), 4) if hits else "",
                first_d_empty=first_d_empty, round_d1=round_d1,
                iou_traj=json.dumps(traj)))
            fout.flush()
        if n_done % 50 == 0:
            print(f"[{args.dataset}/{args.strategy}] {n_done} done", flush=True)

    fout.close()
    print(f"[{args.dataset}/{args.strategy}] NOC_FINISHED n={n_done}", flush=True)


if __name__ == "__main__":
    main()
