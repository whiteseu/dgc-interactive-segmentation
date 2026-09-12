"""Rebuttal R2-W4 / R3-W2: run a TRAINED interactive segmenter (SimpleClick, ICCV'23)
through OUR NoC harness, so the comparison is protocol-matched.

Protocol is identical to runners/run_noc.py:
  - same instance list (our manifests), same first click = DT centre of GT (click 1),
  - T=20 rounds max, NoC@85 / NoC@90, instances never reaching the target get NoC=21,
  - IoU computed on the same valid (non-void) region.
The ONLY thing that differs from run_noc.py is the segmenter: SimpleClick replaces the
frozen SAM, and it is run in its own published configuration (NoBRS + ZoomIn +
horizontal-flip TTA, prob_thresh 0.49, `--eval-mode=cvpr` target sizes), so we do not
handicap it.

SimpleClick has no multimask candidates, so our disagreement strategy does not exist for
it; the strategies here are the click policies that DO transfer:
  oracle    DT centre of the largest error component  (SimpleClick's own eval protocol)
  random    uniform over the image
  boundary  random point on the current mask boundary band
  entropy   binary entropy of SimpleClick's probability map (label-free analogue)

usage:
  python run_noc_sc.py --checkpoint weights_sc/cocolvis_vit_base.pth \
      --strategy oracle --dataset grabcut --manifest ... --data-root ... --out ...
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import argparse
import csv
import json
import logging
import sys
import types
import zlib
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

SC_ROOT = Path(rf"{ROOT}/SimpleClick_v1")
sys.path.insert(0, str(SC_ROOT))
sys.path.insert(0, str(Path(rf"{ROOT}/runners")))
sys.path.insert(0, rf"{ROOT}")

import sc_shim  # noqa: F401,E402  (maps mmcv 1.x -> mmcv-lite 2.x + mmengine)


# --- shim the three swin_transformer_helper modules -------------------------------
# They only exist to load *pretrained Swin backbones* and to register a BACKBONES
# registry, neither of which is exercised when we load SimpleClick's own checkpoint.
# Their mmcv-1.x imports (mmcv.fileio / mmcv.parallel / mmcv.runner) were removed in
# mmcv 2.x, so we stub them out rather than pin an unbuildable mmcv.
def _install_shims():
    pkg = "isegm.model.modeling.swin_transformer_helper"
    import isegm.model.modeling  # noqa: F401  (ensures parent packages exist)

    class _Registry:
        def __init__(self, name):
            self.name, self._m = name, {}

        def register_module(self, *a, **k):
            def deco(cls):
                self._m[cls.__name__] = cls
                return cls
            return deco

    mods = {
        "checkpoint": {"load_checkpoint": lambda *a, **k: None},
        "logger": {"get_root_logger": lambda *a, **k: logging.getLogger("simpleclick")},
        "builder": {"BACKBONES": _Registry("backbones")},
    }
    for name, attrs in mods.items():
        full = f"{pkg}.{name}"
        if full in sys.modules:
            continue
        m = types.ModuleType(full)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[full] = m


_install_shims()

from isegm.utils.serialization import load_model            # noqa: E402
from isegm.inference.predictors.base import BasePredictor   # noqa: E402
from isegm.inference.transforms import ZoomIn               # noqa: E402
from isegm.inference.clicker import Clicker, Click          # noqa: E402
from isegm.model.modeling.pos_embed import interpolate_pos_embed_inference  # noqa: E402

from run_sam import load_instances, first_click, align_gt_to_image  # noqa: E402

T_MAX = 20
THRESH = (0.85, 0.90)
PROB_THR = 0.49


def iou(a, b, valid=None):
    if valid is not None:
        a, b = a & valid, b & valid
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def build_predictor(checkpoint, device, target_size):
    # torch 2.6 defaults to weights_only=True, which cannot unpickle the stored config
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    # eval_ritm=False is the SimpleClick path (the flag only forces use_rgb_conv for RITM)
    model = load_model(state["config"], False)
    model.load_state_dict(state["state_dict"], strict=True)
    for p in model.parameters():
        p.requires_grad = False
    model.to(device)
    model.eval()
    # SimpleClick interpolates the ViT positional embedding to the ZoomIn crop size
    interpolate_pos_embed_inference(model.backbone, target_size, device)
    zoom_in = ZoomIn(skip_clicks=-1, target_size=target_size)
    # published config: NoBRS + ZoomIn + horizontal-flip TTA
    return BasePredictor(model, device, zoom_in=zoom_in, with_flip=True,
                         optimize_after_n_clicks=1)


def next_click_xy(strategy, pred_mask, gt, probs, rng, clicker, pts=None):
    """Return (x, y) for the next click, or None to stop. Mirrors run_noc.py rules."""
    H, W = pred_mask.shape
    if strategy == "fps":
        # farthest-point exploration, byte-for-byte the rule run_noc.py uses: the
        # strongest label-free baseline in our Table 4, so SimpleClick gets it too.
        region = (ndimage.binary_dilation(pred_mask, iterations=5) if pred_mask.any()
                  else np.ones((H, W), bool))
        if not region.any():
            region = np.ones((H, W), bool)
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
    if strategy == "oracle":
        # SimpleClick's own protocol: DT centre of the largest FN/FP region
        if not (pred_mask ^ gt).any():
            return None
        c = clicker._get_next_click(pred_mask)
        return int(c.coords[1]), int(c.coords[0])
    if strategy == "oracle_ours":
        # OUR oracle rule (run_noc.py `oracle`): DT-deepest point of the LARGEST
        # connected component of the symmetric difference, so that the frozen-SAM and
        # SimpleClick upper bounds are produced by an identical policy.
        err = pred_mask ^ gt
        if not err.any():
            return None
        lab, n = ndimage.label(err)
        if n == 0:
            return None
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        comp = lab == (int(np.argmax(sizes)) + 1)
        dt = ndimage.distance_transform_edt(comp)
        y, x = np.unravel_index(int(np.argmax(dt)), dt.shape)
        return int(x), int(y)
    if strategy == "random":
        return int(rng.integers(W)), int(rng.integers(H))
    if strategy == "boundary":
        if not pred_mask.any():
            return int(rng.integers(W)), int(rng.integers(H))
        band = (ndimage.binary_dilation(pred_mask, iterations=3)
                ^ ndimage.binary_erosion(pred_mask, iterations=3))
        ys, xs = np.nonzero(band)
        if not len(ys):
            return int(rng.integers(W)), int(rng.integers(H))
        k = rng.integers(len(ys))
        return int(xs[k]), int(ys[k])
    if strategy == "entropy":
        p = np.clip(probs, 1e-12, 1 - 1e-12)
        ent = (-(p * np.log(p) + (1 - p) * np.log(1 - p))).astype(np.float32)
        region = pred_mask | (ent > 0.3)
        s = np.where(region, ent, 0.0)
        if not region.any() or float(s.max()) <= 0:
            return int(rng.integers(W)), int(rng.integers(H))
        thr = np.quantile(s[region], 0.9)
        lab, n = ndimage.label(s >= max(thr, 1e-9))
        if n == 0:
            y, x = np.unravel_index(int(np.argmax(s)), s.shape)
            return int(x), int(y)
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        comp = lab == (int(np.argmax(sizes)) + 1)
        ys, xs = np.nonzero(comp)
        w = s[ys, xs]
        cy = float((ys * w).sum() / w.sum())
        cx = float((xs * w).sum() / w.sum())
        d2 = (ys - cy) ** 2 + (xs - cx) ** 2
        j = int(np.argmin(d2))
        return int(xs[j]), int(ys[j])
    raise ValueError(strategy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--strategy", required=True,
                    choices=["oracle", "oracle_ours", "random", "boundary", "entropy",
                             "fps"])
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # published `--eval-mode=cvpr` crop sizes
    tsize = (672, 672) if args.dataset.lower() == "davis" else (448, 448)
    predictor = build_predictor(args.checkpoint, device, tsize)
    print(f"[{args.dataset}/{args.strategy}] predictor ready, zoom_in target={tsize}",
          flush=True)

    rng = np.random.default_rng(args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        with open(args.out, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                done.add(r["id"])
    mode = "a" if done else "w"
    fout = open(args.out, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fout, fieldnames=[
        "dataset", "id", "strategy", "n_clicks_85", "n_clicks_90",
        "final_iou", "hit_rate", "iou_traj"])
    if mode == "w":
        writer.writeheader()

    n_done = 0
    for inst_id, img_path, gt, void in load_instances(args.data_root, args.manifest):
        if args.limit and n_done >= args.limit:
            break
        n_done += 1
        if inst_id in done:
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        gt, void = align_gt_to_image(gt, void, image)
        valid = ~void if void is not None else None
        crng = np.random.default_rng(args.seed * 99_991 + zlib.crc32(str(inst_id).encode()))

        clicker = Clicker(gt_mask=gt.astype(np.int32))
        predictor.set_input_image(image)

        cx, cy = first_click(gt)                      # identical to run_noc.py click 1
        clicker.add_click(Click(is_positive=True, coords=(cy, cx)))

        traj, hits, n85, n90 = [], [], None, None
        pts = [(cx, cy)]                 # accumulated click coords, for fps
        pred_mask = np.zeros_like(gt, dtype=bool)
        with torch.no_grad():
            for t in range(1, T_MAX + 1):
                probs = predictor.get_prediction(clicker)
                pred_mask = probs > PROB_THR
                cur = iou(pred_mask, gt, valid)
                traj.append(round(cur, 4))
                if n85 is None and cur >= THRESH[0]:
                    n85 = t
                if n90 is None and cur >= THRESH[1]:
                    n90 = t
                if n90 is not None or t == T_MAX:
                    break
                nxt = next_click_xy(args.strategy, pred_mask, gt, probs, crng, clicker,
                                    pts=pts)
                if nxt is None:
                    break
                nx = int(np.clip(nxt[0], 0, gt.shape[1] - 1))
                ny = int(np.clip(nxt[1], 0, gt.shape[0] - 1))
                hits.append(bool((pred_mask ^ gt)[ny, nx]))
                pts.append((nx, ny))
                # Polarity always comes from the GT label at the clicked pixel, the
                # same simulated-user rule run_noc.py uses. For `oracle` this agrees
                # with _get_next_click's own is_positive (FN -> +, FP -> -).
                clicker.add_click(Click(is_positive=bool(gt[ny, nx]), coords=(ny, nx)))

        writer.writerow(dict(
            dataset=args.dataset, id=inst_id, strategy=args.strategy,
            n_clicks_85=n85 if n85 is not None else T_MAX + 1,
            n_clicks_90=n90 if n90 is not None else T_MAX + 1,
            final_iou=traj[-1],
            hit_rate=round(float(np.mean(hits)), 4) if hits else "",
            iou_traj=json.dumps(traj)))
        fout.flush()
        if n_done % 25 == 0:
            print(f"[{args.dataset}/{args.strategy}] {n_done} done", flush=True)

    fout.close()
    print(f"[{args.dataset}/{args.strategy}] SC_NOC_FINISHED n={n_done}", flush=True)


if __name__ == "__main__":
    main()
