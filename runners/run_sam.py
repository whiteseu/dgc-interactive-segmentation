"""SAM (ViT-B/L/H) inference runner: instance loading, first click and GT alignment.

Per instance: the distance-transform maximum of the ground-truth mask is the first click
(the standard simulated-user protocol); one multimask pass returns three candidates with
their predicted IoU and stability scores; the working mask is the highest-scoring
candidate. The first click is then jittered N times within a disk of radius
jitter_frac * sqrt(area of the working mask) -- a quantity internal to the method that
never touches the ground truth -- and each jittered prompt is re-run.

Dumped fields:
base_mask bool[H,W] / cand_masks bool[3,H,W] / pred_iou f32[3] / stability f32[3]
/ jitter_masks bool[N,H,W] / gt_mask bool[H,W]

Usage:
  python run_sam.py --model-type vit_b --checkpoint weights/sam_vit_b_01ec64.pth \
      --dataset grabcut --data-root datasets --dump-root dumps/sam_vit_b --n-jitter 10
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy import ndimage

from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.amg import calculate_stability_score


def load_binary_mask(path) -> np.ndarray:
    """Handle the three mask encodings in use: palette indices (mode P, 0/1/2... for
    multiple objects), 0/1 greyscale and 0/255 greyscale.

    convert("L") must not be used -- it maps palette indices to luminance, turning DAVIS's
    index 1 into 38. Threshold rule: only when a value >= 128 is present is the mask read
    as 0/255 greyscale with a threshold of 127; otherwise any value > 0 is foreground
    (135 DAVIS frames are indexed {0, 2})."""
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr.max(-1)
    return arr > (127 if arr.max() >= 128 else 0)


def load_instances(data_root: Path, manifest: Path):
    """Read the manifest.csv written by prep_datasets.py: columns id, img, gt, void, with
    paths relative to data_root."""
    import csv
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        gt = load_binary_mask(data_root / r["gt"])
        if gt.sum() < 100:  # skip tiny instances (also filtered in prep_datasets.py)
            continue
        void = load_binary_mask(data_root / r["void"]) if r.get("void") else None
        yield r["id"], data_root / r["img"], gt, void


def align_gt_to_image(gt, void, image):
    """Some COD/CAMO masks are 1-2 px larger or smaller than their image: nearest-neighbour
    resize the ground truth to the image size."""
    H, W = image.shape[:2]
    if gt.shape != (H, W):
        gt = np.array(Image.fromarray(gt.astype(np.uint8) * 255)
                      .resize((W, H), Image.NEAREST)) > 127
        if void is not None:
            void = np.array(Image.fromarray(void.astype(np.uint8) * 255)
                            .resize((W, H), Image.NEAREST)) > 127
    return gt, void


def first_click(gt: np.ndarray):
    """Point of the ground-truth mask farthest from its boundary (the RITM protocol).
    Returns (x, y)."""
    dist = ndimage.distance_transform_edt(gt)
    y, x = np.unravel_index(int(np.argmax(dist)), dist.shape)
    return x, y


def predict_once(predictor, point_xy, with_stability=True):
    coords = np.array([point_xy], dtype=np.float32)
    labels = np.ones(1, dtype=np.int64)
    masks, ious, low_res = predictor.predict(point_coords=coords, point_labels=labels,
                                             multimask_output=True, return_logits=True)
    if with_stability:
        # Same convention as SAM's automatic mask generator: stability is computed on
        # the low-resolution logits
        stability = calculate_stability_score(torch.as_tensor(low_res), 0.0, 1.0).numpy()
        stability = stability.astype(np.float32)
    else:
        stability = None
    return (masks > 0.0), ious.astype(np.float32), stability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-type", required=True, choices=["vit_b", "vit_l", "vit_h"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True, help="name of the dump subdirectory")
    ap.add_argument("--manifest", type=Path, required=True, help="path to manifest.csv")
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dump-root", type=Path, required=True)
    ap.add_argument("--n-jitter", type=int, default=10)
    ap.add_argument("--jitter-frac", type=float, default=0.1)
    ap.add_argument("--prompt-tag", default="click1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="run only the first N instances (smoke test)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
    predictor = SamPredictor(sam)
    rng = np.random.default_rng(args.seed)
    out_dir = args.dump_root / args.dataset / args.prompt_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done, ious_vs_gt = 0, []
    for inst_id, img_path, gt, void in load_instances(args.data_root, args.manifest):
        if args.limit and n_done >= args.limit:
            break
        out_path = out_dir / f"{inst_id}.npz"
        if out_path.exists():
            n_done += 1
            continue
        image = np.array(Image.open(img_path).convert("RGB"))
        gt, void = align_gt_to_image(gt, void, image)
        predictor.set_image(image)
        cx, cy = first_click(gt)

        cand_masks, pred_iou, stability = predict_once(predictor, (cx, cy))
        best = int(np.argmax(pred_iou))
        base_mask = cand_masks[best]

        radius = max(args.jitter_frac * float(np.sqrt(base_mask.sum())), 3.0)
        jitters = []
        H, W = gt.shape
        for _ in range(args.n_jitter):
            ang = rng.uniform(0, 2 * np.pi)
            r = radius * np.sqrt(rng.uniform())
            jx = int(np.clip(cx + r * np.cos(ang), 0, W - 1))
            jy = int(np.clip(cy + r * np.sin(ang), 0, H - 1))
            jm, jiou, _ = predict_once(predictor, (jx, jy), with_stability=False)
            jitters.append(jm[int(np.argmax(jiou))])

        arrays = dict(base_mask=base_mask, cand_masks=cand_masks,
                      pred_iou=pred_iou, stability=stability,
                      jitter_masks=np.stack(jitters), gt_mask=gt)
        if void is not None:
            arrays["void_mask"] = void
        np.savez_compressed(out_path, **arrays)

        valid = ~void if void is not None else np.ones_like(gt)
        inter = np.logical_and(base_mask & valid, gt & valid).sum()
        union = np.logical_or(base_mask & valid, gt & valid).sum()
        ious_vs_gt.append(inter / union if union else 0.0)
        n_done += 1
        if n_done % 50 == 0:
            print(f"[{args.dataset}] {n_done} done, running mIoU={np.mean(ious_vs_gt):.4f}",
                  flush=True)

    print(f"[{args.dataset}] FINISHED n={n_done}, 1-click mIoU={np.mean(ious_vs_gt):.4f}",
          flush=True)


if __name__ == "__main__":
    main()
