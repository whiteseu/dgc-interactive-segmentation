"""SAM 2.1 image-mode inference runner.

Follows exactly the protocol of run_sam.py and reuses its instance loading and jitter
logic, so the two backbones differ only in the segmenter. The dumped field contract is the
same (base_mask / cand_masks / pred_iou / stability / jitter_masks / gt_mask [/ void_mask]);
SAM 2 likewise returns three candidates with predicted IoU and low-resolution logits.

Usage:
  python run_sam2.py --config configs/sam2.1/sam2.1_hiera_l.yaml \
      --checkpoint weights/sam2.1_hiera_large.pt --dataset grabcut \
      --manifest datasets/GrabCut/manifest.csv --data-root datasets \
      --dump-root dumps/sam2_hiera_l
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from run_sam import load_instances, first_click, align_gt_to_image  # noqa: E402

from sam2.build_sam import build_sam2  # noqa: E402
from sam2.sam2_image_predictor import SAM2ImagePredictor  # noqa: E402
from sam2.utils.amg import calculate_stability_score  # noqa: E402


def predict_once(predictor, point_xy, with_stability=True):
    coords = np.array([point_xy], dtype=np.float32)
    labels = np.ones(1, dtype=np.int64)
    masks, ious, low_res = predictor.predict(point_coords=coords, point_labels=labels,
                                             multimask_output=True, return_logits=True)
    if with_stability:
        stability = calculate_stability_score(torch.as_tensor(low_res), 0.0, 1.0).numpy()
        stability = stability.astype(np.float32)
    else:
        stability = None
    return (masks > 0.0), ious.astype(np.float32), stability


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dump-root", type=Path, required=True)
    ap.add_argument("--n-jitter", type=int, default=10)
    ap.add_argument("--jitter-frac", type=float, default=0.1)
    ap.add_argument("--prompt-tag", default="click1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=100,
                    help="skip instances with fewer than this many foreground "
                         "pixels; 0 dumps every instance in the manifest")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_sam2(args.config, args.checkpoint, device=device)
    predictor = SAM2ImagePredictor(model)
    rng = np.random.default_rng(args.seed)
    out_dir = args.dump_root / args.dataset / args.prompt_tag
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done, ious_vs_gt = 0, []
    for inst_id, img_path, gt, void in load_instances(args.data_root, args.manifest, args.min_area):
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
