"""Second dump pass: the selected candidate's low-resolution logits and a box-perturbation
mask set, which the entropy and SAM-U-style baselines need.

Same protocol as the main dump (same first click, same candidate selection); the forward
pass is deterministic, so the working mask reproduces the base_mask already on disk. Per
instance: one set_image, one multimask forward, and five box-perturbation forwards, which
share the image embedding and so cost only the decoder.

Writes dumps/<model>/<dataset>/click1_logits/<id>.npz:
  logit_low f16[256,256]  low-resolution logits of the selected candidate
  pred_iou  f32[3]
  box_masks bool[5,H,W]   five masks from bbox perturbations (+/-5% in scale and offset),
                          each a single-output box prompt
  box_base  f32[4]        bbox of the base mask, (x0, y0, x1, y1)

Usage:
  python run_logits.py --backend sam --model-type vit_b --checkpoint weights/sam_vit_b.pth \
      --dataset grabcut --manifest datasets/GrabCut/manifest.csv --data-root datasets \
      --dump-root dumps/sam_vit_b
  python run_logits.py --backend sam2 --config configs/sam2.1/sam2.1_hiera_l.yaml \
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


def build_predictor(args, device):
    if args.backend == "sam":
        from segment_anything import sam_model_registry, SamPredictor
        sam = sam_model_registry[args.model_type](checkpoint=args.checkpoint).to(device)
        return SamPredictor(sam)
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    return SAM2ImagePredictor(build_sam2(args.config, args.checkpoint, device=device))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["sam", "sam2"])
    ap.add_argument("--model-type", default="")
    ap.add_argument("--config", default="")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--dump-root", type=Path, required=True)
    ap.add_argument("--n-box", type=int, default=5)
    ap.add_argument("--box-frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min-area", type=int, default=100,
                    help="skip instances with fewer than this many foreground "
                         "pixels; 0 dumps every instance in the manifest")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_predictor(args, device)
    rng = np.random.default_rng(args.seed)
    out_dir = args.dump_root / args.dataset / "click1_logits"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_done = 0
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

        coords = np.array([(cx, cy)], dtype=np.float32)
        labels = np.ones(1, dtype=np.int64)
        masks, ious, low_res = predictor.predict(point_coords=coords, point_labels=labels,
                                                 multimask_output=True, return_logits=True)
        sel = int(np.argmax(ious))
        base = np.asarray(masks[sel]) > 0.0
        logit_low = np.asarray(low_res[sel], dtype=np.float16)

        H, W = gt.shape
        box_masks = np.zeros((0, H, W), bool)
        box_base = np.zeros(4, np.float32)
        if base.any():
            ys, xs = np.nonzero(base)
            x0, x1 = float(xs.min()), float(xs.max())
            y0, y1 = float(ys.min()), float(ys.max())
            box_base = np.array([x0, y0, x1, y1], np.float32)
            bw, bh = x1 - x0 + 1, y1 - y0 + 1
            outs = []
            for _ in range(args.n_box):
                jit = rng.uniform(-args.box_frac, args.box_frac, 4)
                bx = np.array([x0 + jit[0] * bw, y0 + jit[1] * bh,
                               x1 + jit[2] * bw, y1 + jit[3] * bh], np.float32)
                bx[0], bx[2] = np.clip(bx[0], 0, W - 2), np.clip(bx[2], 1, W - 1)
                bx[1], bx[3] = np.clip(bx[1], 0, H - 2), np.clip(bx[3], 1, H - 1)
                bm, _, _ = predictor.predict(box=bx, multimask_output=False,
                                             return_logits=True)
                outs.append(np.asarray(bm[0]) > 0.0)
            box_masks = np.stack(outs)

        np.savez_compressed(out_path, logit_low=logit_low,
                            pred_iou=np.asarray(ious, np.float32),
                            box_masks=box_masks, box_base=box_base)
        n_done += 1
        if n_done % 100 == 0:
            print(f"[{args.dataset}] {n_done} done", flush=True)

    print(f"[{args.dataset}] LOGITS_FINISHED n={n_done}", flush=True)


if __name__ == "__main__":
    main()
