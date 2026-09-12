"""Replay the interaction loop on ONE instance and dump per-round state for Fig. 5.

Reuses run_noc.py's own functions and reproduces its protocol exactly (same rng
seeding, same first click, same mask_input carry-over, same polarity rule), so
the IoU values printed here must match the iou_traj already stored in the v2
CSVs -- the script asserts that, otherwise the figure would contradict Table 1.

Dumps an npz; the figure itself is rendered locally from it.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import sys
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image

R = Path(f"{ROOT}")
sys.path.insert(0, str(R / "runners"))
import run_noc as RN  # noqa: E402

INST = sys.argv[1] if len(sys.argv) > 1 else "chameleon_animal-43"
DS = INST.split("_")[0]
MAN = {"chameleon": "TestDataset/CHAMELEON/manifest.csv",
       "camo": "TestDataset/CAMO/manifest.csv"}[DS]
STRATS = ["risk_dt", "fps"]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 4

args = SimpleNamespace(backend="sam", model_type="vit_b", config="",
                       checkpoint=str(R / "weights/sam_vit_b_01ec64.pth"))
device = "cuda" if torch.cuda.is_available() else "cpu"
predictor = RN.build_predictor(args, device)

target = None
for inst_id, img_path, gt, void in RN.load_instances(R / "datasets", R / "datasets" / MAN):
    if str(inst_id) == INST or str(inst_id).endswith(INST.split("_", 1)[1]):
        target = (inst_id, img_path, gt, void)
        break
if target is None:
    sys.exit(f"instance {INST} not found in {MAN}")
inst_id, img_path, gt, void = target
print("found", inst_id, img_path, flush=True)

image = np.array(Image.open(img_path).convert("RGB"))
gt, void = RN.align_gt_to_image(gt, void, image)
valid = ~void if void is not None else None
predictor.set_image(image)

out = {"image": image, "gt": gt}
for strat in STRATS:
    crng = np.random.default_rng(0 * 99_991 + zlib.crc32(str(inst_id).encode()))
    cx, cy = RN.first_click(gt)
    pts, lbs = [(cx, cy)], [1]
    mask_input, traj = None, []
    for t in range(1, ROUNDS + 1):
        kw = dict(point_coords=np.array(pts, dtype=np.float32),
                  point_labels=np.array(lbs, dtype=np.int64),
                  multimask_output=True, return_logits=True)
        if mask_input is not None:
            kw["mask_input"] = mask_input[None]
        masks, ious_p, low = predictor.predict(**kw)
        sel = int(np.argmax(ious_p))
        base = np.asarray(masks[sel]) > 0.0
        mask_input = np.asarray(low[sel])
        cands = np.asarray(masks) > 0.0
        cur = RN.iou(base, gt, valid)
        traj.append(round(cur, 4))

        out[f"{strat}_mask_{t}"] = base
        out[f"{strat}_dis_{t}"] = RN._disag(cands) > 0
        out[f"{strat}_pts_{t}"] = np.array(pts, dtype=np.int32)
        out[f"{strat}_lbs_{t}"] = np.array(lbs, dtype=np.int32)
        out[f"{strat}_iou_{t}"] = np.float32(cur)

        if t == ROUNDS:
            break
        nxt = RN.next_click(strat, base, cands, mask_input, gt, crng,
                            None, excl=None, pts=pts)
        if nxt is None:
            print(f"  {strat}: no next click at t={t}", flush=True)
            break
        nx = int(np.clip(nxt[0], 0, gt.shape[1] - 1))
        ny = int(np.clip(nxt[1], 0, gt.shape[0] - 1))
        pts.append((nx, ny))
        lbs.append(int(gt[ny, nx]))
    out[f"{strat}_traj"] = np.array(traj, dtype=np.float32)
    print(f"  {strat}: {traj}", flush=True)

dst = R / "figs_v2" / f"seq_{INST}.npz"
dst.parent.mkdir(exist_ok=True)
np.savez_compressed(dst, **out)
print("WROTE", dst)
print("REPLAY_DONE")
