"""The intro quotes predicted IoU 0.88 vs true IoU 0.40 after the first click, but only on
the two camouflage sets. A reviewer can read that as cherry-picking, so compute the same
pair on every benchmark: if the standard sets show a small gap and the camouflage sets a
large one, quoting a standard set alongside makes the claim harder to dismiss.

First click = distance-transform maximum of the GT mask (identical to the NoC protocol),
multimask pass, selected candidate = highest predicted IoU -- exactly what Table 1 does at
round 1.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, f"{ROOT}/runners")
from run_sam import load_instances, first_click, align_gt_to_image  # noqa: E402
from segment_anything import sam_model_registry, SamPredictor  # noqa: E402

ROOT = Path(f"{ROOT}/datasets")
SETS = [("grabcut", "GrabCut/manifest.csv"), ("berkeley", "Berkeley/manifest.csv"),
        ("davis", "DAVIS345/manifest.csv"), ("cocomval", "COCO_MVal/manifest.csv"),
        ("camo", "TestDataset/CAMO/manifest.csv"),
        ("chameleon", "TestDataset/CHAMELEON/manifest.csv")]

dev = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=f"{ROOT}/weights/sam_vit_b_01ec64.pth").to(dev)
pred = SamPredictor(sam)

print(f"{'dataset':12s}{'n':>6s}{'pred IoU':>10s}{'true IoU':>10s}{'gap':>8s}")
allp, allt = [], []
for name, man in SETS:
    ps, ts = [], []
    for inst_id, img_path, gt, void in load_instances(ROOT, ROOT / man):
        image = np.array(Image.open(img_path).convert("RGB"))
        gt2, void2 = align_gt_to_image(gt, void, image)
        valid = ~void2 if void2 is not None else None
        pred.set_image(image)
        cx, cy = first_click(gt2)
        masks, ious_p, _ = pred.predict(point_coords=np.array([[cx, cy]], np.float32),
                                        point_labels=np.ones(1, np.int64),
                                        multimask_output=True)
        k = int(np.argmax(ious_p))
        m = np.asarray(masks[k]) > 0
        a, b = (m & gt2, m | gt2) if valid is None else (m & gt2 & valid, (m | gt2) & valid)
        ts.append(float(a.sum() / b.sum()) if b.sum() else 0.0)
        ps.append(float(ious_p[k]))
    allp += ps; allt += ts
    print(f"{name:12s}{len(ps):6d}{np.mean(ps):10.3f}{np.mean(ts):10.3f}"
          f"{np.mean(ps) - np.mean(ts):8.3f}")
print(f"{'ALL':12s}{len(allp):6d}{np.mean(allp):10.3f}{np.mean(allt):10.3f}"
      f"{np.mean(allp) - np.mean(allt):8.3f}")
print("CALIB_DONE")
