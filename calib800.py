"""The intro's predicted-IoU vs true-IoU pair after the first click, on the full 800 split.

Read straight from the first-click dumps rather than re-running SAM: base_mask is already
the highest-scoring candidate and pred_iou holds the three scores, so the selected
candidate's predicted IoU is max(pred_iou) and its true IoU is IoU(base_mask, gt_mask) --
the same quantities the NoC protocol sees at round 1.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import glob
import os

import numpy as np

ROOT = f"{ROOT}/dumps"
SIX = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = {"vit_b": "sam_vit_b", "vit_l": "sam_vit_l", "vit_h": "sam_vit_h",
      "sam2": "sam2_hiera_l"}

for d in BB.values():
    n = len(glob.glob(f"{ROOT}/{d}/cocomval/click1/*.npz"))
    assert n == 800, f"{d}: {n} cocomval dumps, expected 800"

print(f"{'bb':7s}{'n':>6s}{'pred IoU':>10s}{'true IoU':>10s}{'gap':>8s}   per-dataset (vit_b)")
for bb, d in BB.items():
    per = {}
    ps, ts = [], []
    for ds in SIX:
        a, b = [], []
        for p in sorted(glob.glob(f"{ROOT}/{d}/{ds}/click1/*.npz")):
            with np.load(p) as z:
                gt = z["gt_mask"].astype(bool)
                base = z["base_mask"].astype(bool)
                valid = (~z["void_mask"].astype(bool)) if "void_mask" in z.files \
                    else np.ones_like(gt)
                a.append(float(z["pred_iou"].max()))
            u = ((base | gt) & valid).sum()
            b.append(float((base & gt & valid).sum() / u) if u else 1.0)
        per[ds] = (np.mean(a), np.mean(b), len(a))
        ps += a
        ts += b
    line = f"{bb:7s}{len(ps):6d}{np.mean(ps):10.3f}{np.mean(ts):10.3f}{np.mean(ps) - np.mean(ts):8.3f}"
    if bb == "vit_b":
        line += "   " + "  ".join(f"{k}:{v[0]:.2f}/{v[1]:.2f}" for k, v in per.items())
    print(line)
print("CALIB800_DONE")
