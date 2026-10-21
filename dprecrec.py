"""Direct localization quality of the native disagreement support, six benchmarks,
first-click dumps, all four backbones.

  E = b XOR gt (restricted to valid pixels), Omega_D = {u : D(u) > 0}
  precision (pixel-pooled, whole image) = sum|Omega_D & E| / sum|Omega_D|
  precision within b|gt                 = same, both restricted to b|gt
  recall  (per instance, then median/mean over instances with |E|>0)
                                        = |Omega_D & E| / |E|
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import os
from pathlib import Path

import numpy as np

ROOT = Path(ROOT)
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BBS = {"vit_b": "sam_vit_b", "vit_l": "sam_vit_l", "vit_h": "sam_vit_h", "sam2": "sam2_hiera_l"}


def disag(c):
    return ((c[0] ^ c[1]) | (c[0] ^ c[2]) | (c[1] ^ c[2]))


print(f"{'bb':6s} {'n':>5s} {'prec(img)':>9s} {'prec(b|gt)':>10s} {'rec med':>8s} {'rec mean':>8s} "
      f"{'rec med(fail)':>13s}  missing")
for bb, full in BBS.items():
    inter = dsum = inter_u = dsum_u = 0
    recs, recs_fail = [], []
    missing = []
    for ds in SETS:
        tag = ROOT / "dumps" / full / ds / "click1"
        if not tag.exists():
            missing.append(ds)
            continue
        for p in sorted(tag.glob("*.npz")):
            try:
                with np.load(p) as z:
                    d = {k: z[k] for k in z.files}
            except Exception:
                continue
            gt, base = d["gt_mask"].astype(bool), d["base_mask"].astype(bool)
            valid = ~d["void_mask"].astype(bool) if "void_mask" in d else np.ones_like(gt)
            E = (base ^ gt) & valid
            Dm = disag(d["cand_masks"].astype(bool)) & valid
            U = (base | gt) & valid
            inter += int((Dm & E).sum()); dsum += int(Dm.sum())
            inter_u += int((Dm & E & U).sum()); dsum_u += int((Dm & U).sum())
            ne = int(E.sum())
            if ne > 0:
                r = (Dm & E).sum() / ne
                recs.append(r)
                iou = (base & gt & valid).sum() / max(((base | gt) & valid).sum(), 1)
                if iou < 0.85:
                    recs_fail.append(r)
    recs = np.array(recs)
    print(f"{bb:6s} {len(recs):5d} {inter/max(dsum,1):9.3f} {inter_u/max(dsum_u,1):10.3f} "
          f"{np.median(recs):8.3f} {recs.mean():8.3f} {np.median(recs_fail) if recs_fail else float('nan'):13.3f}  "
          f"{missing or '-'}")
print("DPRECREC_DONE")
