"""How often does the jitter radius floor, not the coefficient, set the perturbation radius?

Both perturbation radii are written max(floor, coefficient * scale), and the paper states
only the coefficient. The floor dominates below a critical working-mask area:

  Perturbation-MI, rho=0.1 : max(2.0, 0.1*sqrt(|b|/pi))  ->  |b| <  400*pi = 1257 px
  Perturbation-MI, rho=0.2 : max(2.0, 0.2*sqrt(|b|/pi))  ->  |b| <  100*pi =  314 px
  Sec. 3.3 analysis probe  : max(3.0, 0.1*sqrt(|b|))     ->  |b| <         900 px

Measured on the first-click working mask from the dumps -- the mask the radius is actually
computed from at the first guided round, and the one the analysis probe used.
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
CUTS = [("Pert-MI rho=0.1", 400 * np.pi), ("Pert-MI rho=0.2", 100 * np.pi),
        ("analysis probe", 900.0)]

print(f"{'backbone':10s}{'n':>6s}{'median |b|':>12s}" +
      "".join(f"{c[0]:>18s}" for c in CUTS))
per_ds = {}
for bb, d in BB.items():
    areas, ds_of = [], []
    for ds in SIX:
        fs = sorted(glob.glob(f"{ROOT}/{d}/{ds}/click1/*.npz"))
        assert fs, f"no dumps for {d}/{ds}"
        for p in fs:
            with np.load(p) as z:
                areas.append(int(z["base_mask"].sum()))
            ds_of.append(ds)
    a = np.array(areas, float)
    per_ds[bb] = (a, np.array(ds_of))
    print(f"{bb:10s}{len(a):6d}{np.median(a):12.0f}" +
          "".join(f"{100 * (a < c[1]).mean():17.1f}%" for c in CUTS))

print("\nshare below the Perturbation-MI rho=0.1 floor, per benchmark (ViT-B)")
a, dss = per_ds["vit_b"]
for ds in SIX:
    m = dss == ds
    print(f"  {ds:11s} n={m.sum():4d}  {100 * (a[m] < 400 * np.pi).mean():5.1f}%"
          f"   median |b| = {np.median(a[m]):.0f} px")
print("FLOORSHARE_DONE")
