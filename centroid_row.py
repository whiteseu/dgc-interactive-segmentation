"""Pooled NoC@90 for the centroid variant (strategy 'risk' = weighted centroid of the
largest component) so it can be judged as a second ablation row in Table 1, parallel to
Random-in-Omega_D. Also reports NoF per backbone for every Table 1 row, to see whether a
failure-rate column is worth adding.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import os

import numpy as np

V2 = f"{ROOT}/results/noc_v2"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
ROWS = [("DGC", "noc_{bb}_risk_dt_{ds}.csv"),
        ("Centroid-in-C (abl.)", "noc_{bb}_risk_{ds}.csv"),
        ("Random-in-OmegaD (abl.)", "noc_{bb}_random_in_d_{ds}.csv"),
        ("Farthest-point", "noc_{bb}_fps_{ds}.csv"),
        ("Random", "noc_{bb}_random_{ds}.csv"),
        ("Entropy", "noc_{bb}_entropy_{ds}.csv"),
        ("Pert-MI r0.1", "noc_{bb}_bald_{ds}.csv"),
        ("Pert-MI r0.2", "noc_{bb}_bald_{ds}_rho20.csv"),
        ("Boundary", "noc_{bb}_boundary_{ds}.csv"),
        ("Oracle", "noc_{bb}_oracle_{ds}.csv")]


def load(bb, pat):
    v = []
    for ds in SETS:
        p = f"{V2}/{pat.format(bb=bb, ds=ds)}"
        if not os.path.exists(p):
            return None
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                v.append(min(int(r["n_clicks_90"]), 21))
            except (ValueError, KeyError):
                v.append(21)
    return np.array(v, float)


print(f"{'row':26s}" + "".join(f"{b:>9s}" for b in BB) + "   |" +
      "".join(f"{b+' NoF':>11s}" for b in BB))
dgc = {bb: load(bb, ROWS[0][1]) for bb in BB}
for name, pat in ROWS:
    noc, nof = [], []
    for bb in BB:
        v = load(bb, pat)
        if v is None:
            noc.append("     NA  "); nof.append("      NA   "); continue
        assert len(v) == len(dgc[bb]), (name, bb, len(v), len(dgc[bb]))
        noc.append(f"{v.mean():9.2f}"); nof.append(f"{(v >= 21).mean()*100:10.0f}%")
    print(f"{name:26s}" + "".join(noc) + "   |" + "".join(nof))

print("\ncentroid penalty vs DGC (pooled NoC@90):")
for bb in BB:
    c, a = load(bb, ROWS[1][1]), dgc[bb]
    print(f"  {bb:7s} {c.mean():6.2f} vs {a.mean():6.2f}   +{c.mean()-a.mean():.2f}")
print("CENTROID_ROW_DONE")
