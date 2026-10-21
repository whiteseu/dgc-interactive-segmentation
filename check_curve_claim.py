"""Does DGC really hold a higher mean IoU than every non-oracle baseline from click 2 on?

Fig. 3 plots only DGC, farthest-point, random and the oracle, but the text claims the
ordering against all non-oracle baselines. Check the claim directly on the pooled
trajectories, per backbone, so the sentence either keeps its scope or is narrowed to what
was verified.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import json
import os

import numpy as np

V2 = f"{ROOT}/results/noc_v2"
FULL = f"{ROOT}/results/noc_full"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
POL = {"DGC": "noc_{bb}_risk_dt_{ds}.csv",
       "Farthest-point": "noc_{bb}_fps_{ds}.csv",
       "Random": "noc_{bb}_random_{ds}.csv",
       "Entropy": "noc_{bb}_entropy_{ds}.csv",
       "Pert-MI 0.1": "noc_{bb}_bald_{ds}.csv",
       "Pert-MI 0.2": "noc_{bb}_bald_{ds}_rho20.csv",
       "Boundary": "noc_{bb}_boundary_{ds}.csv"}
T = 20


def curve(bb, pat):
    trajs = []
    for ds in SETS:
        root = FULL if ds == "cocomval" else V2
        p = f"{root}/{pat.format(bb=bb, ds=ds)}"
        assert os.path.exists(p), p
        for r in csv.DictReader(open(p, encoding="utf-8")):
            t = json.loads(r["iou_traj"])
            t = t + [t[-1]] * (T - len(t))
            trajs.append(t[:T])
    assert len(trajs) == 1621, f"{pat}: n={len(trajs)}"
    return np.array(trajs).mean(0)


for bb in BB:
    d = curve(bb, POL["DGC"])
    bad = []
    for name, pat in POL.items():
        if name == "DGC":
            continue
        c = curve(bb, pat)
        losses = [k + 1 for k in range(1, T) if d[k] <= c[k]]   # clicks 2..20
        if losses:
            bad.append(f"{name} at clicks {losses[:6]}")
    print(f"{bb:6s} " + ("holds against all six" if not bad else "FAILS: " + "; ".join(bad)))
print("CURVE_CLAIM_DONE")
