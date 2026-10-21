"""Does 'X% of DGC's clicks fall inside E' mean instance-mean or click-pooled?

hit_rate in each CSV is a per-instance mean over that instance's guided clicks, and
errhit_v2.py averages those instance means unweighted. The paper's wording ("42-55% of
DGC's clicks fall inside E") literally claims the click-pooled quantity. Compute both;
if they differ materially the sentence has to name which one it is.

Guided clicks per instance = rounds executed - 1, where the loop stops at n90 (capped at
T_MAX=20), so it is min(n90, 20) - 1.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import os

import numpy as np

V2 = rf"{ROOT}/results/noc_v2"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
STRATS = ["risk_dt", "fps", "bald", "random"]
T_MAX = 20


def both(bb, strat):
    inst, num, den = [], 0.0, 0
    for ds in SETS:
        p = f"{V2}/noc_{bb}_{strat}_{ds}.csv"
        if not os.path.exists(p):
            return None
        for r in csv.DictReader(open(p, encoding="utf-8")):
            h = r.get("hit_rate", "")
            if h in ("", None):
                continue
            h = float(h)
            k = min(int(r["n_clicks_90"]), T_MAX) - 1     # guided clicks on this instance
            if k <= 0:
                continue
            inst.append(h)
            num += h * k
            den += k
    return np.mean(inst), num / den, len(inst), den


print(f"{'strategy':10s} {'backbone':9s} {'instance-mean':>14s} {'click-pooled':>13s} "
      f"{'diff':>6s}  n_inst  n_clicks")
for s in STRATS:
    for bb in BB:
        r = both(bb, s)
        if r is None:
            continue
        im, cp, ni, nc = r
        print(f"{s:10s} {bb:9s} {im:14.3f} {cp:13.3f} {cp-im:+6.3f}  {ni:6d}  {nc:8d}")
    print()
print("ERRHIT_POOLED_DONE")
