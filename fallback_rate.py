"""How often does the Omega_D = empty fallback actually fire?

`first_d_empty` in the v2 risk_dt CSVs records whether the disagreement support was
empty at the FIRST guided click; `round_d1` is the first round at which it became
non-empty (-1 = never). Together these bound the fallback rate at the first guided
click and identify instances where disagreement never appeared at all.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv, os
import numpy as np

V2 = rf"{ROOT}/results/noc_v2"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]

for bb in BB:
    empty_first, never, n = 0, 0, 0
    for ds in SETS:
        p = f"{V2}/noc_{bb}_risk_dt_{ds}.csv"
        if not os.path.exists(p):
            print("missing", p); continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            n += 1
            if str(r.get("first_d_empty", "")) == "1":
                empty_first += 1
            if str(r.get("round_d1", "-1")) in ("-1", ""):
                never += 1
    print(f"{bb:6s} n={n:5d}  D empty at first guided click: {empty_first:4d} "
          f"({100*empty_first/max(n,1):.2f}%)   D never non-empty: {never} "
          f"({100*never/max(n,1):.2f}%)")
print("FALLBACK_DONE")
