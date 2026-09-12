"""The n_clicks_90 column is what the runner recorded live; iou_traj is a rounded log,
and rounding lets a 0.8999x land on 0.90, which is why my trajectory recompute sat
1-3 instances low per cell and printed four table cells 0.01 too low. The column is
authoritative, so redo the whole Table 1 CI matrix on it -- including the borderline
Random-in-Omega_D row the caption's universal depends on."""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv, os
import numpy as np

V2 = f"{ROOT}/results/noc_v2"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
ROWS = [("Random-in-OmegaD", "noc_{bb}_random_in_d_{ds}.csv"),
        ("Farthest-point",   "noc_{bb}_fps_{ds}.csv"),
        ("Random",           "noc_{bb}_random_{ds}.csv"),
        ("Entropy",          "noc_{bb}_entropy_{ds}.csv"),
        ("Pert-MI rho0.1",   "noc_{bb}_bald_{ds}.csv"),
        ("Pert-MI rho0.2",   "noc_{bb}_bald_{ds}_rho20.csv"),
        ("Boundary",         "noc_{bb}_boundary_{ds}.csv"),
        ("Oracle",           "noc_{bb}_oracle_{ds}.csv")]

def noc(bb, pat):
    out = {}
    for ds in SETS:
        p = f"{V2}/{pat.format(bb=bb, ds=ds)}"
        if not os.path.exists(p):
            return None
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                out[f"{ds}:{r['id']}"] = min(int(r["n_clicks_90"]), 21)
            except (ValueError, KeyError):
                out[f"{ds}:{r['id']}"] = 21
    return out

rng = np.random.default_rng(0)
fails = []
for bb in BB:
    A = noc(bb, "noc_{bb}_risk_dt_{ds}.csv")
    print(f"\n{bb}: DGC {np.mean(list(A.values())):.2f}  n={len(A)}")
    for name, pat in ROWS:
        B = noc(bb, pat)
        if B is None:
            print(f"  {name:17s} MISSING"); fails.append((bb, name)); continue
        ids = sorted(set(A) & set(B))
        assert len(ids) == len(A) == len(B), (bb, name, len(ids), len(A), len(B))
        d = np.array([B[i] - A[i] for i in ids], float)
        lo, hi = np.percentile(d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1),
                               [2.5, 97.5])
        ok = (lo > 0) or (hi < 0)
        if not ok:
            fails.append((bb, name, f"[{lo:+.3f},{hi:+.3f}]"))
        print(f"  {name:17s} {np.mean([B[i] for i in ids]):6.2f}  gap {d.mean():+6.3f}  "
              f"CI [{lo:+.3f},{hi:+.3f}]  excl0={ok}")
print("\nCIs that do NOT exclude zero:", fails if fails else "none")
print("TAB1_COL_CI_DONE")
