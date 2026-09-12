"""Three claims from the review that only data can settle.

(1) "desk-reject level risk": our pooled NoC is far above published NoC. Published
    numbers are PER-BENCHMARK and ORACLE-clicked. So print our per-benchmark oracle
    NoC@90/@85 -- if GrabCut/Berkeley land near published values the pipeline is fine
    and the gap is entirely pooling + camouflage + non-oracle clicks.
(2) The suggested contribution-2 rewrite asserts Random-in-Omega_D "still outperforms
    every external baseline on all four backbones". Pooled that is visible in Table 1;
    per cell it is not checked. Check all 24.
(3) The 25-37% never-reach-90% share: is it concentrated in CAMO/CHAMELEON?
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv, os
import numpy as np

V2 = f"{ROOT}/results/noc_v2"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
EXT = {"fps": "noc_{bb}_fps_{ds}.csv", "random": "noc_{bb}_random_{ds}.csv",
       "entropy": "noc_{bb}_entropy_{ds}.csv", "bald": "noc_{bb}_bald_{ds}.csv",
       "bald20": "noc_{bb}_bald_{ds}_rho20.csv", "boundary": "noc_{bb}_boundary_{ds}.csv"}

def cell(bb, ds, pat, col="n_clicks_90"):
    p = f"{V2}/{pat.format(bb=bb, ds=ds)}"
    if not os.path.exists(p):
        return None
    v = []
    for r in csv.DictReader(open(p, encoding="utf-8")):
        try:
            v.append(min(int(r[col]), 21))
        except (ValueError, KeyError):
            v.append(21)
    return np.array(v, float)

print("=== (1) PER-BENCHMARK ORACLE NoC (our pipeline) ===")
for metric, col in [("NoC@90", "n_clicks_90"), ("NoC@85", "n_clicks_85")]:
    print(f"\n{metric}   " + "".join(f"{s:>11}" for s in SETS) + "   n")
    for bb in BB:
        vals, n = [], 0
        for ds in SETS:
            v = cell(bb, ds, "noc_{bb}_oracle_{ds}.csv", col)
            vals.append("     NA   " if v is None else f"{v.mean():9.2f} ")
            n += 0 if v is None else len(v)
        print(f"{bb:8s}" + "".join(vals) + f"  {n}")

print("\n=== (2) Random-in-Omega_D vs each external baseline, per cell ===")
lose = []
for bb in BB:
    for ds in SETS:
        r = cell(bb, ds, "noc_{bb}_random_in_d_{ds}.csv")
        worst = max((cell(bb, ds, p).mean() for p in EXT.values()))
        best_ext = min((cell(bb, ds, p).mean() for p in EXT.values()))
        if r.mean() >= best_ext:
            lose.append((bb, ds, round(r.mean(), 2), round(best_ext, 2)))
print(f"cells where Random-in-Omega_D does NOT beat the best external baseline: "
      f"{len(lose)}/24")
for x in lose:
    print("   ", x)

print("\n=== (3) share never reaching 90% IoU in 20 clicks, by benchmark (DGC) ===")
print(f"{'':8s}" + "".join(f"{s:>11}" for s in SETS) + "     pooled")
for bb in BB:
    row, allv = [], []
    for ds in SETS:
        v = cell(bb, ds, "noc_{bb}_risk_dt_{ds}.csv")
        row.append(f"{(v >= 21).mean() * 100:9.1f} ")
        allv.append(v)
    a = np.concatenate(allv)
    print(f"{bb:8s}" + "".join(row) + f"  {(a >= 21).mean() * 100:8.1f}")
cnt = {ds: len(cell("vit_b", ds, "noc_{bb}_risk_dt_{ds}.csv")) for ds in SETS}
tot = sum(cnt.values())
print("\ninstances per benchmark:", cnt, f"total {tot}")
print(f"CAMO+CHAMELEON share of instances: "
      f"{(cnt['camo'] + cnt['chameleon']) / tot * 100:.1f}%")
print("THREECLAIMS_DONE")
