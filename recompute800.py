"""Recompute every number the paper quotes, on the full 800-instance COCO-MVal split.

Refuses to report anything until all 40 noc_full CSVs hold exactly 800 rows, so a partially
finished batch fails loudly instead of silently averaging a mixed instance set.

COCO-MVal comes from results/noc_full; the other five benchmarks are unaffected by the
<100 px filter and come from results/noc_v2.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import glob
import os

import numpy as np

V2 = f"{ROOT}/results/noc_v2"
FULL = f"{ROOT}/results/noc_full"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
ROWS = [("DGC", "noc_{bb}_risk_dt_{ds}.csv"),
        ("Farthest-point", "noc_{bb}_fps_{ds}.csv"),
        ("Random", "noc_{bb}_random_{ds}.csv"),
        ("Entropy", "noc_{bb}_entropy_{ds}.csv"),
        ("Pert-MI r0.1", "noc_{bb}_bald_{ds}.csv"),
        ("Pert-MI r0.2", "noc_{bb}_bald_{ds}_rho20.csv"),
        ("Boundary", "noc_{bb}_boundary_{ds}.csv"),
        ("Random-in-OmegaD", "noc_{bb}_random_in_d_{ds}.csv"),
        ("Centroid-in-C", "noc_{bb}_risk_{ds}.csv"),
        ("Oracle", "noc_{bb}_oracle_{ds}.csv")]
BASE_NAMES = ["Farthest-point", "Random", "Entropy", "Pert-MI r0.1", "Pert-MI r0.2", "Boundary"]

bad = [(os.path.basename(p), n) for p in glob.glob(FULL + "/*cocomval*.csv")
       for n in [sum(1 for _ in csv.DictReader(open(p, encoding="utf-8")))] if n != 800]
assert len(glob.glob(FULL + "/*cocomval*.csv")) == 40, "expected 40 cocomval files"
assert not bad, f"not finished: {bad}"
print("all 40 cocomval files at 800 rows\n")


def load(bb, pat, col="n_clicks_90"):
    out = {}
    for ds in SETS:
        root = FULL if ds == "cocomval" else V2
        for r in csv.DictReader(open(f"{root}/{pat.format(bb=bb, ds=ds)}", encoding="utf-8")):
            try:
                out[f"{ds}:{r['id']}"] = min(int(r[col]), 21)
            except (ValueError, KeyError):
                out[f"{ds}:{r['id']}"] = 21
    return out


T = {(n, bb): load(bb, p) for n, p in ROWS for bb in BB}
ids = sorted(T[("DGC", "vit_b")])
print(f"n = {len(ids)} instances per backbone\n")

print("=== Table 1 (pooled NoC@90) ===")
print(f"{'':26s}" + "".join(f"{b:>9s}" for b in BB))
for n, _ in ROWS:
    print(f"{n:26s}" + "".join(f"{np.mean(list(T[(n, b)].values())):9.2f}" for b in BB))

rng = np.random.default_rng(0)
print("\n=== paired 95% CI of each gap to DGC ===")
for bb in BB:
    a = np.array([T[("DGC", bb)][i] for i in ids], float)
    out = []
    for n, _ in ROWS[1:]:
        d = np.array([T[(n, bb)][i] for i in ids], float) - a
        lo, hi = np.percentile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), [2.5, 97.5])
        out.append(f"{n}: {d.mean():+.2f} [{lo:+.2f},{hi:+.2f}]{'' if (lo > 0 or hi < 0) else ' INCLUDES 0'}")
    print(f"  {bb}: " + " | ".join(out))

print("\n=== per-cell: DGC vs strongest non-oracle baseline (24 cells) ===")
deltas, lose = [], []
for bb in BB:
    for ds in SETS:
        k = [i for i in ids if i.startswith(ds + ":")]
        dv = np.mean([T[("DGC", bb)][i] for i in k])
        best = min(np.mean([T[(n, bb)][i] for i in k]) for n in BASE_NAMES)
        deltas.append(best - dv)
        if best - dv <= 0:
            lose.append((bb, ds, round(dv, 2), round(best, 2)))
print(f"  all positive: {not lose}   range {min(deltas):.2f}-{max(deltas):.2f}   {lose if lose else ''}")

print("\n=== failure rate (never reaches 90% IoU in 20 clicks) ===")
for n in ["DGC", "Farthest-point", "Oracle"]:
    print(f"  {n:16s}" + "".join(
        f"{np.mean(np.array(list(T[(n, b)].values())) >= 21) * 100:7.0f}%" for b in BB))
k = [i for i in ids if i.startswith("grabcut:")]
kc = [i for i in ids if i.startswith("camo:")]
print(f"  DGC ViT-B: GrabCut {np.mean([T[('DGC','vit_b')][i] >= 21 for i in k])*100:.0f}%  "
      f"CAMO {np.mean([T[('DGC','vit_b')][i] >= 21 for i in kc])*100:.0f}%")

print("\n=== both-solved decomposition, ViT-B, DGC vs farthest-point ===")
a = np.array([T[("DGC", "vit_b")][i] for i in ids], float)
b = np.array([T[("Farthest-point", "vit_b")][i] for i in ids], float)
both = (a < 21) & (b < 21)
print(f"  both solved n={both.sum()}   DGC {a[both].mean():.2f} vs fps {b[both].mean():.2f}")
shares = []
for bb in BB:
    x = np.array([T[("DGC", bb)][i] for i in ids], float)
    y = np.array([T[("Farthest-point", bb)][i] for i in ids], float)
    m = (x < 21) & (y < 21)
    shares.append((y[m] - x[m]).sum() / len(ids) / (y.mean() - x.mean()) * 100)
print(f"  share of pooled reduction from the both-solved set: "
      f"{min(shares):.0f}-{max(shares):.0f}%")
print("RECOMPUTE800_DONE")
