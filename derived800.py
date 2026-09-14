"""The Sec. 3.3 numbers that come from the NoC CSVs rather than from the pixel dumps,
recomputed on the full 800-instance COCO-MVal split.

Covers: the in-error click fraction per policy, the Omega_D-empty fallback rate, and the
paired CI for frozen SAM + DGC against SimpleClick + entropy in Table 2.

Fails loudly unless every cocomval file holds 800 rows, so a half-finished batch cannot be
mistaken for a result.
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
SC = f"{ROOT}/results/noc_sc"
SCF = f"{ROOT}/results/noc_sc_full"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
T_MAX = 20

for root, tag, want in ((FULL, "SAM", 40), (SCF, "SimpleClick", 5)):
    fs = [p for p in glob.glob(root + "/*cocomval*.csv") if "oracle_ours" not in p]
    bad = [(os.path.basename(p), n) for p in fs
           for n in [sum(1 for _ in csv.DictReader(open(p, encoding="utf-8")))] if n != 800]
    assert len(fs) == want, f"{tag}: expected {want} cocomval files, found {len(fs)}"
    assert not bad, f"{tag} not finished: {bad}"
print("cocomval inputs verified at 800 rows\n")


def rows(root, cocoroot, pat, **kw):
    for ds in SETS:
        r0 = cocoroot if ds == "cocomval" else root
        p = f"{r0}/{pat.format(ds=ds, **kw)}"
        if not os.path.exists(p):
            return
        for r in csv.DictReader(open(p, encoding="utf-8")):
            r["_ds"] = ds
            yield r


print("=== in-error click fraction (mean over instances of per-instance hit_rate) ===")
print(f"{'policy':18s}" + "".join(f"{b:>10s}" for b in BB) + "      n")
for label, strat in [("DGC", "risk_dt"), ("Farthest-point", "fps"),
                     ("Perturbation-MI", "bald"), ("Random", "random")]:
    cells, ns = [], []
    for bb in BB:
        vals = []
        for r in rows(V2, FULL, "noc_{bb}_{s}_{ds}.csv", bb=bb, s=strat):
            h = r.get("hit_rate", "")
            if h in ("", None):
                continue
            if min(int(r["n_clicks_90"]), T_MAX) - 1 <= 0:   # no guided click on this instance
                continue
            vals.append(float(h))
        cells.append(f"{np.mean(vals) * 100:9.1f}%")
        ns.append(len(vals))
    print(f"{label:18s}" + "".join(cells) + f"   {ns}")

print("\n=== Omega_D empty fallback ===")
for bb in BB:
    n = ef = nev = 0
    for r in rows(V2, FULL, "noc_{bb}_risk_dt_{ds}.csv", bb=bb):
        n += 1
        ef += str(r.get("first_d_empty", "")) == "1"
        nev += str(r.get("round_d1", "-1")) in ("-1", "")
    print(f"  {bb:6s} n={n}  empty at first guided click: {ef}  never non-empty: {nev}")

print("\n=== Table 2: frozen SAM + DGC vs SimpleClick + entropy (paired, NoC@90) ===")


def vec(root, cocoroot, pat, col="n_clicks_90"):
    out = {}
    for r in rows(root, cocoroot, pat):
        try:
            out[f"{r['_ds']}:{r['id']}"] = min(int(r[col]), 21)
        except (ValueError, KeyError):
            out[f"{r['_ds']}:{r['id']}"] = 21
    return out


dgc = vec(V2, FULL, "noc_vit_b_risk_dt_{ds}.csv")
ent = vec(SC, SCF, "noc_sc_vitb_entropy_{ds}.csv")
ids = sorted(set(dgc) & set(ent))
assert len(ids) == len(dgc) == len(ent), f"instance sets differ: {len(dgc)} {len(ent)} {len(ids)}"
d = np.array([dgc[i] for i in ids], float) - np.array([ent[i] for i in ids], float)
rng = np.random.default_rng(0)
lo, hi = np.percentile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), [2.5, 97.5])
print(f"  n={len(ids)}  DGC - entropy = {d.mean():+.2f}  CI [{lo:+.2f},{hi:+.2f}]"
      f"{'  (includes 0)' if lo < 0 < hi else ''}")
print("DERIVED800_DONE")
