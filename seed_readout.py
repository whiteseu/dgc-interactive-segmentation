"""Seed sensitivity of the Random-in-Omega_D ablation on ViT-B.

DGC is deterministic (its click is the distance-transform maximum and the stochastic
fallback never fires), so the seed variance of the DGC-vs-ablation gap is entirely the
ablation's.

The paper's only significance claim about the ablation is "the paired 95% CIs exclude zero
only on ViT-B". The decisive question is therefore not the size of the across-seed spread
but whether that CI still excludes zero under every seed. Both are reported.

Decision rule fixed before looking:
  CI excludes zero for every seed -> the claim is seed-robust; leave the text as is
  CI includes zero for some seed  -> the claim is not supportable and must be softened
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import os

import numpy as np

V2 = f"{ROOT}/results/noc_v2"
SD = f"{ROOT}/results/noc_seed"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]


def load(paths):
    out = {}
    for ds, p in paths:
        if not os.path.exists(p):
            return None
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                v = min(int(r["n_clicks_90"]), 21)
            except (ValueError, KeyError):
                v = 21
            out[f"{ds}:{r['id']}"] = v
    return out


dgc = load([(ds, f"{V2}/noc_vit_b_risk_dt_{ds}.csv") for ds in SETS])
runs = {0: load([(ds, f"{V2}/noc_vit_b_random_in_d_{ds}.csv") for ds in SETS])}
for s in (1, 2):
    runs[s] = load([(ds, f"{SD}/noc_vit_b_random_in_d_{ds}_s{s}.csv") for ds in SETS])

ids = sorted(dgc)
a = np.array([dgc[i] for i in ids], float)
print(f"DGC pooled NoC@90 (deterministic): {a.mean():.3f}  n={len(ids)}\n")
print(f"{'seed':>5s}{'ablation':>11s}{'gap':>9s}{'paired 95% CI':>22s}{'excl 0':>8s}")

rng = np.random.default_rng(0)
gaps, excl = [], []
for s, r in runs.items():
    if r is None:
        print(f"{s:5d}      incomplete, skipping")
        continue
    assert set(r) == set(dgc), (len(r), len(dgc))
    b = np.array([r[i] for i in ids], float)
    d = b - a
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ok = (lo > 0) or (hi < 0)
    gaps.append(d.mean()); excl.append(ok)
    print(f"{s:5d}{b.mean():11.3f}{d.mean():+9.3f}   [{lo:+.3f},{hi:+.3f}]{str(ok):>8s}")

if len(gaps) >= 2:
    g = np.array(gaps)
    print(f"\n  across seeds: mean {g.mean():+.3f}  sd {g.std(ddof=1):.3f}  "
          f"range {g.min():+.3f}..{g.max():+.3f}  spread {g.max()-g.min():.3f}")
    print(f"  Table 1 reports the seed-0 value (10.28); seed mean is {np.mean([np.mean([runs[s][i] for i in ids]) for s in runs if runs[s]]):.2f}")
    if all(excl):
        print("  VERDICT: the CI excludes zero under every seed -- the ViT-B claim is "
              "seed-robust, leave the text unchanged")
    else:
        print("  VERDICT: the CI includes zero for at least one seed -- the ViT-B "
              "significance claim must be softened")
print("SEED_READOUT_DONE")
