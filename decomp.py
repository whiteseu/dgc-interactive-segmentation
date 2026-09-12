"""Reviewer's arithmetic, checked directly rather than reverse-engineered.

Claim: pooled NoC@90 is dominated by the 21-click penalty on failures, so DGC's gain over
farthest-point comes almost entirely from rescuing instances that farthest-point cannot
solve in 20 clicks, not from converging faster on instances both solve.

Exact decomposition of the pooled gap, per backbone:
  mean(B) - mean(A) = [share solved by both] * (mean click gap on that subset)
                    + [contributions of the disagreement sets, where one fails and the
                       other succeeds, and of the both-fail set (which contributes 0)]
Also reports how many random seeds the CSVs actually contain, since several policies are
stochastic and a 0.12-click ablation gap could otherwise sit inside seed noise.
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
POL = {"DGC": "noc_{bb}_risk_dt_{ds}.csv",
       "farthest-point": "noc_{bb}_fps_{ds}.csv",
       "Random-in-OmegaD": "noc_{bb}_random_in_d_{ds}.csv",
       "random": "noc_{bb}_random_{ds}.csv",
       "oracle": "noc_{bb}_oracle_{ds}.csv"}


def load(bb, pat):
    out, seeds = {}, set()
    for ds in SETS:
        p = f"{V2}/{pat.format(bb=bb, ds=ds)}"
        if not os.path.exists(p):
            return None, None
        for r in csv.DictReader(open(p, encoding="utf-8")):
            seeds.add(r.get("first_seed", "?"))
            try:
                v = min(int(r["n_clicks_90"]), 21)
            except (ValueError, KeyError):
                v = 21
            out[f"{ds}:{r['id']}"] = v
    return out, seeds


print("=== seeds present in the CSVs ===")
a, s = load("vit_b", POL["DGC"])
print(f"  distinct first_seed values: {sorted(s)}  -> {len(s)} seed(s) per instance\n")

for bb in BB:
    A, _ = load(bb, POL["DGC"])
    print(f"--- {bb}: DGC pooled {np.mean(list(A.values())):.2f} ---")
    for name in ["farthest-point", "Random-in-OmegaD"]:
        B, _ = load(bb, POL[name])
        ids = sorted(A)
        a_ = np.array([A[i] for i in ids], float)
        b_ = np.array([B[i] for i in ids], float)
        fa, fb = a_ >= 21, b_ >= 21
        both = ~fa & ~fb
        gap = b_.mean() - a_.mean()
        # contribution of the both-solved subset to the pooled gap
        contrib_both = (b_[both] - a_[both]).sum() / len(ids)
        # contribution of instances where exactly one policy fails
        only_b_fails = ~fa & fb
        only_a_fails = fa & ~fb
        contrib_bfail = (b_[only_b_fails] - a_[only_b_fails]).sum() / len(ids)
        contrib_afail = (b_[only_a_fails] - a_[only_a_fails]).sum() / len(ids)
        print(f"  vs {name:17s} pooled gap {gap:+.2f}")
        print(f"     failure rate  DGC {fa.mean()*100:5.1f}%   {name} {fb.mean()*100:5.1f}%")
        print(f"     both solved (n={both.sum():4d}, {both.mean()*100:4.1f}%): "
              f"mean clicks DGC {a_[both].mean():.2f} vs {b_[both].mean():.2f} "
              f"(diff {b_[both].mean()-a_[both].mean():+.2f})")
        print(f"     gap decomposition: both-solved {contrib_both:+.2f} "
              f"({100*contrib_both/gap:4.0f}%) | only-{name}-fails {contrib_bfail:+.2f} "
              f"({100*contrib_bfail/gap:4.0f}%) | only-DGC-fails {contrib_afail:+.2f} "
              f"({100*contrib_afail/gap:4.0f}%)")
    # how much of the pooled VALUE is the penalty
    fa = np.array(list(A.values()), float) >= 21
    v = np.array(list(A.values()), float)
    print(f"  share of DGC's pooled NoC@90 contributed by the 21-penalty on failures: "
          f"{21*fa.sum()/v.sum()*100:.0f}%\n")
print("DECOMP_DONE")
