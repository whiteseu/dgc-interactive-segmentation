"""Single-mask protocol ablation readout. Compares, for SAM ViT-B on the six benchmarks,
the pooled NoC@90 of oracle and farthest-point under (a) the Table 1 protocol
(multimask_output=True every round, results/noc_v2) and (b) single-mask decoding for
rounds t>=2 (results/noc_sm). Paired over identical instance ids, 10k bootstrap CI.

Interpretation rule fixed BEFORE looking (from the review):
  |Delta| <= 0.2 on both policies  -> 'changes pooled NoC@90 by only X/Y clicks'
  0.2 < |Delta| <= 0.5             -> report exact numbers, no 'does not alter'
  |Delta| > 0.5 or sign disagrees   -> state numbers only, no claim of insensitivity
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv, os
import numpy as np

V2, SM = f"{ROOT}/results/noc_v2", f"{ROOT}/results/noc_sm"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]

def load(path):
    out = {}
    if not os.path.exists(path):
        return None
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            v = min(int(r["n_clicks_90"]), 21)
        except (ValueError, KeyError):
            v = 21
        out[r["id"]] = v
    return out

rng = np.random.default_rng(0)
verdict = {}
for pol in ["oracle", "fps"]:
    A, B, per = {}, {}, []
    complete = True
    for ds in SETS:
        a = load(f"{V2}/noc_vit_b_{pol}_{ds}.csv")
        b = load(f"{SM}/noc_vit_b_{pol}_sm_{ds}.csv")
        if a is None or b is None or len(b) < len(a):
            print(f"[{pol}/{ds}] incomplete: v2={None if a is None else len(a)} sm={None if b is None else len(b)}")
            complete = False
            continue
        ids = sorted(set(a) & set(b))
        assert len(ids) == len(a), (pol, ds, len(ids), len(a))
        A.update({f"{ds}:{i}": a[i] for i in ids}); B.update({f"{ds}:{i}": b[i] for i in ids})
        per.append((ds, np.mean([a[i] for i in ids]), np.mean([b[i] for i in ids]), len(ids)))
    if not complete:
        print(f"== {pol}: NOT COMPLETE, skipping pooled readout ==\n"); continue
    ids = sorted(A)
    d = np.array([B[i] - A[i] for i in ids], float)
    bs = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    ma, mb = np.mean([A[i] for i in ids]), np.mean([B[i] for i in ids])
    print(f"== {pol}: pooled NoC@90 multimask {ma:.2f} -> single-mask {mb:.2f}   "
          f"Delta {mb-ma:+.3f}  CI [{lo:+.3f},{hi:+.3f}]  n={len(ids)}")
    for ds, x, y, n in per:
        print(f"   {ds:10s} {x:6.2f} -> {y:6.2f}  ({y-x:+.2f})  n={n}")
    verdict[pol] = mb - ma
if len(verdict) == 2:
    dl = [abs(v) for v in verdict.values()]
    same_sign = np.sign(verdict["oracle"]) == np.sign(verdict["fps"]) or min(dl) < 0.05
    if max(dl) <= 0.2:
        print("\nRULE -> 'changes pooled NoC@90 by only X/Y clicks'")
    elif max(dl) <= 0.5 and same_sign:
        print("\nRULE -> report exact numbers; do NOT write 'does not alter'")
    else:
        print("\nRULE -> numbers only; no insensitivity claim")
print("SM_READOUT_DONE")
