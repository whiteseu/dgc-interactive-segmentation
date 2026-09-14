"""Table 2 (frozen SAM vs SimpleClick) on the full 800-instance COCO-MVal split.

Two SimpleClick oracle files exist (oracle, oracle_ours); the paper quotes 3.05/4.24, so
first identify which variant reproduces that on the old 785 split, then report the full
split from the matching one. Prints both rather than assuming.

COCO-MVal comes from results/noc_sc_full (SimpleClick) and results/noc_full (SAM); the
other five benchmarks are unaffected by the <100 px filter.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import os

import numpy as np

R = f"{ROOT}/results"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]


def pooled(pat, full, cocoroot, col):
    v = []
    for ds in SETS:
        root = cocoroot if (full and ds == "cocomval") else pat[0]
        p = f"{R}/{root}/{pat[1].format(ds=ds)}"
        if not os.path.exists(p):
            return None, 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                v.append(min(int(r[col]), 21))
            except (ValueError, KeyError):
                v.append(21)
    return float(np.mean(v)), len(v)


SC = [("SimpleClick + oracle", ("noc_sc", "noc_sc_vitb_oracle_{ds}.csv")),
      ("SimpleClick + oracle_ours", ("noc_sc", "noc_sc_vitb_oracle_ours_{ds}.csv")),
      ("SimpleClick + entropy", ("noc_sc", "noc_sc_vitb_entropy_{ds}.csv")),
      ("SimpleClick + farthest-point", ("noc_sc", "noc_sc_vitb_fps_{ds}.csv")),
      ("SimpleClick + boundary", ("noc_sc", "noc_sc_vitb_boundary_{ds}.csv")),
      ("SimpleClick + random", ("noc_sc", "noc_sc_vitb_random_{ds}.csv"))]
SAM = [("Frozen SAM + oracle", ("noc_v2", "noc_vit_b_oracle_{ds}.csv")),
       ("Frozen SAM + DGC", ("noc_v2", "noc_vit_b_risk_dt_{ds}.csv")),
       ("Frozen SAM + random", ("noc_v2", "noc_vit_b_random_{ds}.csv"))]

print(f"{'system + policy':32s}{'NoC@85 785->800':>26s}{'NoC@90 785->800':>26s}{'n':>14s}")
res = {}
for name, pat in SC + SAM:
    cells, ns = [], []
    for col in ("n_clicks_85", "n_clicks_90"):
        o, no = pooled(pat, False, None, col)
        n, nn = pooled(pat, True, "noc_sc_full" if "sc" in pat[0] else "noc_full", col)
        if o is None or n is None:
            cells.append(f"{'NA':>26s}"); ns.append(0); continue
        res[(name, col)] = (o, n)
        cells.append(f"{o:11.2f} ->{n:11.2f}")
        ns.append((no, nn))
    print(f"{name:32s}" + "".join(cells) + f"{str(ns[-1]):>14s}")

print("\n=== which oracle variant gives the paper's 3.05 / 4.24 on 785? ===")
for v in ("SimpleClick + oracle", "SimpleClick + oracle_ours"):
    a = res.get((v, "n_clicks_85"), (None,))[0]
    b = res.get((v, "n_clicks_90"), (None,))[0]
    if a is None:
        print(f"  {v}: incomplete"); continue
    print(f"  {v}: {a:.2f} / {b:.2f}"
          f"{'   <-- MATCHES PAPER' if abs(a - 3.05) < .005 and abs(b - 4.24) < .005 else ''}")

print("\n=== derived claims (800 split) ===")
for v in ("SimpleClick + oracle", "SimpleClick + oracle_ours"):
    if (v, "n_clicks_90") not in res:
        continue
    so = res[(v, "n_clicks_90")][1]
    sb = min(res[(k, "n_clicks_90")][1] for k in
             ("SimpleClick + entropy", "SimpleClick + farthest-point",
              "SimpleClick + boundary", "SimpleClick + random"))
    fo = res[("Frozen SAM + oracle", "n_clicks_90")][1]
    fd = res[("Frozen SAM + DGC", "n_clicks_90")][1]
    print(f"  [{v}] SimpleClick best non-oracle {sb:.2f}, oracle {so:.2f} "
          f"-> policy gap {sb - so:.2f}")
    print(f"        frozen SAM DGC {fd:.2f}, oracle {fo:.2f} -> policy gap {fd - fo:.2f}")
    print(f"        segmenter oracle gap {fo - so:.2f}   DGC - SC-best {fd - sb:+.2f}")
print("TAB2_800_DONE")
