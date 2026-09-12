"""Protocol-fidelity evidence for Sec. 3.1. The cleanest check is same-model-vs-published:
SimpleClick ViT-B's own paper reports GrabCut/Berkeley/DAVIS NoC@85/@90 (SBD-trained:
1.40/1.54, 1.44/2.46, 4.10/5.48; COCO+LVIS-trained: 1.38/1.48, 1.36/1.97, 3.66/5.06),
and SegNext reports SAM ViT-B DAVIS NoC@90 = 5.14. Print our per-benchmark ORACLE
numbers for SimpleClick and for SAM so the sentence can quote both sides exactly.
Also find which SimpleClick checkpoint we ran, since the two published rows differ."""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv, glob, os
import numpy as np

ROOT = f"{ROOT}/results"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]

def noc(path, col):
    v = []
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            v.append(min(int(r[col]), 21))
        except (ValueError, KeyError):
            v.append(21)
    return np.array(v, float)

print("SimpleClick oracle result files found:")
sc = sorted(glob.glob(f"{ROOT}/**/noc_*simpleclick*oracle*.csv", recursive=True) +
            glob.glob(f"{ROOT}/**/noc_sc*oracle*.csv", recursive=True) +
            glob.glob(f"{ROOT}/**/*sc*_oracle_*.csv", recursive=True))
for p in sorted(set(sc)):
    print("  ", p.replace(ROOT, "results"))

print("\nper-benchmark ORACLE NoC@85 / NoC@90:")
print(f"{'':22s}" + "".join(f"{s:>14}" for s in SETS))
for label, pat in [("SAM ViT-B", "noc_v2/noc_vit_b_oracle_{ds}.csv"),
                   ("SimpleClick ViT-B", None)]:
    row = []
    for ds in SETS:
        if pat:
            p = f"{ROOT}/{pat.format(ds=ds)}"
        else:
            c = [x for x in set(sc) if f"_{ds}" in os.path.basename(x)]
            p = c[0] if c else None
        if p and os.path.exists(p):
            a, b = noc(p, "n_clicks_85"), noc(p, "n_clicks_90")
            row.append(f"{a.mean():5.2f}/{b.mean():5.2f}  ")
        else:
            row.append(f"{'NA':>12}  ")
    print(f"{label:22s}" + "".join(row))

print("\ncheckpoint used for SimpleClick (grep in run scripts):")
for f in ["run_noc_sc.py", "sc_shim.py", "agg_sc_final.py"]:
    p = f"{ROOT}/{f}"
    if os.path.exists(p):
        for line in open(p, encoding="utf-8", errors="ignore"):
            if any(k in line.lower() for k in ["ckpt", "checkpoint", ".pth", "cocolvis", "sbd", "weights"]):
                print(f"  {f}: {line.strip()[:140]}")
print("SC_PERBENCH_DONE")
