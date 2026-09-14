"""Exact unrounded values behind the three gaps quoted in Sec. 3.3 and contribution 3.

The paper prints Table 2 to two decimals and then quotes differences of those entries. If a
difference of the unrounded values rounds differently from the difference of the printed
values, a reader with a calculator sees a mismatch, so print both forms for each gap.
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


def pooled(root, cocoroot, pat, col="n_clicks_90"):
    v = []
    for ds in SETS:
        p = f"{R}/{cocoroot if ds == 'cocomval' else root}/{pat.format(ds=ds)}"
        assert os.path.exists(p), p
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                v.append(min(int(r[col]), 21))
            except (ValueError, KeyError):
                v.append(21)
    return float(np.mean(v))


sc_o = pooled("noc_sc", "noc_sc_full", "noc_sc_vitb_oracle_{ds}.csv")
sc_e = pooled("noc_sc", "noc_sc_full", "noc_sc_vitb_entropy_{ds}.csv")
sam_o = pooled("noc_v2", "noc_full", "noc_vit_b_oracle_{ds}.csv")
sam_d = pooled("noc_v2", "noc_full", "noc_vit_b_risk_dt_{ds}.csv")

print(f"{'quantity':34s}{'exact':>10s}{'printed':>10s}")
for n, v in (("SimpleClick + oracle", sc_o), ("SimpleClick + entropy", sc_e),
             ("Frozen SAM + oracle", sam_o), ("Frozen SAM + DGC", sam_d)):
    print(f"{n:34s}{v:10.4f}{v:10.2f}")

print(f"\n{'gap':34s}{'from exact':>12s}{'from printed':>14s}{'agree':>7s}")
for n, a, b in (("segmenter oracle gap", sam_o, sc_o),
                ("frozen SAM policy gap", sam_d, sam_o),
                ("SimpleClick policy gap", sc_e, sc_o),
                ("DGC - SimpleClick entropy", sam_d, sc_e)):
    ex = a - b
    pr = round(a, 2) - round(b, 2)
    print(f"{n:34s}{ex:12.4f}{pr:14.2f}{'  ok' if abs(round(ex, 2) - round(pr, 2)) < 5e-3 else '  MISMATCH'}")
print("GAPDIGITS_DONE")
