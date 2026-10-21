"""Fig. 3 caption claims DGC leads every Table 1 baseline from the second click onward
and that the ordering holds on all four backbones, while the figure draws ViT-B only.
That is 6 baselines x clicks 2..20 x 4 backbones = 456 comparisons. Check every one;
if any fails the caption must be narrowed to ViT-B.

Mean-IoU curves come from iou_traj, which is unaffected by the n_clicks_90 rounding
issue -- that only bit the NoC@90 threshold crossing, not the IoU values themselves.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv, json, os
import numpy as np

V2 = f"{ROOT}/results/noc_v2"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
BASE = [("farthest-point", "noc_{bb}_fps_{ds}.csv"),
        ("random",         "noc_{bb}_random_{ds}.csv"),
        ("entropy",        "noc_{bb}_entropy_{ds}.csv"),
        ("pert-MI r0.1",   "noc_{bb}_bald_{ds}.csv"),
        ("pert-MI r0.2",   "noc_{bb}_bald_{ds}_rho20.csv"),
        ("boundary",       "noc_{bb}_boundary_{ds}.csv")]
T = 20

def curve(bb, pat):
    tr = []
    for ds in SETS:
        p = f"{V2}/{pat.format(bb=bb, ds=ds)}"
        if not os.path.exists(p):
            return None, 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            t = json.loads(r["iou_traj"])
            tr.append((t + [t[-1]] * (T - len(t)))[:T])
    a = np.array(tr)
    return a.mean(0), len(a)

fails = []
for bb in BB:
    A, n = curve(bb, "noc_{bb}_risk_dt_{ds}.csv")
    print(f"\n{bb}  (n={n})")
    for name, pat in BASE:
        B, m = curve(bb, pat)
        if B is None:
            print(f"  {name:15s} MISSING"); fails.append((bb, name, "missing")); continue
        assert m == n, (bb, name, m, n)
        d = A[1:] - B[1:]                      # clicks 2..20
        ok = bool((d > 0).all())
        if not ok:
            lost = [i + 2 for i, v in enumerate(d) if v <= 0]
            fails.append((bb, name, f"loses at clicks {lost}"))
        print(f"  {name:15s} leads clicks 2-20: {str(ok):5s}  min margin {d.min():+.4f} "
              f"(at click {int(np.argmin(d)) + 2})  click-20 margin {d[-1]:+.4f}")
print("\nFAILURES:", fails if fails else "none -- caption's all-four-backbones claim holds")
print("FIG3_ALLBB_DONE")
