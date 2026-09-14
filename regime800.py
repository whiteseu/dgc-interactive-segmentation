"""The Sec. 3.3 numbers that come from the first-click dumps, on the full 800 split.

Recomputes, over the six evaluated benchmarks and all four backbones:
  - coverage: median over first-round failures of |Omega_D & E| / |E|   (paper: 72-80%)
  - precision: pixel-pooled sum|Omega_D & E| / sum|Omega_D|              (paper: 0.20-0.37)
  - operating regime: share of first-round failures whose residual error is invisible to
    both signals, S = |E & D=0 & F=0| / |E| >= 0.5, with F the prompt-jitter flip map from
    the first K=3 jitter masks at rho=0.1 -- the probe the paper actually describes
  - gap closure (NoC_random - NoC_DGC) / (NoC_random - NoC_oracle) on each group

The published 27%/39% came from results/stable_robust.csv, which covers five benchmarks
and excludes COCO-MVal entirely; that basis is printed alongside so the change is visible
rather than silent.

Asserts 800 cocomval dumps per backbone before computing anything.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import glob
import os

import numpy as np

ROOT = ROOT
V2 = f"{ROOT}/results/noc_v2"
FULL = f"{ROOT}/results/noc_full"
SIX = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
FIVE = [s for s in SIX if s != "cocomval"]
BB = {"vit_b": "sam_vit_b", "vit_l": "sam_vit_l", "vit_h": "sam_vit_h",
      "sam2": "sam2_hiera_l"}
K = 3

for d in BB.values():
    n = len(glob.glob(f"{ROOT}/dumps/{d}/cocomval/click1/*.npz"))
    assert n == 800, f"{d}: {n} cocomval dumps, expected 800"
print("all four backbones have 800 cocomval dumps\n")


def scan(dumpdir):
    """Per-instance record for every first-click dump of one backbone."""
    rec = {}
    for ds in SIX:
        for p in sorted(glob.glob(f"{ROOT}/dumps/{dumpdir}/{ds}/click1/*.npz")):
            with np.load(p) as z:
                gt = z["gt_mask"].astype(bool)
                base = z["base_mask"].astype(bool)
                cm = z["cand_masks"].astype(bool)
                jm = z["jitter_masks"][:K].astype(bool)
                valid = (~z["void_mask"].astype(bool)) if "void_mask" in z.files \
                    else np.ones_like(gt)
            E = (base ^ gt) & valid
            Dm = ((cm[0] ^ cm[1]) | (cm[0] ^ cm[2]) | (cm[1] ^ cm[2])) & valid
            F = (jm ^ base[None]).any(0) & valid
            ne = int(E.sum())
            inter = (base & gt & valid).sum()
            union = ((base | gt) & valid).sum()
            rec[os.path.splitext(os.path.basename(p))[0]] = dict(
                ds=ds, n_err=ne, iou=float(inter / union) if union else 1.0,
                dh=int((Dm & E).sum()), dn=int(Dm.sum()),
                stable=float((E & ~Dm & ~F).sum() / ne) if ne else 0.0)
    return rec


def noc(bb, strat):
    out = {}
    for ds in SIX:
        root = FULL if ds == "cocomval" else V2
        for r in csv.DictReader(open(f"{root}/noc_{bb}_{strat}_{ds}.csv", encoding="utf-8")):
            try:
                out[r["id"]] = min(int(r["n_clicks_90"]), 21)
            except (ValueError, KeyError):
                out[r["id"]] = 21
    return out


print(f"{'bb':6s}{'n':>7s}{'cover med(fail)':>17s}{'precision':>11s}"
      f"{'regime six/800':>16s}{'regime five':>13s}")
REC = {}
for bb, d in BB.items():
    rec = scan(d)
    REC[bb] = rec
    fail = [v for v in rec.values() if v["n_err"] > 0 and v["iou"] < 0.85]
    cov = np.median([v["dh"] / v["n_err"] for v in fail])
    prec = sum(v["dh"] for v in rec.values()) / max(sum(v["dn"] for v in rec.values()), 1)
    six = np.mean([v["stable"] >= 0.5 for v in fail])
    f5 = [v for v in fail if v["ds"] in FIVE]
    print(f"{bb:6s}{len(rec):7d}{cov:17.3f}{prec:11.3f}"
          f"{six * 100:15.1f}%{np.mean([v['stable'] >= 0.5 for v in f5]) * 100:12.1f}%"
          f"   (n_fail six={len(fail)} five={len(f5)})")

print("\n=== gap closure on each regime group: (random - DGC) / (random - oracle) ===")
for bb in BB:
    dgc, rnd, orc = noc(bb, "risk_dt"), noc(bb, "random"), noc(bb, "oracle")
    rec = REC[bb]
    fail = [k for k, v in rec.items() if v["n_err"] > 0 and v["iou"] < 0.85]
    miss = [k for k in fail if k not in dgc]
    assert not miss, f"{bb}: {len(miss)} failures absent from the NoC CSVs, e.g. {miss[:3]}"
    out = []
    for label, sel in (("invisible (S>=0.5)", lambda v: v["stable"] >= 0.5),
                       ("remaining", lambda v: v["stable"] < 0.5)):
        g = [k for k in fail if sel(rec[k])]
        a = np.mean([dgc[k] for k in g])
        r = np.mean([rnd[k] for k in g])
        o = np.mean([orc[k] for k in g])
        out.append(f"{label}: n={len(g)} closure {100 * (r - a) / (r - o):+.0f}% "
                   f"(rnd {r:.2f} DGC {a:.2f} orc {o:.2f})")
    print(f"  {bb:6s} " + " | ".join(out))
print("REGIME800_DONE")
