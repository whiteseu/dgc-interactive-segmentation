"""Every headline number under the two failure-scoring conventions, side by side.

The reference implementation that RITM and SimpleClick share scores an instance that never
reaches the threshold as max_clicks (20); this paper scored it 21. The convention does not
change which instances fail, but it changes every reported NoC, so print both and check
whether any claim depends on the choice.
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
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
ROWS = [("DGC", "noc_{bb}_risk_dt_{ds}.csv"), ("Farthest-point", "noc_{bb}_fps_{ds}.csv"),
        ("Random", "noc_{bb}_random_{ds}.csv"), ("Entropy", "noc_{bb}_entropy_{ds}.csv"),
        ("Pert-MI 0.1", "noc_{bb}_bald_{ds}.csv"),
        ("Pert-MI 0.2", "noc_{bb}_bald_{ds}_rho20.csv"),
        ("Boundary", "noc_{bb}_boundary_{ds}.csv"),
        ("Random-in-OmegaD", "noc_{bb}_random_in_d_{ds}.csv"),
        ("Centroid-in-C", "noc_{bb}_risk_{ds}.csv"), ("Oracle", "noc_{bb}_oracle_{ds}.csv")]
BASE = [r[0] for r in ROWS[1:7]]


def load(root, cocoroot, pat, cap, col="n_clicks_90", **kw):
    out = {}
    for ds in SETS:
        p = f"{R}/{cocoroot if ds == 'cocomval' else root}/{pat.format(ds=ds, **kw)}"
        assert os.path.exists(p), p
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                out[f"{ds}:{r['id']}"] = min(int(r[col]), cap)
            except (ValueError, KeyError):
                out[f"{ds}:{r['id']}"] = cap
    return out


for cap in (21, 20):
    print(f"\n{'='*20} failures scored {cap} {'='*20}")
    T = {(n, bb): load("noc_v2", "noc_full", p, cap, bb=bb) for n, p in ROWS for bb in BB}
    ids = sorted(T[("DGC", "vit_b")])
    print(f"{'':20s}" + "".join(f"{b:>8s}" for b in BB))
    for n, _ in ROWS:
        print(f"{n:20s}" + "".join(f"{np.mean(list(T[(n, b)].values())):8.2f}" for b in BB))

    rng = np.random.default_rng(0)
    print("  paired CI vs DGC, excludes zero?")
    for bb in BB:
        a = np.array([T[("DGC", bb)][i] for i in ids], float)
        out = []
        for n in BASE + ["Random-in-OmegaD", "Centroid-in-C"]:
            dd = np.array([T[(n, bb)][i] for i in ids], float) - a
            lo, hi = np.percentile(dd[rng.integers(0, len(dd), (4000, len(dd)))].mean(1),
                                   [2.5, 97.5])
            out.append(f"{n.split('-in-')[0][:6]}:{'Y' if lo > 0 else 'n'}")
        print(f"    {bb:6s} " + " ".join(out))

    deltas = []
    for bb in BB:
        for ds in SETS:
            k = [i for i in ids if i.startswith(ds + ":")]
            dv = np.mean([T[("DGC", bb)][i] for i in k])
            deltas.append(min(np.mean([T[(n, bb)][i] for i in k]) for n in BASE) - dv)
    print(f"  24-cell margin: all positive={min(deltas) > 0}  range {min(deltas):.2f}-{max(deltas):.2f}")
    g = [np.mean(list(T[("Farthest-point", b)].values())) - np.mean(list(T[("DGC", b)].values()))
         for b in BB]
    print(f"  DGC vs farthest-point pooled gap: {min(g):.2f}-{max(g):.2f}")
    rid = [np.mean(list(T[("Random-in-OmegaD", b)].values())) - np.mean(list(T[("DGC", b)].values()))
           for b in BB]
    cen = [np.mean(list(T[("Centroid-in-C", b)].values())) - np.mean(list(T[("DGC", b)].values()))
           for b in BB]
    print(f"  Random-in-OmegaD - DGC: {min(rid):.2f}-{max(rid):.2f}   "
          f"Centroid-in-C - DGC: {min(cen):.2f}-{max(cen):.2f}")
    print(f"  Centroid - Random-in-OmegaD: "
          f"{min(c - r for c, r in zip(cen, rid)):.2f}-{max(c - r for c, r in zip(cen, rid)):.2f}")

    sc = {n: load("noc_sc", "noc_sc_full", p, cap) for n, p in
          [("oracle", "noc_sc_vitb_oracle_{ds}.csv"), ("entropy", "noc_sc_vitb_entropy_{ds}.csv"),
           ("fps", "noc_sc_vitb_fps_{ds}.csv"), ("boundary", "noc_sc_vitb_boundary_{ds}.csv"),
           ("random", "noc_sc_vitb_random_{ds}.csv")]}
    sco = np.mean(list(sc["oracle"].values()))
    scb = min(np.mean(list(sc[k].values())) for k in ("entropy", "fps", "boundary", "random"))
    fo = np.mean(list(T[("Oracle", "vit_b")].values()))
    fd = np.mean(list(T[("DGC", "vit_b")].values()))
    print(f"  Table 2: SC oracle {sco:.2f}, SC best {scb:.2f}, SAM oracle {fo:.2f}, DGC {fd:.2f}")
    print(f"           policy gaps {fd-fo:.2f} / {scb-sco:.2f}, segmenter oracle gap {fo-sco:.2f}, "
          f"DGC-SCbest {fd-scb:+.2f}")
print("CAP_DONE")
