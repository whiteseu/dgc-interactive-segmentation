"""Every number the paper quotes, recomputed with the field's failure-scoring convention.

RITM and SimpleClick score an instance that never reaches the threshold as max_clicks (20);
this paper scored it 21, which inflates every reported NoC and makes the absolute values
incomparable with published ones. Switch to 20.

Failure detection must not use the capped value: at cap 20 a genuine 20-click success and a
failure are the same number. Failure is therefore raw >= 21 throughout, while the reported
score is min(raw, 20).
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import json
import os

import numpy as np

R = f"{ROOT}/results"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
CAP = 20
ROWS = [("DGC", "noc_{bb}_risk_dt_{ds}.csv"), ("Farthest-point", "noc_{bb}_fps_{ds}.csv"),
        ("Random", "noc_{bb}_random_{ds}.csv"), ("Entropy", "noc_{bb}_entropy_{ds}.csv"),
        ("Pert-MI 0.1", "noc_{bb}_bald_{ds}.csv"),
        ("Pert-MI 0.2", "noc_{bb}_bald_{ds}_rho20.csv"),
        ("Boundary", "noc_{bb}_boundary_{ds}.csv"),
        ("Random-in-OmegaD", "noc_{bb}_random_in_d_{ds}.csv"),
        ("Centroid-in-C", "noc_{bb}_risk_{ds}.csv"), ("Oracle", "noc_{bb}_oracle_{ds}.csv")]
BASE = [r[0] for r in ROWS[1:7]]
rng = np.random.default_rng(0)


def raw(root, cocoroot, pat, col="n_clicks_90", **kw):
    out = {}
    for ds in SETS:
        p = f"{R}/{cocoroot if ds == 'cocomval' else root}/{pat.format(ds=ds, **kw)}"
        assert os.path.exists(p), p
        n = 0
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                v = int(r[col])
            except (ValueError, KeyError):
                v = 21
            out[f"{ds}:{r['id']}"] = v
            n += 1
        assert ds != "cocomval" or n == 800, f"{p}: {n} rows"
    return out


def score(d):
    return {k: min(v, CAP) for k, v in d.items()}


def ci(a, b, ids, B=10000):
    dd = np.array([b[i] for i in ids], float) - np.array([a[i] for i in ids], float)
    lo, hi = np.percentile(dd[rng.integers(0, len(dd), (B, len(dd)))].mean(1), [2.5, 97.5])
    return dd.mean(), lo, hi


RAW = {(n, bb): raw("noc_v2", "noc_full", p, bb=bb) for n, p in ROWS for bb in BB}
S = {k: score(v) for k, v in RAW.items()}
ids = sorted(S[("DGC", "vit_b")])
print(f"n = {len(ids)}\n=== TABLE 1 (cap {CAP}) ===")
print(f"{'':32s}" + "".join(f"{b:>8s}" for b in BB))
for n, _ in ROWS:
    print(f"{n:32s}" + "".join(f"{np.mean(list(S[(n, b)].values())):8.2f}" for b in BB))

print("\n=== paired CI vs DGC ===")
for bb in BB:
    out = []
    for n, _ in ROWS[1:]:
        m, lo, hi = ci(S[("DGC", bb)], S[(n, bb)], ids)
        out.append(f"{n}:{m:+.2f}[{lo:+.2f},{hi:+.2f}]{'' if lo > 0 or hi < 0 else '*0*'}")
    print(f"  {bb}: " + " | ".join(out))

d24 = []
for bb in BB:
    for ds in SETS:
        k = [i for i in ids if i.startswith(ds + ":")]
        dv = np.mean([S[("DGC", bb)][i] for i in k])
        d24.append(min(np.mean([S[(n, bb)][i] for i in k]) for n in BASE) - dv)
print(f"\n24-cell margin  all>0={min(d24) > 0}  range {min(d24):.2f}-{max(d24):.2f}")
g = [np.mean(list(S[("Farthest-point", b)].values())) - np.mean(list(S[("DGC", b)].values()))
     for b in BB]
print(f"DGC vs farthest-point pooled: {min(g):.2f}-{max(g):.2f}")

print("\n=== failure rate (raw >= 21; unchanged by the cap) ===")
for n in ("DGC", "Farthest-point", "Oracle"):
    print(f"  {n:16s}" + "".join(
        f"{np.mean(np.array(list(RAW[(n, b)].values())) >= 21) * 100:7.0f}%" for b in BB))
for ds in ("grabcut", "camo"):
    k = [i for i in ids if i.startswith(ds + ":")]
    print(f"  DGC ViT-B {ds}: {np.mean([RAW[('DGC','vit_b')][i] >= 21 for i in k])*100:.0f}%")

print("\n=== both-solved decomposition, ViT-B ===")
a, b = RAW[("DGC", "vit_b")], RAW[("Farthest-point", "vit_b")]
both = [i for i in ids if a[i] < 21 and b[i] < 21]
print(f"  n={len(both)}  DGC {np.mean([a[i] for i in both]):.2f} vs fps "
      f"{np.mean([b[i] for i in both]):.2f}")
sh = []
for bb in BB:
    x, y = RAW[("DGC", bb)], RAW[("Farthest-point", bb)]
    m = [i for i in ids if x[i] < 21 and y[i] < 21]
    num = sum(min(y[i], CAP) - min(x[i], CAP) for i in m) / len(ids)
    den = np.mean(list(S[("Farthest-point", bb)].values())) - np.mean(list(S[("DGC", bb)].values()))
    sh.append(100 * num / den)
print(f"  share from both-solved: {min(sh):.0f}-{max(sh):.0f}%")

print("\n=== TABLE 2 ===")
SC = [("SimpleClick + oracle", "noc_sc_vitb_oracle_{ds}.csv"),
      ("SimpleClick + entropy", "noc_sc_vitb_entropy_{ds}.csv"),
      ("SimpleClick + farthest-point", "noc_sc_vitb_fps_{ds}.csv"),
      ("SimpleClick + boundary", "noc_sc_vitb_boundary_{ds}.csv"),
      ("SimpleClick + random", "noc_sc_vitb_random_{ds}.csv")]
t2 = {}
for name, pat in SC:
    for col, tag in (("n_clicks_85", "85"), ("n_clicks_90", "90")):
        t2[(name, tag)] = score(raw("noc_sc", "noc_sc_full", pat, col))
for name, key in (("Frozen SAM + oracle", "Oracle"), ("Frozen SAM + DGC", "DGC"),
                  ("Frozen SAM + random", "Random")):
    for col, tag in (("n_clicks_85", "85"), ("n_clicks_90", "90")):
        t2[(name, tag)] = score(raw("noc_v2", "noc_full",
                                    dict(ROWS)[key].replace("{bb}", "vit_b"), col))
for name in [s[0] for s in SC] + ["Frozen SAM + oracle", "Frozen SAM + DGC",
                                  "Frozen SAM + random"]:
    print(f"  {name:32s}{np.mean(list(t2[(name,'85')].values())):7.2f}"
          f"{np.mean(list(t2[(name,'90')].values())):7.2f}")
sco = np.mean(list(t2[("SimpleClick + oracle", "90")].values()))
scb = min(np.mean(list(t2[(n, "90")].values())) for n in [s[0] for s in SC[1:]])
fo = np.mean(list(t2[("Frozen SAM + oracle", "90")].values()))
fd = np.mean(list(t2[("Frozen SAM + DGC", "90")].values()))
print(f"  gaps: SAM {fd-fo:.2f} ({fd-fo:.1f})  SC {scb-sco:.2f} ({scb-sco:.1f})  "
      f"segmenter {fo-sco:.2f} ({fo-sco:.1f})")
ent = t2[("SimpleClick + entropy", "90")]
dgc90 = t2[("Frozen SAM + DGC", "90")]
k = sorted(set(dgc90) & set(ent))
assert len(k) == len(dgc90) == len(ent)
m, lo, hi = ci(ent, dgc90, k)
print(f"  DGC - SC entropy: {m:+.2f} CI [{lo:+.2f},{hi:+.2f}]")

print("\n=== operating regime gap closure (cap 20) ===")
import glob
DROOT = f"{ROOT}/dumps"
DD = {"vit_b": "sam_vit_b", "vit_l": "sam_vit_l", "vit_h": "sam_vit_h", "sam2": "sam2_hiera_l"}
for bb, dd in DD.items():
    stable, fail = {}, {}
    for ds in SETS:
        for p in sorted(glob.glob(f"{DROOT}/{dd}/{ds}/click1/*.npz")):
            with np.load(p) as z:
                gt, base = z["gt_mask"].astype(bool), z["base_mask"].astype(bool)
                cm, jm = z["cand_masks"].astype(bool), z["jitter_masks"][:3].astype(bool)
                valid = (~z["void_mask"].astype(bool)) if "void_mask" in z.files else np.ones_like(gt)
            E = (base ^ gt) & valid
            if not E.sum():
                continue
            D = ((cm[0] ^ cm[1]) | (cm[0] ^ cm[2]) | (cm[1] ^ cm[2])) & valid
            F = (jm ^ base[None]).any(0) & valid
            u = ((base | gt) & valid).sum()
            if (base & gt & valid).sum() / max(u, 1) >= 0.85:
                continue
            key = f"{ds}:{os.path.splitext(os.path.basename(p))[0]}"
            fail[key] = True
            stable[key] = float((E & ~D & ~F).sum() / E.sum()) >= 0.5
    ks = [k for k in fail if k in S[("DGC", bb)]]
    out = []
    for lbl, sel in (("invisible", True), ("remaining", False)):
        gg = [k for k in ks if stable[k] == sel]
        av = np.mean([S[("DGC", bb)][k] for k in gg])
        rv = np.mean([S[("Random", bb)][k] for k in gg])
        ov = np.mean([S[("Oracle", bb)][k] for k in gg])
        out.append(f"{lbl} n={len(gg)} closure {100*(rv-av)/(rv-ov):+.0f}%")
    print(f"  {bb:6s} share invisible {100*np.mean([stable[k] for k in ks]):.0f}%  " + " | ".join(out))
print("ALLNUMBERS20_DONE")
