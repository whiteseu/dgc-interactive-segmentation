"""Side-by-side of every Table 1 number on the filtered 785-instance COCO-MVal split and on
the full 800-instance split.

The 15 restored instances have 30-84 foreground pixels. If they are failures for every
policy including the oracle, all absolute values rise and the paired gaps barely move, and
the paper needs new numbers but no new claims. That is the hypothesis; this prints the
evidence either way, including whether any of the 24 per-cell wins flips sign.

Reads COCO-MVal from results/noc_full and the other five benchmarks from results/noc_v2,
which are unaffected by the filter.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import os

import numpy as np

V2 = f"{ROOT}/results/noc_v2"
FULL = f"{ROOT}/results/noc_full"
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
ROWS = [("DGC", "noc_{bb}_risk_dt_{ds}.csv"),
        ("Random-in-OmegaD", "noc_{bb}_random_in_d_{ds}.csv"),
        ("Centroid-in-C", "noc_{bb}_risk_{ds}.csv"),
        ("Farthest-point", "noc_{bb}_fps_{ds}.csv"),
        ("Random", "noc_{bb}_random_{ds}.csv"),
        ("Entropy", "noc_{bb}_entropy_{ds}.csv"),
        ("Pert-MI r0.1", "noc_{bb}_bald_{ds}.csv"),
        ("Pert-MI r0.2", "noc_{bb}_bald_{ds}_rho20.csv"),
        ("Boundary", "noc_{bb}_boundary_{ds}.csv"),
        ("Oracle", "noc_{bb}_oracle_{ds}.csv")]
BASES = [r for r in ROWS if r[0] not in ("DGC", "Oracle", "Random-in-OmegaD", "Centroid-in-C")]


def load(bb, pat, full):
    out = {}
    for ds in SETS:
        root = FULL if (full and ds == "cocomval") else V2
        p = f"{root}/{pat.format(bb=bb, ds=ds)}"
        if not os.path.exists(p):
            return None
        for r in csv.DictReader(open(p, encoding="utf-8")):
            try:
                out[f"{ds}:{r['id']}"] = min(int(r["n_clicks_90"]), 21)
            except (ValueError, KeyError):
                out[f"{ds}:{r['id']}"] = 21
    return out


print("=== the 15 restored instances: is every policy failing on them? ===")
new_ids = None
for bb in BB:
    a, b = load(bb, ROWS[0][1], False), load(bb, ROWS[0][1], True)
    if a is None or b is None:
        print(f"  {bb}: incomplete"); continue
    if new_ids is None:
        new_ids = sorted(set(b) - set(a))
    print(f"  {bb}: n {len(a)} -> {len(b)}")
if new_ids:
    print(f"  restored: {len(new_ids)} instances")
    print(f"{'policy':18s}" + "".join(f"{b:>22s}" for b in BB))
    for name, pat in ROWS:
        cells = []
        for bb in BB:
            f = load(bb, pat, True)
            v = np.array([f[i] for i in new_ids if i in f], float) if f else np.array([])
            if v.size == 0:
                cells.append(f"{'not yet run':>22s}"); continue
            cells.append(f"{v.mean():8.1f} ({(v >= 21).mean()*100:3.0f}% fail)")
        print(f"{name:18s}" + "".join(cells))

print("\n=== Table 1: 785 -> 800 ===")
print(f"{'policy':18s}" + "".join(f"{b:>18s}" for b in BB))
old_t, new_t = {}, {}
for name, pat in ROWS:
    cells = []
    for bb in BB:
        a, b = load(bb, pat, False), load(bb, pat, True)
        if a is None or b is None:
            cells.append(f"{'NA':>18s}"); continue
        oa, nb = np.mean(list(a.values())), np.mean(list(b.values()))
        old_t[(name, bb)], new_t[(name, bb)] = oa, nb
        cells.append(f"{oa:6.2f} -> {nb:6.2f} ")
    print(f"{name:18s}" + "".join(cells))

print("\n=== does DGC still beat the strongest non-oracle baseline in all 24 cells? ===")
flips = []
checked = 0
for bb in BB:
    d = load(bb, ROWS[0][1], True)
    loaded = {n: load(bb, p, True) for n, p in BASES}
    if d is None or any(v is None for v in loaded.values()):
        print(f"  {bb}: baselines not all present yet, skipping")
        continue
    # a partially finished backbone mixes 800-row and 785-row files; comparing on the
    # intersection would silently drop the very instances under test, so require equality
    sizes = {len(d)} | {len(v) for v in loaded.values()}
    if len(sizes) > 1 or set(d) != set(next(iter(loaded.values()))):
        print(f"  {bb}: files not all at the same instance set {sorted(sizes)}, skipping")
        continue
    for ds in SETS:
        k = [i for i in d if i.startswith(ds + ":")]
        dv = np.mean([d[i] for i in k])
        best = min(np.mean([v[i] for i in k]) for v in loaded.values())
        checked += 1
        if dv >= best:
            flips.append((bb, ds, round(dv, 2), round(best, 2)))
print(f"  cells checked: {checked}/24   DGC no longer wins in: "
      f"{len(flips)}  {flips if flips else '(none)'}")
print("COMPARE800_DONE")
