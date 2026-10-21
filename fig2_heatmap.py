"""Fig. 2 with DGC's absolute per-benchmark NoC@90 under each cell's reduction.

The paper otherwise reports only pooled NoC and per-cell reductions, so a reader cannot
line either up against published per-benchmark SAM numbers. Printing DGC's own NoC@90 in
parentheses inside the cell adds the missing 24 values without taking any page area.

The reduction is against the strongest of the six training-free baselines in that cell, at
the failure-scoring convention the paper reports (failures score 20).
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V2 = Path(f"{ROOT}/results/noc_v2")
FULL = Path(f"{ROOT}/results/noc_full")
OUT = Path(f"{ROOT}/figs_800")
SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
SET_LBL = ["GrabCut", "Berkeley", "DAVIS", "COCO-MVal", "CAMO", "CHAMELEON"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
BB_LBL = ["ViT-B", "ViT-L", "ViT-H", "SAM 2"]
CAP = 20
BASES = ["noc_{bb}_fps_{ds}.csv", "noc_{bb}_random_{ds}.csv", "noc_{bb}_entropy_{ds}.csv",
         "noc_{bb}_bald_{ds}.csv", "noc_{bb}_bald_{ds}_rho20.csv",
         "noc_{bb}_boundary_{ds}.csv"]


def m90(bb, pat, ds):
    root = FULL if ds == "cocomval" else V2
    v = []
    for r in csv.DictReader(open(root / pat.format(bb=bb, ds=ds), encoding="utf-8")):
        try:
            v.append(min(int(r["n_clicks_90"]), CAP))
        except (ValueError, KeyError):
            v.append(CAP)
    return sum(v) / len(v)


D = np.zeros((len(BB), len(SETS)))     # reduction
A = np.zeros((len(BB), len(SETS)))     # DGC absolute
for j, bb in enumerate(BB):
    for i, ds in enumerate(SETS):
        A[j, i] = m90(bb, "noc_{bb}_risk_dt_{ds}.csv", ds)
        D[j, i] = min(m90(bb, s, ds) for s in BASES) - A[j, i]
print("DGC NoC@90 per cell:")
for j, bb in enumerate(BB):
    print(f"  {bb:6s}" + "".join(f"{A[j, i]:7.2f}" for i in range(len(SETS))))
assert (D > 0).all() and abs(D.min() - 0.44) < 0.005 and abs(D.max() - 3.14) < 0.005

plt.rcParams.update({"font.size": 14})
fig, ax = plt.subplots(figsize=(6.4, 3.5))
im = ax.imshow(D, cmap="Greens", vmin=0, vmax=max(4.0, D.max()))
for j in range(len(BB)):
    for i in range(len(SETS)):
        c = "white" if D[j, i] > 0.6 * D.max() else "#1a4d1a"
        ax.text(i, j - 0.16, f"{D[j, i]:.1f}", ha="center", va="center",
                fontsize=13, weight="bold", color=c)
        ax.text(i, j + 0.24, f"({A[j, i]:.1f})", ha="center", va="center",
                fontsize=10.5, color=c)
# 18 degrees ran the longer benchmark names into each other at column width
ax.set_xticks(range(len(SETS)), SET_LBL, fontsize=11.5, rotation=28, ha="right")
ax.set_yticks(range(len(BB)), BB_LBL, fontsize=11.5)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
cb.set_label("NoC@90 reduction (clicks) $\\uparrow$", fontsize=11)
cb.ax.tick_params(labelsize=10)
fig.tight_layout()
fig.savefig(OUT / "fig_noc_heatmap.png", dpi=220, bbox_inches="tight")
im2 = plt.imread(OUT / "fig_noc_heatmap.png")
print(f"\nsaved, {im2.shape[1]}x{im2.shape[0]} px -> on page 243.8 x "
      f"{243.8 * im2.shape[0] / im2.shape[1]:.1f} pt")
