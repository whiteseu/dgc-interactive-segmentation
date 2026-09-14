"""Redraw the two ICASSP result figures on the full 800-instance COCO-MVal split.

Two changes from figs_v2.py:
  * COCO-MVal is read from results/noc_full, the other five benchmarks from results/noc_v2;
  * the heatmap's baseline set now includes perturbation-MI at rho=0.2. Its files carry a
    _rho20 suffix that the old name pattern could not express, so that policy was silently
    absent while the caption claimed the strongest non-oracle baseline -- in any cell where
    rho=0.2 is the strongest, the old figure overstated the reduction.

Asserts the pooled instance count and that the matrix matches the range quoted in Sec. 3.2.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V2 = Path(f"{ROOT}/results/noc_v2")
FULL = Path(f"{ROOT}/results/noc_full")
OUT = Path(f"{ROOT}/figs_800")
OUT.mkdir(parents=True, exist_ok=True)

SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
SET_LBL = ["GrabCut", "Berkeley", "DAVIS", "COCO-MVal", "CAMO", "CHAMELEON"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
BB_LBL = ["ViT-B", "ViT-L", "ViT-H", "SAM 2"]
T = 20
N_TOTAL = 1621


def rows(bb, pat, ds):
    root = FULL if ds == "cocomval" else V2
    return list(csv.DictReader(open(root / pat.format(bb=bb, ds=ds), encoding="utf-8")))


# ---------------- curve panel: mean IoU vs clicks, ViT-B pooled ----------------
STRATS = [("noc_{bb}_risk_dt_{ds}.csv", "disagreement-guided (ours)", "#d62728", "-", 3.0),
          ("noc_{bb}_fps_{ds}.csv", "farthest-point", "#1f77b4", "--", 2.0),
          ("noc_{bb}_random_{ds}.csv", "random", "#7f7f7f", "--", 2.0),
          ("noc_{bb}_oracle_{ds}.csv", "oracle", "k", ":", 2.4)]

plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(7.6, 3.5))
for pat, lbl, color, ls, lw in STRATS:
    trajs = []
    for ds in SETS:
        for r in rows("vit_b", pat, ds):
            t = json.loads(r["iou_traj"])
            t = t + [t[-1]] * (T - len(t))       # carry forward after reaching target
            trajs.append(t[:T])
    assert len(trajs) == N_TOTAL, f"{lbl}: pooled n={len(trajs)}, expected {N_TOTAL}"
    ax.plot(range(1, T + 1), np.array(trajs).mean(0), ls, color=color,
            label=lbl, lw=lw + 0.6)
    print(f"curve {lbl:28s}: n={len(trajs)}")
ax.set_xlabel("number of clicks")
ax.set_ylabel("mean IoU")
ax.set_xticks([1, 5, 10, 15, 20])
ax.tick_params(labelsize=13)
ax.grid(alpha=0.25)
# legend below the axes so it never covers the curve tails at large click counts
ax.legend(fontsize=12, loc="upper center", bbox_to_anchor=(0.5, -0.26), ncol=4,
          frameon=False, handlelength=1.8, columnspacing=1.1)
fig.tight_layout()
fig.savefig(OUT / "fig_noc_main.png", dpi=200, bbox_inches="tight")
print("saved fig_noc_main.png")

# ---------------- heatmap: per-cell clicks saved vs strongest baseline ----------------
BASES = ["noc_{bb}_fps_{ds}.csv", "noc_{bb}_random_{ds}.csv", "noc_{bb}_entropy_{ds}.csv",
         "noc_{bb}_bald_{ds}.csv", "noc_{bb}_bald_{ds}_rho20.csv",
         "noc_{bb}_boundary_{ds}.csv"]


def m90(bb, pat, ds):
    v = [min(int(r["n_clicks_90"]), 21) for r in rows(bb, pat, ds)]
    return sum(v) / len(v)


M = np.zeros((len(SETS), len(BB)))
for i, ds in enumerate(SETS):
    for j, bb in enumerate(BB):
        M[i, j] = min(m90(bb, s, ds) for s in BASES) - m90(bb, "noc_{bb}_risk_dt_{ds}.csv", ds)
print("heatmap matrix (clicks saved):")
for i, ds in enumerate(SETS):
    print("   ", [round(x, 2) for x in M[i]], "#", ds)
assert (M > 0).all(), "a cell went non-positive -- the 24/24 claim would be wrong!"
print(f"range {M.min():.2f}-{M.max():.2f}")
assert abs(M.min() - 0.42) < 0.005 and abs(M.max() - 3.28) < 0.005, \
    "matrix disagrees with the range recomputed in Sec. 3.2"

plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(7.6, 3.4))
M = M.T   # rows: backbones, cols: datasets -- wide grid for a single column
im = ax.imshow(M, cmap="Greens", vmin=0, vmax=max(4.0, M.max()))
for i in range(len(BB)):
    for j in range(len(SETS)):
        ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                fontsize=14, weight="bold",
                color="white" if M[i, j] > 0.6 * M.max() else "#1a4d1a")
ax.set_xticks(range(len(SETS)), SET_LBL, fontsize=12, rotation=18, ha="right")
ax.set_yticks(range(len(BB)), BB_LBL, fontsize=12)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.022, pad=0.02)
cb.set_label("NoC@90 reduction (clicks) $\\uparrow$", fontsize=12)
cb.ax.tick_params(labelsize=11)
fig.tight_layout()
fig.savefig(OUT / "fig_noc_heatmap.png", dpi=200, bbox_inches="tight")
print("saved fig_noc_heatmap.png")
print("FIGS_800_DONE")
