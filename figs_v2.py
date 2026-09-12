"""Redraw the two ICASSP result figures from the v2 (single-code-revision) grid.

Fixes two data-provenance issues in the ACCV-era figures:
  * the old curve panel plotted the legacy centroid `risk` strategy, not the
    published deepest-point `risk_dt`;
  * both figures were rendered from pre-rebuild CSVs, which differ from the v2
    numbers by up to ~0.2 clicks on the baseline rows.
Also drops the "Fewer clicks to the target" title a reviewer flagged.

Outputs $DGC_ROOT/figs_v2/fig_noc_main.png and fig_noc_heatmap.png.
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

V2 = Path(rf"{ROOT}/results/noc_v2")
OUT = Path(rf"{ROOT}/figs_v2")
OUT.mkdir(parents=True, exist_ok=True)

SETS = ["grabcut", "berkeley", "davis", "cocomval", "camo", "chameleon"]
SET_LBL = ["GrabCut", "Berkeley", "DAVIS", "COCO-MVal", "CAMO", "CHAMELEON"]
BB = ["vit_b", "vit_l", "vit_h", "sam2"]
BB_LBL = ["ViT-B", "ViT-L", "ViT-H", "SAM 2"]
T = 20


def rows(bb, strat, ds):
    p = V2 / f"noc_{bb}_{strat}_{ds}.csv"
    return list(csv.DictReader(open(p, encoding="utf-8")))


# ---------------- curve panel: mean IoU vs clicks, ViT-B pooled ----------------
STRATS = [("risk_dt", "disagreement-guided (ours)", "#d62728", "-", 3.0),
          ("fps", "farthest-point", "#1f77b4", "--", 2.0),
          ("random", "random", "#7f7f7f", "--", 2.0),
          ("oracle", "oracle", "k", ":", 2.4)]

plt.rcParams.update({"font.size": 15})
fig, ax = plt.subplots(figsize=(7.6, 3.5))
for strat, lbl, color, ls, lw in STRATS:
    trajs = []
    for ds in SETS:
        for r in rows("vit_b", strat, ds):
            t = json.loads(r["iou_traj"])
            t = t + [t[-1]] * (T - len(t))       # carry forward after reaching target
            trajs.append(t[:T])
    ax.plot(range(1, T + 1), np.array(trajs).mean(0), ls, color=color,
            label=lbl, lw=lw + 0.6)
    print(f"curve {strat:8s}: n={len(trajs)}")
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

# ---------------- heatmap: per-cell clicks saved vs best baseline ----------------
# Every evaluated non-oracle baseline, farthest-point INCLUDED. It was missing
# here while the caption claimed "best evaluated non-oracle baseline", which
# inflated the cells where fps is the strongest baseline (ViT-B/GrabCut showed
# +5.4 instead of the honest +2.5).
BASES = ["random", "entropy", "boundary", "bald", "fps"]


def m90(bb, strat, ds):
    v = [min(int(r["n_clicks_90"]), 21) for r in rows(bb, strat, ds)]
    return sum(v) / len(v)


M = np.zeros((len(SETS), len(BB)))
for i, ds in enumerate(SETS):
    for j, bb in enumerate(BB):
        rk = m90(bb, "risk_dt", ds)
        best = min(m90(bb, s, ds) for s in BASES)
        M[i, j] = best - rk
print("v2 heatmap matrix (clicks saved):")
for i, ds in enumerate(SETS):
    print("   ", [round(x, 1) for x in M[i]], "#", ds)
assert (M > 0).all(), "a cell went non-positive -- the 24/24 claim would be wrong!"

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
print("FIGS_V2_DONE")
