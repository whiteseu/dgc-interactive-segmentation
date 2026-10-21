"""Redraw Fig. 3: same eight curves, cleaner styling, larger on the page.

Three changes from the previous version:
  * designed at a narrower source width, so everything inside scales up when the figure is
    placed at \\columnwidth -- the panel is printed taller and the type is legible;
  * the four baselines that bunch together are thin solid lines in one cool family instead
    of four dash-dot strokes, which is what made the panel look busy;
  * chart junk removed: no top or right spine, horizontal grid only, short legend labels in
    two rows rather than three.

Asserts the claim the figure has to support before writing anything.
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
T, N_TOTAL = 20, 1621

CURVES = [
    ("noc_{bb}_risk_dt_{ds}.csv", "DGC (ours)", "#d1495b", "-", 3.2, 3.0),
    ("noc_{bb}_oracle_{ds}.csv", "oracle", "#2b2b2b", (0, (1, 1.6)), 2.2, 2.5),
    ("noc_{bb}_fps_{ds}.csv", "farthest-point", "#2e6f95", "--", 2.2, 2.0),
    ("noc_{bb}_random_{ds}.csv", "random", "#8d99ae", "--", 2.0, 1.5),
    ("noc_{bb}_bald_{ds}_rho20.csv", "pert-MI 0.2", "#4c9f70", "-", 1.5, 1.0),
    ("noc_{bb}_entropy_{ds}.csv", "entropy", "#79a9d1", "-", 1.5, 1.0),
    ("noc_{bb}_bald_{ds}.csv", "pert-MI 0.1", "#9d79bc", "-", 1.5, 1.0),
    ("noc_{bb}_boundary_{ds}.csv", "boundary", "#c99a4e", "-", 1.5, 1.0),
]


def curve(bb, pat):
    trajs = []
    for ds in SETS:
        root = FULL if ds == "cocomval" else V2
        for r in csv.DictReader(open(root / pat.format(bb=bb, ds=ds), encoding="utf-8")):
            t = json.loads(r["iou_traj"])
            trajs.append((t + [t[-1]] * (T - len(t)))[:T])
    assert len(trajs) == N_TOTAL, f"{pat}: pooled n={len(trajs)}, expected {N_TOTAL}"
    return np.array(trajs).mean(0)


ys = {lbl: curve("vit_b", pat) for pat, lbl, *_ in CURVES}
dgc = ys["DGC (ours)"]
for lbl, y in ys.items():
    if lbl in ("DGC (ours)", "oracle"):
        continue
    bad = [k + 1 for k in range(1, T) if dgc[k] <= y[k]]
    assert not bad, f"DGC not above {lbl} at clicks {bad}"
print("verified: DGC above every non-oracle baseline from click 2 to 20")

plt.rcParams.update({"font.size": 13, "axes.linewidth": 0.9,
                     "font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(6.2, 4.05))
x = range(1, T + 1)
for pat, lbl, color, ls, lw, z in CURVES:
    ax.plot(x, ys[lbl], linestyle=ls, color=color, label=lbl, lw=lw,
            zorder=z, solid_capstyle="round")

ax.set_xlabel("number of clicks", labelpad=6)
ax.set_ylabel("mean IoU", labelpad=6)
ax.set_xlim(1, T)
ax.set_xticks([1, 5, 10, 15, 20])
ax.tick_params(labelsize=12, length=3, width=0.9, colors="#333333")
ax.grid(axis="y", color="#d9d9d9", lw=0.7)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
for side in ("left", "bottom"):
    ax.spines[side].set_color("#999999")

# order the legend the way the curves stack at the right edge, so the key reads like the plot
order = sorted(ys, key=lambda k: -ys[k][-1])
handles, labels = ax.get_legend_handles_labels()
h = dict(zip(labels, handles))
ax.legend([h[l] for l in order], order, fontsize=11.5, loc="upper center",
          bbox_to_anchor=(0.5, -0.17), ncol=4, frameon=False,
          handlelength=1.9, columnspacing=1.4, handletextpad=0.6, labelspacing=0.5)
fig.tight_layout()
fig.savefig(OUT / "fig_noc_main.png", dpi=220, bbox_inches="tight")
print("saved", OUT / "fig_noc_main.png")
