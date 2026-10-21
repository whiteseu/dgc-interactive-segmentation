"""Render Fig. 5: click-by-click comparison of DGC against farthest-point.

Built at the final printed size (one spconf column = 241.2 pt), so the font sizes
set here are the sizes that reach the page. The full frame is kept rather than
cropped to the object: farthest-point's clicks land on the image corners, and
cropping them away would hide exactly what the figure is about.
"""
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt              # noqa: E402
from matplotlib.patches import Rectangle     # noqa: E402

# Directory holding seq_00114.npz, the click sequences dumped by replay_seq.py.
SCR = Path(os.environ.get("DGC_SCRATCH", "."))
Z = np.load(SCR / "seq_00114.npz")
ROWS = [("risk_dt", "DGC (ours)", "#b34700"), ("fps", "Farthest-point", "0.30")]
SHOW = [1, 3, 5]
COL_PT = 241.2
MASK_RGB = np.array([255, 138, 0])

img = Z["image"]
ih, iw = img.shape[:2]

lab_w, head_f = 0.055, 0.030               # label strip / header, width fractions
panel_w = (1 - lab_w) / len(SHOW)
fig_w_in = COL_PT / 72.0
panel_w_in = fig_w_in * panel_w
panel_h_in = panel_w_in * ih / iw
head_in = fig_w_in * head_f
fig_h_in = 2 * panel_h_in + head_in

fig = plt.figure(figsize=(fig_w_in, fig_h_in), dpi=600)
for r, (strat, rlabel, rcol) in enumerate(ROWS):
    for j, t in enumerate(SHOW):
        ax = fig.add_axes([lab_w + j * panel_w,
                           1 - (head_in + (r + 1) * panel_h_in) / fig_h_in,
                           panel_w, panel_h_in / fig_h_in])
        a = img.astype(np.float32).copy()
        m = Z[f"{strat}_mask_{t}"]
        a[m] = 0.55 * a[m] + 0.45 * MASK_RGB
        ax.imshow(a.astype(np.uint8), interpolation="bilinear", aspect="auto")
        ax.contour(m.astype(float), levels=[0.5], colors=[(1, .54, 0)],
                   linewidths=0.5)
        pts, lbs = Z[f"{strat}_pts_{t}"], Z[f"{strat}_lbs_{t}"]
        for i, ((px, py), lb) in enumerate(zip(pts, lbs)):
            newest = (i == len(pts) - 1)
            ax.plot(px, py, marker="+" if lb == 1 else "x",
                    color="#00e63c" if lb == 1 else "#ff2020",
                    ms=5.0 if newest else 3.2,
                    mew=1.3 if newest else 0.8,
                    clip_on=True, zorder=5)
        # farthest-point clicks land on the image corners; pad the view a little
        # so those markers are not sliced in half by the panel edge.
        mx, my = 0.035 * iw, 0.035 * ih
        ax.set_xlim(-mx, iw + mx); ax.set_ylim(ih + my, -my)
        v = float(Z[f"{strat}_iou_{t}"])
        ax.add_patch(Rectangle((0.02, 0.80), 0.30, 0.175, transform=ax.transAxes,
                               facecolor="black", alpha=0.62, lw=0, zorder=6))
        ax.text(0.045, 0.838, f"{v:.2f}", transform=ax.transAxes, color="white",
                fontsize=4.8, weight="bold", zorder=7)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("0.4")
        if r == 0:
            ax.set_title(f"after click {t}", fontsize=5.2, pad=1.5)
    fig.text(lab_w * 0.42,
             1 - (head_in + (r + 0.5) * panel_h_in) / fig_h_in,
             rlabel, ha="center", va="center", rotation=90,
             fontsize=5.2, weight="bold", color=rcol)

out = SCR / "fig_seq.png"
fig.savefig(out, dpi=600, facecolor="white")
print("WROTE", out)
print(f"figure {fig_w_in*72:.1f} x {fig_h_in*72:.1f} pt, panel {panel_w_in*72:.1f} pt wide")
for s, _, _ in ROWS:
    print(f"  {s:8s}", [round(float(Z[f'{s}_iou_{t}']), 3) for t in SHOW])
