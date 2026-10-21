"""Repaint Fig. 5's bottom row label from "Probe-stable" to "Non-informative".

The label is rasterised into the JPEG. Regenerating the figure is not an option here: the
selection logic re-reads the dump tree, which has since gained the restored COCO-MVal
instances, and re-running it picks different example rows (the figure height changes from
1029 to 949 px). So repaint in pixel space instead.

The two row labels are separated by colour, not position: the upper one is orange and the
lower one grey. The font size is calibrated by re-rendering the old string until its ink box
matches the measured one, so nothing about the size is assumed; the script refuses to write
if the calibrated render disagrees with the measurement, or if the longer replacement would
collide with the label above or leave the canvas.
"""

import os as _os
# Working directory for figure assets. Point DGC_SCRATCH at a directory holding the
# figure files this script reads and writes.
SCRATCH = _os.environ.get("DGC_SCRATCH", ".")
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib

D = PathSCRATCH
SRC = D / "code" / "paper_icassp_figures_fig_stable_v2.jpg"
DST = D / "fig_stable_v2_relabel.jpg"
OLD, NEW = "Probe-stable", "Non-informative"
COLOR = (0x49, 0x49, 0x49)
TTF = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans-Bold.ttf"
assert TTF.exists(), TTF

img = Image.open(SRC).convert("RGB")
a = np.asarray(img).astype(int)
H, W = a.shape[:2]
STRIP = 60

# grey ink only: dark, and nearly unsaturated (the upper label is orange)
sub = a[:, :STRIP]
dark = sub.max(axis=2) < 200
grey = (sub.max(axis=2) - sub.min(axis=2)) < 30
mask = dark & grey
ys, xs = np.where(mask)
assert len(ys), "no grey ink found in the left strip"
y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
print(f"old label ink box: x {x0}-{x1} (thick {x1-x0+1}), y {y0}-{y1} (len {y1-y0+1})")

# the orange label above, so the replacement can be checked for collision
om = dark & ~grey
oy = np.where(om.any(1))[0]
print(f"orange label ends at y {oy.max()}; canvas height {H}")


def ink_box(text, size):
    f = ImageFont.truetype(str(TTF), size)
    tmp = Image.new("L", (2000, 200), 255)
    ImageDraw.Draw(tmp).text((20, 40), text, font=f, fill=0)
    b = np.asarray(tmp) < 200
    r, c = np.where(b)
    return c.max() - c.min() + 1, r.max() - r.min() + 1   # length, thickness


lo, hi = 4, 120
for _ in range(40):
    mid = (lo + hi) // 2 or 1
    if ink_box(OLD, mid)[0] < (y1 - y0 + 1):
        lo = mid + 1
    else:
        hi = mid
S = lo
L, T = ink_box(OLD, S)
print(f"calibrated size {S} px: '{OLD}' renders {L} x {T}, measured {y1-y0+1} x {x1-x0+1}")
assert abs(L - (y1 - y0 + 1)) / (y1 - y0 + 1) < 0.06, "length calibration off"
assert abs(T - (x1 - x0 + 1)) / (x1 - x0 + 1) < 0.30, "thickness calibration off"

NL, NT = ink_box(NEW, S)
cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
ny0, ny1 = cy - NL / 2, cy + NL / 2
print(f"'{NEW}' renders {NL} x {NT} -> would span y {ny0:.0f}-{ny1:.0f}")
assert ny0 > oy.max() + 10, f"collides with the label above (ends at {oy.max()})"
assert ny1 < H - 5, "runs off the bottom of the canvas"

# repaint: clear the old ink, then draw the new string rotated the same way
out = img.copy()
ImageDraw.Draw(out).rectangle([x0 - 6, y0 - 6, x1 + 6, y1 + 6], fill=(255, 255, 255))

f = ImageFont.truetype(str(TTF), S)
tile = Image.new("RGB", (NL + 80, NT + 80), (255, 255, 255))
ImageDraw.Draw(tile).text((40, 40), NEW, font=f, fill=COLOR)
t = np.asarray(tile.convert("L")) < 200
r, c = np.where(t)
tile = tile.crop((c.min(), r.min(), c.max() + 1, r.max() + 1)).rotate(90, expand=True)
out.paste(tile, (int(round(cx - tile.width / 2)), int(round(cy - tile.height / 2))))
out.save(DST, quality=95, subsampling=0)
print("wrote", DST, f"{DST.stat().st_size} bytes, size {out.size}")

chk = np.asarray(Image.open(DST).convert("RGB")).astype(int)[:, :STRIP]
m2 = (chk.max(axis=2) < 200) & ((chk.max(axis=2) - chk.min(axis=2)) < 30)
ys2, xs2 = np.where(m2)
print(f"new label ink box: x {xs2.min()}-{xs2.max()}, y {ys2.min()}-{ys2.max()} "
      f"(centre {(ys2.min()+ys2.max())/2:.0f} vs old {cy:.0f})")
