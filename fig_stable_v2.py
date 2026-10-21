"""ICASSP Fig. 4: qualitative operating-regime grid drawn from the SIX evaluated benchmarks.

Rows: 2 informative failures (disagreement overlaps the residual error) + 1 probe-stable
failure (S >= 0.5). Columns: Input + click | SAM mask | Disagreement support | Residual error.
Candidates are selected automatically from the SAM ViT-B first-click dumps with the same
S definition as the paper (rho=0.10, K=3); the chosen ids are printed for the record.
Writes $DGC_ROOT/figs_v2/fig_stable_v2.jpg.
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
from PIL import Image
from scipy.ndimage import distance_transform_edt

ROOT = Path(rf"{ROOT}")
DUMP = ROOT / "dumps" / "sam_vit_b"
MAN = {"grabcut": "GrabCut/manifest.csv", "berkeley": "Berkeley/manifest.csv",
       "davis": "DAVIS345/manifest.csv", "cocomval": "COCO_MVal/manifest.csv",
       "camo": "TestDataset/CAMO/manifest.csv", "chameleon": "TestDataset/CHAMELEON/manifest.csv"}
K = 3
VMAX = 2.0 / 3.0


def disag(c):
    return ((c[0] ^ c[1]).astype(np.float32) + (c[0] ^ c[2]) + (c[1] ^ c[2])) / 3


def iou(a, b, valid=None):
    if valid is not None:
        a, b = a & valid, b & valid
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


# ---------------- scan & score candidates ----------------
rows = []
for ds in MAN:
    for p in sorted((DUMP / ds / "click1").glob("*.npz")):
        with np.load(p) as z:
            d = {k: z[k] for k in z.files}
        gt, base = d["gt_mask"].astype(bool), d["base_mask"].astype(bool)
        valid = ~d["void_mask"].astype(bool) if "void_mask" in d else np.ones_like(gt)
        io = iou(base, gt, valid)
        if io >= 0.85:
            continue
        E = (base ^ gt) & valid
        ne = int(E.sum())
        if ne == 0:
            continue
        D0 = disag(d["cand_masks"].astype(bool)) == 0
        F0 = (d["jitter_masks"][:K] ^ base[None]).mean(0) == 0
        S = float((E & D0 & F0).sum()) / ne
        Dm = ~D0 & valid
        rec = float((Dm & E).sum()) / ne
        efrac = ne / max(int(((base | gt) & valid).sum()), 1)
        H, W = gt.shape
        rows.append(dict(ds=ds, id=p.stem, iou=io, S=S, rec=rec, efrac=efrac,
                         aspect=W / H, dsz=int(Dm.sum()) / max(ne, 1)))
print(f"failed masks scanned: {len(rows)}")

# informative: low S, high recall, moderate error, near-4:3 aspect, D not hugely over-covering
inf = [r for r in rows if r["S"] < 0.10 and r["rec"] > 0.75 and 0.08 < r["efrac"] < 0.45
       and 0.9 < r["aspect"] < 1.9 and r["dsz"] < 3.0 and 0.45 < r["iou"] < 0.8]
inf.sort(key=lambda r: (-r["rec"], r["S"]))
# probe-stable: S high, sizeable error, non-degenerate IoU (a mask that is at least on
# the object), moderate aspect. Faces are excluded by hand via EXCLUDE after screening.
import sys
EXCLUDE = {"berkeley_189011"}                     # child's face -- no faces in figures
FORCE_STABLE = None
for a in sys.argv[1:]:
    if a.startswith("--stable-id="):
        FORCE_STABLE = a.split("=", 1)[1]
stb = [r for r in rows if r["S"] >= 0.7 and r["efrac"] > 0.15 and 0.9 < r["aspect"] < 1.9
       and 0.2 <= r["iou"] < 0.75 and r["id"] not in EXCLUDE]
stb.sort(key=lambda r: (-r["S"], -r["efrac"]))
if FORCE_STABLE:
    stb = [r for r in rows if r["id"] == FORCE_STABLE] + stb

if "--sheet" in sys.argv:
    # contact sheet of probe-stable candidates for manual face screening
    man0 = {}
    for ds, m in MAN.items():
        with open(ROOT / "datasets" / m, encoding="utf-8") as f:
            man0[ds] = {row["id"]: row["img"] for row in csv.DictReader(f)}
    top = stb[:12]
    fig, axs = plt.subplots(3, 4, figsize=(14, 9))
    for ax, r in zip(axs.ravel(), top):
        img = np.array(Image.open(ROOT / "datasets" / man0[r["ds"]][r["id"]]).convert("RGB"))
        ax.imshow(img); ax.axis("off")
        ax.set_title(f"{r['id']}\nIoU={r['iou']:.2f} S={r['S']:.2f} efrac={r['efrac']:.2f}", fontsize=8)
    for ax in axs.ravel()[len(top):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(ROOT / "figs_v2" / "stable_candidates_sheet.jpg", dpi=110)
    print("SHEET saved; candidates:", [r["id"] for r in top])
    sys.exit(0)

print("\ninformative candidates (top 8):")
for r in inf[:8]:
    print(f"  {r['ds']:9s} {r['id']:40s} IoU={r['iou']:.2f} S={r['S']:.2f} rec={r['rec']:.2f} efrac={r['efrac']:.2f}")
print("probe-stable candidates (top 8):")
for r in stb[:8]:
    print(f"  {r['ds']:9s} {r['id']:40s} IoU={r['iou']:.2f} S={r['S']:.2f} rec={r['rec']:.2f} efrac={r['efrac']:.2f}")

# two informative rows from two different datasets, then one probe-stable
picks, seen = [], set()
for r in inf:
    if r["ds"] not in seen:
        picks.append(("Informative", r)); seen.add(r["ds"])
    if len(picks) == 2:
        break
picks.append(("Probe-stable", stb[0]))
print("\nPICKED:", [(t, r["ds"], r["id"]) for t, r in picks])

# ---------------- render ----------------
man = {}
for ds, m in MAN.items():
    with open(ROOT / "datasets" / m, encoding="utf-8") as f:
        man[ds] = {row["id"]: row["img"] for row in csv.DictReader(f)}

COLS = ["Input + click", "SAM mask", "Disagreement support", "Residual error"]
# row heights follow each image's aspect so wide images do not leave vertical gaps
loaded = []
for tag, r in picks:
    with np.load(DUMP / r["ds"] / "click1" / f"{r['id']}.npz") as z:
        d = {k: z[k] for k in z.files}
    img = np.array(Image.open(ROOT / "datasets" / man[r["ds"]][r["id"]]).convert("RGB"))
    gt = d["gt_mask"].astype(bool)
    if gt.shape != img.shape[:2]:
        img = np.array(Image.fromarray(img).resize((gt.shape[1], gt.shape[0])))
    loaded.append((d, img))
hr = [d["gt_mask"].shape[0] / d["gt_mask"].shape[1] for d, _ in loaded]   # H/W per row
fig_h = 2.3 * sum(hr) + 0.6
fig, axes = plt.subplots(3, 4, figsize=(9.6, fig_h), gridspec_kw=dict(height_ratios=hr))
for ri, (tag, r) in enumerate(picks):
    d, img = loaded[ri]
    gt, base, cm = d["gt_mask"].astype(bool), d["base_mask"].astype(bool), d["cand_masks"].astype(bool)
    gray = img.mean(2)
    dis = disag(cm)
    err = base ^ gt
    dtc = distance_transform_edt(gt)
    cy, cx = np.unravel_index(int(np.argmax(dtc)), dtc.shape)
    for ci, ax in enumerate(axes[ri]):
        if ci == 0:
            ax.imshow(img)
            ax.plot(cx, cy, "+", color="k", ms=17, mew=5.5)
            ax.plot(cx, cy, "+", color="#39ff14", ms=14, mew=2.6)
            ax.text(0.035, 0.96, f"IoU {r['iou']:.2f}", transform=ax.transAxes, fontsize=14,
                    va="top", ha="left", color="white", weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.74))
        elif ci == 1:
            ax.imshow(img)
            ax.imshow(np.ma.masked_where(~base, np.ones_like(base, float)),
                      cmap="autumn", vmin=0, vmax=1.4, alpha=0.45)
            ax.contour(base.astype(float), levels=[0.5], colors="#ff7f0e", linewidths=1.8)
        elif ci == 2:
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            ax.imshow(np.ma.masked_where(dis <= 0.02, dis), cmap="autumn_r",
                      alpha=0.9, vmin=0, vmax=VMAX)
            ax.contour((dis > 0.02).astype(float), levels=[0.5], colors="#6e0010",
                       linestyles="dashed", linewidths=1.6)
        else:
            ax.imshow(gray, cmap="gray", vmin=0, vmax=255)
            ax.imshow(np.ma.masked_where(~err, np.ones_like(err, float)),
                      cmap="cool", vmin=0, vmax=1.4, alpha=0.55)
            ax.contour(err.astype(float), levels=[0.5], colors="cyan", linewidths=1.6)
        ax.set_xticks([]); ax.set_yticks([])
        if ri == 0:
            ax.set_title(COLS[ci], fontsize=14, weight="bold", pad=6)

fig.subplots_adjust(left=0.055, right=0.995, top=0.945, bottom=0.012, wspace=0.03, hspace=0.06)
# place group labels / separator from the actual axes positions
b0, b1, b2 = (axes[i][0].get_position() for i in range(3))
y_inf = 0.5 * (b0.y1 + b1.y0)
y_sep = 0.5 * (b1.y0 + b2.y1)
y_stb = 0.5 * (b2.y0 + b2.y1)
fig.text(0.012, y_inf, "Informative", rotation=90, va="center", ha="center",
         fontsize=15, weight="bold", color="#b5560a")
fig.text(0.012, y_stb, "Probe-stable", rotation=90, va="center", ha="center",
         fontsize=14, weight="bold", color="#494949")
fig.add_artist(plt.Line2D([0.05, 0.997], [y_sep, y_sep], color="0.3", lw=1.4,
                          ls=(0, (6, 4)), transform=fig.transFigure))
out = ROOT / "figs_v2" / "fig_stable_v2.jpg"
fig.savefig(out, dpi=180, bbox_inches="tight", pil_kwargs={"quality": 92})
print(f"saved {out}")
print("FIGSTABLE_DONE")
