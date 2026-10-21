"""Asset pack for redrawing Fig. 1 (method overview) with NON-FACE examples.

For each example instance (ViT-B, first-click dumps on the six benchmarks) this writes, at
full image resolution, every thumbnail the pipeline figure needs:

  01_input.png             original image
  02_input_click.png       image + first click marker (DT centre of GT, as in the protocol)
  03_cand_1.png / 2 / 3    the three native candidate masks m1,m2,m3 (white on black)
  04_selected_mask.png     working mask b = m_{k*} (white on black)
  05_selected_logits.png   REAL low-res logits of the selected candidate (SAM re-run,
                           same checkpoint), rendered as a heat map
  06_disagreement.png      D(u) on black background (red = 2/3), i.e. the support Omega_D
  06b_disagreement_overlay.png   D over the greyscale image
  07_support_largest_cc.png      largest connected component of Omega_D
  08_deepest_point.png     largest component + q* marker (distance-transform deepest point)
  09_refined_mask.png      SAM re-segmentation with the guided click appended (real run)
  09b_refined_overlay.png  refined mask over the image
  10_residual_error.png    residual error b XOR y before the guided click (cyan on grey)
  meta.json                ids, predicted IoUs, IoU before/after, click coordinates

All masks/maps come from the stored dumps; the logits and the refined mask are produced
by re-running the frozen SAM ViT-B with the identical prompts.
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import csv
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.ndimage import distance_transform_edt, label

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from segment_anything import sam_model_registry, SamPredictor

ROOT = Path(rf"{ROOT}")
OUT = ROOT / "fig1_assets_nonface"
MAN = {"grabcut": "GrabCut/manifest.csv", "chameleon": "TestDataset/CHAMELEON/manifest.csv",
       "berkeley": "Berkeley/manifest.csv", "davis": "DAVIS345/manifest.csv",
       "cocomval": "COCO_MVal/manifest.csv", "camo": "TestDataset/CAMO/manifest.csv"}
EXAMPLES = [("ocelot_grabcut", "grabcut", "grabcut_326038"),
            ("gecko_chameleon", "chameleon", "chameleon_animal-62")]

dev = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(ROOT / "weights/sam_vit_b_01ec64.pth")).to(dev)
pred = SamPredictor(sam)


def disag(c):
    return ((c[0] ^ c[1]).astype(np.float32) + (c[0] ^ c[2]) + (c[1] ^ c[2])) / 3


def iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


def save_bw(mask, path):
    Image.fromarray((mask.astype(np.uint8) * 255)).save(path)


def save_rgb(arr, path):
    Image.fromarray(arr.astype(np.uint8)).save(path)


def marker(ax, x, y, color="#39ff14"):
    ax.plot(x, y, "+", color="k", ms=26, mew=8)
    ax.plot(x, y, "+", color=color, ms=22, mew=4)


def fig_save(arr_rgb, path, overlay=None, cmap=None, alpha=0.8, contour=None,
             ccolor="#d62728", pts=None, vmax=None, dpi=200):
    h, w = arr_rgb.shape[:2]
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(arr_rgb)
    if overlay is not None:
        ax.imshow(overlay, cmap=cmap, alpha=alpha, vmin=0, vmax=vmax)
    if contour is not None:
        ax.contour(contour.astype(float), levels=[0.5], colors=ccolor, linewidths=2.2)
    if pts:
        for (x, y, col) in pts:
            marker(ax, x, y, col)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


for tag, ds, iid in EXAMPLES:
    od = OUT / tag
    od.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "datasets" / MAN[ds], encoding="utf-8") as f:
        img_rel = {r["id"]: r["img"] for r in csv.DictReader(f)}[iid]
    img = np.array(Image.open(ROOT / "datasets" / img_rel).convert("RGB"))
    with np.load(ROOT / "dumps" / "sam_vit_b" / ds / "click1" / f"{iid}.npz") as z:
        d = {k: z[k] for k in z.files}
    gt = d["gt_mask"].astype(bool)
    if gt.shape != img.shape[:2]:
        img = np.array(Image.fromarray(img).resize((gt.shape[1], gt.shape[0])))
    H, W = gt.shape
    gray = np.repeat(img.mean(2, keepdims=True), 3, axis=2).astype(np.uint8)
    cands = d["cand_masks"].astype(bool)
    piou = [float(x) for x in d["pred_iou"]]
    ksel = int(np.argmax(piou))
    b = cands[ksel]
    D = disag(cands)
    omega = D > 0
    lab, n = label(omega)
    sizes = np.bincount(lab.ravel()); sizes[0] = 0
    cc = lab == int(sizes.argmax())
    dt = distance_transform_edt(cc)
    qy, qx = np.unravel_index(int(np.argmax(dt)), dt.shape)
    dtg = distance_transform_edt(gt)
    cy, cx = np.unravel_index(int(np.argmax(dtg)), dtg.shape)
    pol = int(gt[qy, qx])

    # real SAM re-runs: (a) first click -> selected low-res logits; (b) + guided click
    pred.set_image(img)
    m1, s1, low1 = pred.predict(point_coords=np.array([[cx, cy]], np.float32),
                                point_labels=np.ones(1, np.int64), multimask_output=True,
                                return_logits=True)
    k1 = int(np.argmax(s1))
    low_sel = np.asarray(low1[k1])
    m2, s2, low2 = pred.predict(point_coords=np.array([[cx, cy], [qx, qy]], np.float32),
                                point_labels=np.array([1, pol], np.int64),
                                mask_input=low_sel[None], multimask_output=True,
                                return_logits=True)
    refined = np.asarray(m2[int(np.argmax(s2))]) > 0

    save_rgb(img, od / "01_input.png")
    fig_save(img, od / "02_input_click.png", pts=[(cx, cy, "#39ff14")])
    for i in range(3):
        save_bw(cands[i], od / f"03_cand_{i+1}.png")
    save_bw(b, od / "04_selected_mask.png")
    # logits heat map (256x256 native low-res, upsampled to image size for the thumbnail)
    # SAM's 256x256 low-res logits cover the image padded to a square (longest side 1024):
    # crop the valid region before resizing, otherwise the padding appears as a band.
    L = max(H, W)
    vh, vw = int(round(256 * H / L)), int(round(256 * W / L))
    lo = np.asarray(Image.fromarray(low_sel[:vh, :vw].astype(np.float32), mode="F").resize((W, H)))
    fig = plt.figure(figsize=(W / 200, H / 200), dpi=200); ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    ax.imshow(lo, cmap="RdBu_r", vmin=-np.abs(lo).max(), vmax=np.abs(lo).max()); fig.savefig(od / "05_selected_logits.png", dpi=200); plt.close(fig)
    # disagreement on black, red = 2/3
    blk = np.zeros_like(img)
    fig_save(blk, od / "06_disagreement.png", overlay=np.ma.masked_where(~omega, D),
             cmap="autumn_r", alpha=1.0, vmax=2 / 3)
    fig_save(gray, od / "06b_disagreement_overlay.png",
             overlay=np.ma.masked_where(~omega, D), cmap="autumn_r", alpha=0.85, vmax=2 / 3)
    save_bw(omega, od / "06c_support_bw.png")
    fig_save(blk, od / "07_support_largest_cc.png",
             overlay=np.ma.masked_where(~cc, np.ones_like(D)), cmap="autumn_r", alpha=1.0, vmax=1)
    fig_save(blk, od / "08_deepest_point.png",
             overlay=np.ma.masked_where(~cc, np.ones_like(D)), cmap="autumn_r", alpha=1.0,
             vmax=1, pts=[(qx, qy, "#39ff14" if pol else "#ff3b3b")])
    save_bw(refined, od / "09_refined_mask.png")
    fig_save(img, od / "09b_refined_overlay.png",
             overlay=np.ma.masked_where(~refined, np.ones_like(D)), cmap="Greens", alpha=0.55,
             vmax=1.6, contour=refined, ccolor="#137333")
    err = b ^ gt
    fig_save(gray, od / "10_residual_error.png", overlay=np.ma.masked_where(~err, np.ones_like(D)),
             cmap="cool", alpha=0.6, vmax=1.4, contour=err, ccolor="cyan")
    meta = dict(dataset=ds, id=iid, image=img_rel, size=[W, H],
                predicted_iou=[round(x, 3) for x in piou], selected_candidate=ksel + 1,
                iou_first_click=round(iou(b, gt), 3), iou_after_guided_click=round(iou(refined, gt), 3),
                first_click_xy=[int(cx), int(cy)], guided_click_xy=[int(qx), int(qy)],
                guided_click_polarity="foreground" if pol else "background",
                D_values="{0, 2/3}", omega_D_pixels=int(omega.sum()), largest_cc_pixels=int(cc.sum()))
    (od / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(tag, json.dumps(meta))
print("ASSETS_DONE")
