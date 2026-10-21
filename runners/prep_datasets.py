"""Unpack the evaluation datasets and write one manifest.csv per dataset.

Each dataset directory gets a manifest.csv with the columns:
  id   unique, filename-safe instance name
  img  image path, relative to data_root
  gt   foreground binary mask png, relative to data_root
  void ignore-region mask png, relative to data_root; may be empty (GrabCut's 128 band)

Source layouts, each verified against the archives:
  GrabCut.zip    data_GT/ = images (mixed jpg|bmp|JPG), boundary_GT/ = masks
                 (bmp, values {0, 128, 255})
  Berkeley.zip   images/<stem>.jpg paired with masks/<stem>.png by name, including
                 multi-instance stems such as 189011a/b
  DAVIS.zip      DAVIS345/img/<n>.jpg + DAVIS345/gt/<n>.png
  COCO_MVal.zip  COCO_MVal/img and COCO_MVal/gt paired by name (filenames contain spaces)
  sbd_benchmark.tgz  benchmark_RELEASE/dataset/{img,inst}/ and val.txt; inst holds .mat
                 files with multiple instances

Usage: python prep_datasets.py --data-root $DGC_ROOT/datasets
"""

import os as _os
# Repository root on the machine the experiments were run on. Point DGC_ROOT at a
# checkout that also holds datasets/, weights/ and results/ to re-run anything here.
ROOT = _os.environ.get("DGC_ROOT", ".")
import argparse
import csv
import tarfile
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image


def write_manifest(ds_dir: Path, rows):
    with open(ds_dir / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "img", "gt", "void"])
        w.writeheader()
        w.writerows(rows)
    print(f"[{ds_dir.name}] manifest: {len(rows)} instances", flush=True)


def unzip(zip_path: Path, dest: Path):
    if not zip_path.exists():
        print(f"SKIP missing {zip_path}", flush=True)
        return False
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(dest)
    return True


def prep_grabcut(root: Path):
    if not unzip(root / "GrabCut.zip", root):
        return
    ds = root / "GrabCut"
    gt_norm = ds / "gt_norm"; gt_norm.mkdir(exist_ok=True)
    void_dir = ds / "void"; void_dir.mkdir(exist_ok=True)
    images = {p.stem: p for p in (ds / "data_GT").iterdir() if p.is_file()}
    rows = []
    for mp in sorted((ds / "boundary_GT").glob("*.bmp")):
        img = images.get(mp.stem)
        if img is None:
            print(f"WARN no image for {mp.stem}", flush=True)
            continue
        m = np.array(Image.open(mp).convert("L"))
        fg, void = m >= 200, (m > 50) & (m < 200)   # 255=fg, 128=ignore
        Image.fromarray(fg.astype(np.uint8) * 255).save(gt_norm / f"{mp.stem}.png")
        row_void = ""
        if void.any():
            Image.fromarray(void.astype(np.uint8) * 255).save(void_dir / f"{mp.stem}.png")
            row_void = f"GrabCut/void/{mp.stem}.png"
        rows.append(dict(id=f"grabcut_{mp.stem}", img=f"GrabCut/data_GT/{img.name}",
                         gt=f"GrabCut/gt_norm/{mp.stem}.png", void=row_void))
    write_manifest(ds, rows)


def pair_dir(root: Path, ds_name: str, img_sub: str, gt_sub: str, prefix: str):
    ds = root / ds_name
    images = {p.stem: p for p in (ds / img_sub).iterdir() if p.is_file()}
    rows = []
    for gp in sorted((ds / gt_sub).glob("*.png")):
        img = images.get(gp.stem)
        if img is None:
            print(f"WARN no image for {gp.stem}", flush=True)
            continue
        safe = gp.stem.replace(" ", "-")
        rows.append(dict(id=f"{prefix}_{safe}", img=f"{ds_name}/{img_sub}/{img.name}",
                         gt=f"{ds_name}/{gt_sub}/{gp.name}", void=""))
    write_manifest(ds, rows)


def prep_pairs(root: Path, zip_name: str, ds_name: str, img_sub: str, gt_sub: str, prefix: str):
    if not unzip(root / zip_name, root):
        return
    pair_dir(root, ds_name, img_sub, gt_sub, prefix)


def prep_cod(root: Path):
    """The three COD test sets: TestDataset/{COD10K,CAMO,CHAMELEON}/{Imgs,GT}."""
    if not (root / "TestDataset").exists():
        if not unzip(root / "cod_testdataset.zip", root):
            return
    for name, prefix in (("COD10K", "cod10k"), ("CAMO", "camo"), ("CHAMELEON", "chameleon")):
        pair_dir(root, f"TestDataset/{name}", "Imgs", "GT", prefix)


def prep_sbd(root: Path):
    tgz = root / "sbd_benchmark.tgz"
    if not tgz.exists():
        print("SKIP missing sbd_benchmark.tgz", flush=True)
        return
    if not (root / "benchmark_RELEASE").exists():
        with tarfile.open(tgz) as t:
            t.extractall(root)
    from scipy.io import loadmat
    base = root / "benchmark_RELEASE" / "dataset"
    inst_out = root / "SBD_inst"; inst_out.mkdir(exist_ok=True)
    val_ids = (base / "val.txt").read_text().split()
    rows = []
    for stem in val_ids:
        mat = loadmat(str(base / "inst" / f"{stem}.mat"))
        seg = mat["GTinst"]["Segmentation"][0, 0]      # [H, W], instance labels 0..K
        for k in range(1, int(seg.max()) + 1):
            m = seg == k
            if m.sum() < 100:
                continue
            name = f"{stem}__inst{k}.png"
            Image.fromarray(m.astype(np.uint8) * 255).save(inst_out / name)
            rows.append(dict(id=f"sbd_{stem}_inst{k}", img=f"benchmark_RELEASE/dataset/img/{stem}.jpg",
                             gt=f"SBD_inst/{name}", void=""))
    write_manifest(root / "benchmark_RELEASE", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, required=True)
    ap.add_argument("--only", default="", help="comma-separated list of datasets to process")
    args = ap.parse_args()
    root = args.data_root
    only = set(args.only.split(",")) if args.only else None

    def want(n):
        return only is None or n in only

    if want("grabcut"):
        prep_grabcut(root)
    if want("berkeley"):
        prep_pairs(root, "Berkeley.zip", "Berkeley", "images", "masks", "berkeley")
    if want("davis"):
        prep_pairs(root, "DAVIS.zip", "DAVIS345", "img", "gt", "davis")
    if want("cocomval"):
        prep_pairs(root, "COCO_MVal.zip", "COCO_MVal", "img", "gt", "cocomval")
    if want("sbd"):
        prep_sbd(root)
    if want("cod"):
        prep_cod(root)
    print("PREP_DONE", flush=True)


if __name__ == "__main__":
    main()
