# DGC — Disagreement-Guided Clicking

Code behind *DGC: Training-Free Next-Click Guidance from Multimask Disagreement in SAM*.

DGC picks the next corrective click for interactive segmentation from the disagreement
among SAM's three native multimask candidates: no training, no extra parameters, and no
decoder passes beyond the multimask pass SAM already performs.

This repository is a record of what produced the paper's numbers and figures. It is **not**
a self-contained release — model weights, datasets and result files are not included, and
the vendored SAM / SAM 2 / SimpleClick trees are omitted.

## Setup

Scripts resolve paths from an environment variable:

```bash
export DGC_ROOT=/path/to/workspace     # must contain datasets/, weights/, results/
export DGC_SCRATCH=/path/to/scratch    # only render_seq.py needs this
```

Two interpreters were used: one for SAM / SAM 2, one for SimpleClick (the two have
incompatible dependency pins).

## Layout

| | |
|---|---|
| `runners/` | experiment entry points |
| `*.py`, `*.ps1` | figure generators and the scripts behind numbers quoted in the paper |

Results land in `$DGC_ROOT/results/`:

| directory | what it is |
|---|---|
| `noc_full` | **COCO-MVal for Table 1 and Figs. 2–3**, all 800 instances. |
| `noc_v2` | **the other five benchmarks.** All policies × 4 backbones. |
| `noc_sc_full`, `noc_sc` | **Table 2.** SimpleClick runs, same split. |
| `noc_sm` | single-mask protocol check. |
| `noc_seed` | Random-in-Ω_D under seeds 1 and 2. |

Read NoC from the `n_clicks_90` / `n_clicks_85` columns, **not** by re-deriving it from
`iou_traj` — the trajectory is stored rounded and crosses the 0.90 threshold 1–3 instances
early per cell.

### The COCO-MVal split

The runners take `--min-area`, the minimum foreground area in pixels for an instance to be
evaluated. It defaults to 100, but **the paper reports `--min-area 0`**: all 800 instances
of the RITM COCO-MVal split, 1,621 in total across the six benchmarks.

The 15 instances the default would drop are failures for every policy, the oracle included,
so the filter was suppressing DGC's margin rather than inflating it: absolute NoC rises by
about 0.1 click for every policy, the paired gaps move by at most 0.02, and DGC still wins
all 24 dataset–backbone cells. `compare800.py` prints that comparison both ways.

## Entry points

| script | produces |
|---|---|
| `runners/run_noc.py` | every policy in Table 1. `--single-mask` switches rounds t≥2 to single-mask decoding. |
| `runners/run_sam.py`, `runners/run_sam2.py` | instance loading, first click, ground-truth alignment, first-click dumps. |
| `runners/run_logits.py` | low-resolution logits and box-perturbation masks the baselines need. |
| `runners/prep_datasets.py` | builds each dataset's `manifest.csv`. |
| `run_noc_sc.py` + `sc_shim.py` | SimpleClick runs for Table 2. |
| `full800.ps1`, `sc_full800.ps1`, `dumps800.ps1` | the batch invocations behind the reported split. |
| `sc_batch*.ps1`, `sm_batch.ps1`, `seed_batch.ps1` | earlier batch invocations. |

## Figures

| script | figure |
|---|---|
| `figs800.py` | Fig. 2 (heatmap), Fig. 3 (IoU curves) |
| `render_seq.py`, `pick_seq.py`, `replay_seq.py` | Fig. 4 (click sequence) |
| `fig_stable_v2.py` | Fig. 5 (qualitative panel) |
| `make_fig1_assets.py` | crops feeding the Fig. 1 teaser (the teaser itself is hand-drawn) |

## Numbers quoted in the paper

| script | what it verifies |
|---|---|
| `recompute800.py` | every Table 1 value, every paired bootstrap CI, the 24-cell win check, failure rates |
| `tab2_800.py` | every Table 2 value and the three policy/segmenter gaps |
| `derived800.py` | clicks-inside-error rates (42–54%), the Ω_D = ∅ fallback never firing, the DGC-vs-SimpleClick paired CI |
| `regime800.py` | support coverage (71–79%), pixel precision (0.20–0.37), the operating-regime split and its gap closure |
| `calib800.py` | predicted vs true IoU after the first click (0.91 / 0.53) |
| `decomp.py` | the both-solved decomposition (793 instances, 2.25 vs 3.40, 18–44%) |
| `gapdigits.py` | whether a quoted gap survives the rounding of the table entries it is read from |
| `compare800.py` | the filtered-vs-full split, side by side |
| `three.py` | per-benchmark oracle NoC, failure-rate split, per-cell ablation |
| `dprecrec.py` | disagreement-support precision and recall, pixel level |
| `centroid_row.py`, `tab1_col_ci.py` | earlier single-split readouts of the Table 1 rows and CIs |
| `sm_readout.py`, `seed_readout.py` | single-mask protocol check; seed sensitivity of the ablation gap |
| `sc_perbench.py` | SimpleClick per-benchmark reproduction against published values |
| `calib.py`, `errhit_pooled.py`, `fallback_rate.py` | the pre-800 versions of the three readouts above |

Each prints what it checked, and most assert completeness first, so a missing cell fails
loudly instead of silently averaging fewer instances.
