# DGC — Disagreement-Guided Clicking

Code behind *DGC: Training-Free Next-Click Guidance from Multimask Disagreement in SAM*.

DGC picks the next corrective click for interactive segmentation from the disagreement
among SAM's three native multimask candidates: no training, no extra parameters, and no
decoder passes beyond the multimask pass SAM already performs.

This repository is a record of what produced the paper's numbers and figures. It is **not**
a self-contained release — model weights, datasets and result files are not included, and
the vendored SAM / SAM 2 / SimpleClick trees are omitted.

## Setup

Scripts resolve paths from environment variables:

```bash
export DGC_ROOT=/path/to/workspace     # must contain datasets/, weights/, results/
export DGC_SCRATCH=/path/to/scratch    # figure assets: render_seq.py, patch_fig5_label.py
```

Two interpreters were used: one for SAM / SAM 2, one for SimpleClick (the two have
incompatible dependency pins).

## Two conventions worth knowing before reading any number

**Failures score 20.** An instance that never reaches the IoU target within 20 clicks is
scored 20, as in the `compute_noc_metric` shared by RITM and SimpleClick. Because a genuine
20-click success and a failure are then the same number, failure detection must read the
**raw** value, where a failure is stored as 21 — scoring and failure detection are separate
operations throughout, and `allnumbers20.py` keeps them apart explicitly. `cap20_vs_21.py`
prints every headline number under both conventions; the earlier readouts listed at the
bottom of this file still score failures 21 and so do not match the paper.

**Read NoC from the columns, not the trajectory.** Use `n_clicks_90` / `n_clicks_85`, never
a value re-derived from `iou_traj` — the trajectory is stored rounded and crosses the 0.90
threshold 1–3 instances early per cell.

## Layout

| | |
|---|---|
| `runners/` | experiment entry points |
| `*.py`, `*.ps1` | figure generators and the scripts behind numbers quoted in the paper |

Results land in `$DGC_ROOT/results/`:

| directory | what it is |
|---|---|
| `noc_full` | **COCO-MVal for Table 1 and Figs. 2–3**, all 800 instances. |
| `noc_v2` | **the other five benchmarks.** All policies × 4 models. |
| `noc_sc_full`, `noc_sc` | **Table 2.** SimpleClick runs, same split. |
| `noc_sm` | single-mask protocol check. |
| `noc_seed` | Random-in-Ω_D under seeds 1 and 2. |

### The COCO-MVal split

The runners take `--min-area`, the minimum foreground area in pixels for an instance to be
evaluated. It defaults to 100, but **the paper reports `--min-area 0`**: all 800 instances
of the RITM COCO-MVal split, 1,621 in total across the six benchmarks.

The 15 instances the default would drop are failures for every policy, the oracle included,
so the filter was suppressing DGC's margin rather than inflating it: absolute NoC rises by
about 0.1 click for every policy, the paired gaps move by at most 0.02, and DGC still wins
all 24 dataset–model cells. `compare800.py` prints that comparison both ways.

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
| `fig2_heatmap.py` | Fig. 2 (per-cell reduction, with DGC's absolute NoC@90 in each cell) |
| `fig3_curves.py` | Fig. 3 (IoU vs clicks, all policies except the two ablations) |
| `render_seq.py`, `pick_seq.py`, `replay_seq.py` | Fig. 4 (click sequence) |
| `fig_stable_v2.py` | Fig. 5 (qualitative panel) |
| `patch_fig5_label.py` | repaints Fig. 5's bottom row label in pixel space — re-running `fig_stable_v2.py` re-reads the dump tree and picks different example rows, so the published panel is patched rather than regenerated |
| `make_fig1_assets.py` | crops feeding the Fig. 1 teaser (the teaser itself is hand-drawn) |

## Numbers quoted in the paper

| script | what it verifies |
|---|---|
| `allnumbers20.py` | **everything in Tables 1 and 2**: every cell, every paired CI, the 24-cell win check, failure rates, the both-solved decomposition, the policy and segmenter gaps, and the operating-regime gap closure |
| `regime800.py` | support coverage, pixel precision, and the operating-regime split |
| `derived800.py` | clicks-inside-error rates, the Ω_D = ∅ fallback never firing, the DGC-vs-SimpleClick paired CI |
| `calib800.py` | predicted vs true IoU after the first click |
| `check_curve_claim.py` | the Fig. 3 claim: DGC above every non-oracle baseline from click 2, on all four models |
| `floorshare.py` | how often each perturbation radius is set by its floor rather than its coefficient |
| `cap20_vs_21.py` | every headline number under both failure-scoring conventions |
| `compare800.py` | the filtered-vs-full COCO-MVal split, side by side |
| `gapdigits.py` | whether a quoted gap survives the rounding of the table entries it is read from |
| `dprecrec.py` | disagreement-support precision and recall, pixel level |
| `three.py`, `centroid_row.py`, `tab1_col_ci.py`, `decomp.py` | earlier readouts; these score failures **21** and predate the full split |
| `sm_readout.py`, `seed_readout.py` | single-mask protocol check; seed sensitivity of the ablation gap |
| `sc_perbench.py` | SimpleClick per-benchmark reproduction against published values |
| `calib.py`, `errhit_pooled.py`, `fallback_rate.py` | the pre-800 versions of three readouts above |

Each prints what it checked, and most assert completeness first, so a missing cell fails
loudly instead of silently averaging fewer instances.
