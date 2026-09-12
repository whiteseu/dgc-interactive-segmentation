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
| `noc_v2` | **Table 1 and Figs. 2–3.** All policies × 4 backbones × 6 benchmarks. |
| `noc_sc` | **Table 2.** SimpleClick runs. |
| `noc_sm` | single-mask protocol check. |
| `noc_seed` | Random-in-Ω_D under seeds 1 and 2. |

Read NoC from the `n_clicks_90` / `n_clicks_85` columns, **not** by re-deriving it from
`iou_traj` — the trajectory is stored rounded and crosses the 0.90 threshold 1–3 instances
early per cell.

## Entry points

| script | produces |
|---|---|
| `runners/run_noc.py` | every policy in Table 1. `--single-mask` switches rounds t≥2 to single-mask decoding. |
| `runners/run_sam.py` | instance loading, first click, ground-truth alignment. Drops instances with <100 foreground px. |
| `runners/prep_datasets.py` | builds each dataset's `manifest.csv`. |
| `run_noc_sc.py` + `sc_shim.py` | SimpleClick runs for Table 2. |
| `sc_batch*.ps1`, `sm_batch.ps1`, `seed_batch.ps1` | the exact batch invocations used. |

## Figures

| script | figure |
|---|---|
| `figs_v2.py` | Fig. 2 (heatmap), Fig. 3 (IoU curves) |
| `render_seq.py`, `pick_seq.py`, `replay_seq.py` | Fig. 4 (click sequence) |
| `fig_stable_v2.py` | Fig. 5 (qualitative panel) |
| `make_fig1_assets.py` | crops feeding the Fig. 1 teaser (the teaser itself is hand-drawn) |

## Numbers quoted in the paper

| script | what it verifies |
|---|---|
| `tab1_col_ci.py` | every Table 1 value and every paired bootstrap CI |
| `centroid_row.py` | the Centroid-in-C row; also per-row failure rates |
| `decomp.py` | the both-solved decomposition (793 instances, 2.25 vs 3.40, 18–44%) |
| `calib.py` | predicted vs true IoU after the first click (0.91 / 0.54) |
| `three.py` | per-benchmark oracle NoC, failure-rate split, per-cell ablation |
| `errhit_pooled.py` | clicks-inside-error rates (42–55%) |
| `fig3_allbb.py` | the Fig. 3 caption claim, all four backbones |
| `sm_readout.py` | single-mask protocol check |
| `seed_readout.py` | seed sensitivity of the ablation gap |
| `sc_perbench.py` | SimpleClick per-benchmark reproduction against published values |
| `fallback_rate.py` | the Ω_D = ∅ fallback never firing |

Each prints what it checked, and most assert completeness first, so a missing cell fails
loudly instead of silently averaging fewer instances.
