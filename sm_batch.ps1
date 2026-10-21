# Single-mask protocol ablation (review item): SAM ViT-B, six benchmarks, oracle + fps,
# rounds t>=2 decoded with multimask_output=False. Output goes to a fresh directory so
# nothing in results/noc_v2 (the Table 1 source) can be overwritten.
$ROOT = if ($env:DGC_ROOT) { $env:DGC_ROOT } else { '.' }
$ErrorActionPreference = 'Continue'
$py   = "$ROOT/py310/python.exe"
$run  = "$ROOT/runners/run_noc.py"
$ck   = "$ROOT/weights/sam_vit_b_01ec64.pth"
$root = "$ROOT/datasets"
$outd = "$ROOT/results/noc_sm"
New-Item -ItemType Directory -Force $outd | Out-Null

$sets = @(
  @{n='grabcut';   m="$root/GrabCut/manifest.csv"},
  @{n='berkeley';  m="$root/Berkeley/manifest.csv"},
  @{n='davis';     m="$root/DAVIS345/manifest.csv"},
  @{n='cocomval';  m="$root/COCO_MVal/manifest.csv"},
  @{n='camo';      m="$root/TestDataset/CAMO/manifest.csv"},
  @{n='chameleon'; m="$root/TestDataset/CHAMELEON/manifest.csv"}
)
foreach ($s in @('oracle','fps')) {
  foreach ($d in $sets) {
    $out = "$outd/noc_vit_b_$($s)_sm_$($d.n).csv"
    Write-Output "=== START $($d.n)/$s single-mask $(Get-Date -Format 'HH:mm:ss') ==="
    & $py $run --backend sam --model-type vit_b --checkpoint $ck --strategy $s `
        --dataset $d.n --manifest $d.m --data-root $root --out $out --single-mask
    Write-Output "=== END   $($d.n)/$s  $(Get-Date -Format 'HH:mm:ss') ==="
  }
}
Write-Output 'SM_BATCH_ALL_DONE'
