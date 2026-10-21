# Seed sensitivity of the Random-in-Omega_D ablation (the only 0.12-click claim that could
# sit inside sampling noise). DGC itself is deterministic -- its click is the distance
# transform maximum, and the stochastic fallback never fires -- so the seed variance of the
# DGC-vs-ablation gap is entirely the ablation's. Results go to a fresh directory so
# results/noc_v2, the Table 1 source, cannot be touched.
$ROOT = if ($env:DGC_ROOT) { $env:DGC_ROOT } else { '.' }
$ErrorActionPreference = 'Continue'
$py   = "$ROOT/py310/python.exe"
$run  = "$ROOT/runners/run_noc.py"
$ck   = "$ROOT/weights/sam_vit_b_01ec64.pth"
$root = "$ROOT/datasets"
$outd = "$ROOT/results/noc_seed"
New-Item -ItemType Directory -Force $outd | Out-Null

$sets = @(
  @{n='grabcut';   m="$root/GrabCut/manifest.csv"},
  @{n='berkeley';  m="$root/Berkeley/manifest.csv"},
  @{n='davis';     m="$root/DAVIS345/manifest.csv"},
  @{n='cocomval';  m="$root/COCO_MVal/manifest.csv"},
  @{n='camo';      m="$root/TestDataset/CAMO/manifest.csv"},
  @{n='chameleon'; m="$root/TestDataset/CHAMELEON/manifest.csv"}
)
foreach ($s in 1, 2) {
  foreach ($d in $sets) {
    $out = "$outd/noc_vit_b_random_in_d_$($d.n)_s$s.csv"
    Write-Output "=== START $($d.n) seed=$s $(Get-Date -Format 'HH:mm:ss') ==="
    & $py $run --backend sam --model-type vit_b --checkpoint $ck --strategy random_in_d `
        --dataset $d.n --manifest $d.m --data-root $root --out $out --seed $s
    Write-Output "=== END   $($d.n) seed=$s $(Get-Date -Format 'HH:mm:ss') ==="
  }
}
Write-Output 'SEED_BATCH_ALL_DONE'
