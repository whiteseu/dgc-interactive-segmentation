$ROOT = if ($env:DGC_ROOT) { $env:DGC_ROOT } else { '.' }
$ErrorActionPreference = 'Continue'
$py   = "$ROOT/py_sc/Scripts/python.exe"
$run  = "$ROOT/run_noc_sc.py"
$ck   = "$ROOT/weights_sc/cocolvis_vit_base.pth"
$root = "$ROOT/datasets"
$outd = "$ROOT/results/noc_sc"
New-Item -ItemType Directory -Force $outd | Out-Null

$sets = @(
  @{n='grabcut';   m="$root/GrabCut/manifest.csv"},
  @{n='berkeley';  m="$root/Berkeley/manifest.csv"},
  @{n='davis';     m="$root/DAVIS345/manifest.csv"},
  @{n='cocomval';  m="$root/COCO_MVal/manifest.csv"},
  @{n='camo';      m="$root/TestDataset/CAMO/manifest.csv"},
  @{n='chameleon'; m="$root/TestDataset/CHAMELEON/manifest.csv"}
)
# oracle first (SimpleClick's own published protocol), then the floor, then label-free
$strats = @('oracle','random','entropy','boundary')

foreach ($s in $strats) {
  foreach ($d in $sets) {
    $out = "$outd/noc_sc_vitb_$($s)_$($d.n).csv"
    Write-Output "=== START $($d.n)/$s -> $out  $(Get-Date -Format 'HH:mm:ss') ==="
    & $py $run --checkpoint $ck --strategy $s --dataset $d.n `
        --manifest $d.m --data-root $root --out $out
    Write-Output "=== END   $($d.n)/$s  $(Get-Date -Format 'HH:mm:ss') ==="
  }
}
Write-Output 'SC_BATCH_ALL_DONE'
