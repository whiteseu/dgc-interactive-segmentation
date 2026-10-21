$ROOT = if ($env:DGC_ROOT) { $env:DGC_ROOT } else { '.' }
$ErrorActionPreference = 'Continue'
$py   = "$ROOT/py_sc/Scripts/python.exe"
$run  = "$ROOT/run_noc_sc.py"
$ck   = "$ROOT/weights_sc/cocolvis_vit_base.pth"
$root = "$ROOT/datasets"
$outd = "$ROOT/results/noc_sc"

$sets = @(
  @{n='grabcut';   m="$root/GrabCut/manifest.csv"},
  @{n='berkeley';  m="$root/Berkeley/manifest.csv"},
  @{n='davis';     m="$root/DAVIS345/manifest.csv"},
  @{n='cocomval';  m="$root/COCO_MVal/manifest.csv"},
  @{n='camo';      m="$root/TestDataset/CAMO/manifest.csv"},
  @{n='chameleon'; m="$root/TestDataset/CHAMELEON/manifest.csv"}
)
foreach ($d in $sets) {
  $out = "$outd/noc_sc_vitb_fps_$($d.n).csv"
  Write-Output "=== START $($d.n)/fps $(Get-Date -Format 'HH:mm:ss') ==="
  & $py $run --checkpoint $ck --strategy fps --dataset $d.n `
      --manifest $d.m --data-root $root --out $out
}
Write-Output 'SC_FPS_DONE'
