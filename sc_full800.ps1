# Full 800-instance COCO-MVal for the SimpleClick side of Table 2, mirroring full800.ps1.
# results/noc_sc is copied to results/noc_sc_full first so Table 2's current source is
# untouched; resume replays only the 15 previously skipped instances.
# Repository root on the machine the experiments were run on. Set DGC_ROOT to a
# checkout that also holds datasets/, weights/ and results/ before running this.
$DgcRoot = if ($env:DGC_ROOT) { $env:DGC_ROOT } else { '.' }
$ErrorActionPreference = 'Continue'
$py   = "$DgcRoot/py_sc/Scripts/python.exe"
$run  = "$DgcRoot/run_noc_sc.py"
$ck   = "$DgcRoot/weights_sc/cocolvis_vit_base.pth"
$root = "$DgcRoot/datasets"
$man  = "$root/COCO_MVal/manifest.csv"
$src  = "$DgcRoot/results/noc_sc"
$dst  = "$DgcRoot/results/noc_sc_full"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$src/*cocomval*.csv" $dst -Force
Write-Output "copied $((Get-ChildItem $dst -Filter '*cocomval*').Count) csvs"
foreach ($s in 'oracle','entropy','fps','boundary','random') {
  $out = "$dst/noc_sc_vitb_${s}_cocomval.csv"
  if (-not (Test-Path $out)) { Write-Output "MISSING BASE $out"; continue }
  $before = (Import-Csv $out).Count
  Write-Output "=== sc/$s base=$before $(Get-Date -Format 'HH:mm:ss') ==="
  & $py $run --checkpoint $ck --strategy $s --dataset cocomval --manifest $man `
      --data-root $root --out $out --min-area 0
  Write-Output "=== END sc/$s  $before -> $((Import-Csv $out).Count) ==="
}
Write-Output 'SC_FULL800_DONE'
