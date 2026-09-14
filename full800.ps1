# Evaluate the full 800-instance COCO-MVal split instead of the filtered 785.
#
# results/noc_v2 is copied to results/noc_full first and the runs are pointed at the copies,
# so the Table 1 source stays untouched and run_noc.py's resume logic replays only the 15
# instances that were previously skipped. --min-area 0 disables the <100 px filter.
# Repository root on the machine the experiments were run on. Set DGC_ROOT to a
# checkout that also holds datasets/, weights/ and results/ before running this.
$DgcRoot = if ($env:DGC_ROOT) { $env:DGC_ROOT } else { '.' }
$ErrorActionPreference = 'Continue'
$root = "$DgcRoot/datasets"
$man  = "$root/COCO_MVal/manifest.csv"
$py   = "$DgcRoot/py310/python.exe"
$run  = "$DgcRoot/runners/run_noc.py"
$src  = "$DgcRoot/results/noc_v2"
$dst  = "$DgcRoot/results/noc_full"
New-Item -ItemType Directory -Force $dst | Out-Null
Copy-Item "$src/*cocomval*.csv" $dst -Force
Write-Output "copied $((Get-ChildItem $dst -Filter '*cocomval*').Count) cocomval csvs"

$bb = @(
  @{n='vit_b'; be='sam';  mt='vit_b'; ck="$DgcRoot/weights/sam_vit_b_01ec64.pth"; cfg=''},
  @{n='vit_l'; be='sam';  mt='vit_l'; ck="$DgcRoot/weights/sam_vit_l_0b3195.pth"; cfg=''},
  @{n='vit_h'; be='sam';  mt='vit_h'; ck="$DgcRoot/weights/sam_vit_h_4b8939.pth"; cfg=''},
  @{n='sam2';  be='sam2'; mt='';      ck="$DgcRoot/weights/sam2.1_hiera_large.pt";
    cfg='configs/sam2.1/sam2.1_hiera_l.yaml'}
)
# name = file stem after the backbone, strategy = --strategy value, extra = extra flags
$pol = @(
  @{f='risk_dt';        s='risk_dt';     x=@()},
  @{f='risk';           s='risk';        x=@()},
  @{f='random_in_d';    s='random_in_d'; x=@()},
  @{f='fps';            s='fps';         x=@()},
  @{f='random';         s='random';      x=@()},
  @{f='entropy';        s='entropy';     x=@()},
  @{f='boundary';       s='boundary';    x=@()},
  @{f='oracle';         s='oracle';      x=@()},
  @{f='bald';           s='bald';        x=@()},
  @{f='bald_rho20';     s='bald';        x=@('--bald-rho','0.2')}
)

foreach ($b in $bb) {
  foreach ($p in $pol) {
    if ($p.f -eq 'bald_rho20') { $out = "$dst/noc_$($b.n)_bald_cocomval_rho20.csv" }
    else                       { $out = "$dst/noc_$($b.n)_$($p.f)_cocomval.csv" }
    if (-not (Test-Path $out)) { Write-Output "MISSING BASE $out"; continue }
    $before = (Import-Csv $out).Count
    Write-Output "=== $($b.n)/$($p.f)  base=$before  $(Get-Date -Format 'HH:mm:ss') ==="
    $a = @('--backend', $b.be, '--checkpoint', $b.ck, '--strategy', $p.s,
           '--dataset', 'cocomval', '--manifest', $man, '--data-root', $root,
           '--out', $out, '--min-area', '0')
    if ($b.mt)  { $a += @('--model-type', $b.mt) }
    if ($b.cfg) { $a += @('--config', $b.cfg) }
    $a += $p.x
    & $py $run @a
    $after = (Import-Csv $out).Count
    Write-Output "=== END $($b.n)/$($p.f)  $before -> $after  $(Get-Date -Format 'HH:mm:ss') ==="
  }
}
Write-Output 'FULL800_ALL_DONE'
