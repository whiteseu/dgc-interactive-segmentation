# Dump the 15 COCO-MVal instances that the <100 px filter had excluded, so the Sec. 3.3
# pixel-level analysis covers the same 800-instance split as the NoC tables.
#
# Both writers skip an npz that already exists, so this adds exactly the missing files and
# leaves the 785 existing dumps untouched. Counts are printed before and after: if a
# backbone does not gain exactly 15 files per tag, the run is wrong and must not be used.
# Repository root on the machine the experiments were run on. Set DGC_ROOT to a
# checkout that also holds datasets/, weights/ and results/ before running this.
$DgcRoot = if ($env:DGC_ROOT) { $env:DGC_ROOT } else { '.' }
$ErrorActionPreference = 'Continue'
$root = "$DgcRoot/datasets"
$man  = "$root/COCO_MVal/manifest.csv"
$py   = "$DgcRoot/py310/python.exe"
$dmp  = "$DgcRoot/dumps"

$bb = @(
  @{d='sam_vit_b';    be='sam';  mt='vit_b'; ck="$DgcRoot/weights/sam_vit_b_01ec64.pth"; cfg=''},
  @{d='sam_vit_l';    be='sam';  mt='vit_l'; ck="$DgcRoot/weights/sam_vit_l_0b3195.pth"; cfg=''},
  @{d='sam_vit_h';    be='sam';  mt='vit_h'; ck="$DgcRoot/weights/sam_vit_h_4b8939.pth"; cfg=''},
  @{d='sam2_hiera_l'; be='sam2'; mt='';      ck="$DgcRoot/weights/sam2.1_hiera_large.pt";
    cfg='configs/sam2.1/sam2.1_hiera_l.yaml'}
)

function Count($p) { if (Test-Path $p) { (Get-ChildItem $p -Filter '*.npz').Count } else { 0 } }

foreach ($b in $bb) {
  $c1 = "$dmp/$($b.d)/cocomval/click1"
  $cl = "$dmp/$($b.d)/cocomval/click1_logits"
  $b1 = Count $c1; $bl = Count $cl
  Write-Output "=== $($b.d)  before: click1=$b1 logits=$bl  $(Get-Date -Format 'HH:mm:ss') ==="

  $common = @('--dataset', 'cocomval', '--manifest', $man, '--data-root', $root,
              '--dump-root', "$dmp/$($b.d)", '--min-area', '0')
  if ($b.be -eq 'sam') {
    & $py "$DgcRoot/runners/run_sam.py" @common --model-type $b.mt --checkpoint $b.ck
  } else {
    & $py "$DgcRoot/runners/run_sam2.py" @common --config $b.cfg --checkpoint $b.ck
  }
  $lg = @('--backend', $b.be, '--checkpoint', $b.ck) + $common
  if ($b.mt)  { $lg += @('--model-type', $b.mt) }
  if ($b.cfg) { $lg += @('--config', $b.cfg) }
  & $py "$DgcRoot/runners/run_logits.py" @lg

  $a1 = Count $c1; $al = Count $cl
  Write-Output "=== END $($b.d)  click1 $b1 -> $a1   logits $bl -> $al  $(Get-Date -Format 'HH:mm:ss') ==="
  if ($a1 -ne 800 -or $al -ne 800) { Write-Output "WARNING $($b.d): expected 800/800, got $a1/$al" }
}
Write-Output 'DUMPS800_ALL_DONE'
