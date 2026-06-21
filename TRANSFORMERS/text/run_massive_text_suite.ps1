$ErrorActionPreference = "Continue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $Python)) { $Python = "python" }

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True,max_split_size_mb:128"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host " [TEXT TRANSFORMER] MASSIVE EXPERIMENT SUITE INITIATED " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$TextRunner = "TRANSFORMERS\Transformer\Supervised\Runs\run_causal_transformer.py"
$TextAdpModel = "TRANSFORMERS\Transformer\Supervised\Models\model_causal_transformer_adp_width_to_depth.py"

Write-Host "`n>>> Phase 1: Vanilla Ablation (Param Bands 1-10)" -ForegroundColor Yellow
$Depths = @(1, 2, 4, 8, 12)
$Widths = @(64, 128, 256, 512, 1024, 2048)

foreach ($depth in $Depths) {
    foreach ($width in $Widths) {
        $ff = $width * 4
        $nhead = [math]::Max([math]::Floor($width / 64), 1)
        Write-Host "--> Vanilla Ablation: Depth=$depth, Width=$width, FF=$ff, Heads=$nhead"
        & $Python $TextRunner --layers $depth --d_model $width --ff $ff --nhead $nhead --epochs 10 --batch_size 16
    }
}

Write-Host "`n>>> Phase 2: ADP Width-Only Suite (Depths 1 to 5)" -ForegroundColor Yellow
for ($depth = 1; $depth -le 5; $depth++) {
    Write-Host "--> ADP Width-Only Search: Initial Depth=$depth"
    & $Python $TextAdpModel --adp-mode width_only --depth $depth --width 64 --max-epochs 10
}

Write-Host "`n>>> Phase 3: ADP Width-to-Depth (W2D) Suite" -ForegroundColor Yellow
Write-Host "--> Starting dynamic w2d search from minimal seed (Depth=1, Width=64)"
& $Python $TextAdpModel --adp-mode width_to_depth --depth 1 --width 64 --max-epochs 10

Write-Host "============================================================" -ForegroundColor Green
Write-Host " [TEXT TRANSFORMER] MASSIVE EXPERIMENT SUITE COMPLETED " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
