# run_training.ps1 — EPL AI Tactical Analyst training helper
# Run from project root: .\run_training.ps1

param(
    [string]$Mode = "mock",        # "mock" or "real"
    [string]$SoccerNetPassword = "",
    [string]$Model = "all",        # "mlp", "lstm", "transformer", or "all"
    [string]$Subset = "dev",       # "dev" or "full" (real mode only)
    [int]$Window = 20,
    [int]$Stride = 5
)

$ErrorActionPreference = "Stop"
$env:PYTHONIOENCODING = "utf-8"

Write-Host ""
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host "  EPL AI Tactical Analyst — Training Pipeline" -ForegroundColor Cyan
Write-Host "  Mode: $Mode" -ForegroundColor Cyan
Write-Host "====================================================" -ForegroundColor Cyan
Write-Host ""

# ── Step 1: Download / Generate Data ─────────────────────────────────────────
Write-Host "[Step 1] Preparing data..." -ForegroundColor Yellow

if ($Mode -eq "real") {
    if (-not $SoccerNetPassword) {
        Write-Host "  ERROR: Provide -SoccerNetPassword for real mode" -ForegroundColor Red
        Write-Host "  Usage: .\run_training.ps1 -Mode real -SoccerNetPassword YOUR_PASSWORD"
        exit 1
    }
    $env:SOCCERNET_PASSWORD = $SoccerNetPassword
    python scripts/download_soccernet.py --subset $Subset
} else {
    Write-Host "  Using mock data (no SoccerNet account needed)" -ForegroundColor Gray
    python scripts/download_soccernet.py --mock
}

# ── Step 2: Prepare tracking data ────────────────────────────────────────────
Write-Host ""
Write-Host "[Step 2] Preparing tracking data..." -ForegroundColor Yellow
python scripts/prepare_tracking_data.py

# ── Step 3: Build sequences ───────────────────────────────────────────────────
Write-Host ""
Write-Host "[Step 3] Building sequences (window=$Window, stride=$Stride)..." -ForegroundColor Yellow
python scripts/create_sequences.py --window $Window --stride $Stride

# ── Step 4: Inspect dataset ───────────────────────────────────────────────────
Write-Host ""
Write-Host "[Step 4] Inspecting dataset..." -ForegroundColor Yellow
python scripts/inspect_dataset.py

# ── Step 5: Train ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[Step 5] Training..." -ForegroundColor Yellow

if ($Model -eq "all") {
    Write-Host "  Comparing MLP vs LSTM vs Transformer..." -ForegroundColor Cyan
    python training/compare_models.py --config training/configs/transformer_config.yaml
} else {
    $cfgMap = @{
        "mlp"         = "training/configs/mlp_config.yaml"
        "lstm"        = "training/configs/lstm_config.yaml"
        "transformer" = "training/configs/transformer_config.yaml"
    }
    $cfg = $cfgMap[$Model]
    if (-not $cfg) {
        Write-Host "  ERROR: Unknown model '$Model'. Use mlp, lstm, transformer, or all." -ForegroundColor Red
        exit 1
    }
    python training/train.py --model $Model --config $cfg
}

Write-Host ""
Write-Host "====================================================" -ForegroundColor Green
Write-Host "  DONE! Checkpoints saved to models/" -ForegroundColor Green
Write-Host ""
Write-Host "  View TensorBoard:" -ForegroundColor White
Write-Host "    tensorboard --logdir models/" -ForegroundColor Gray
Write-Host ""
Write-Host "  Run unit tests:" -ForegroundColor White
Write-Host "    python -m pytest tests/ -v" -ForegroundColor Gray
Write-Host "====================================================" -ForegroundColor Green
