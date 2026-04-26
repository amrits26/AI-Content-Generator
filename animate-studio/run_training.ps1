# ============================================
# AniMate Studio LoRA Training Launcher
# ============================================

$venv = "..\.venv"
$python = "$venv\Scripts\python.exe"

if (!(Test-Path $python)) {
    Write-Host "❌ Python venv not found at $venv. Please create it first." -ForegroundColor Red
    exit 1
}

Write-Host "📦 Ensuring required packages are installed..." -ForegroundColor Cyan
& $python -m pip install --upgrade pip
& $python -m pip install accelerate diffusers peft transformers safetensors bitsandbytes wandb

Write-Host "🚀 Starting training..." -ForegroundColor Green
& $python train_lora.py --config train_config.yaml
