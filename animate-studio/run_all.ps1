# =========================
# run_all.ps1
# =========================
Write-Host "=== Animate Studio: One-Click Launcher ===" -ForegroundColor Cyan

# 1. Setup API in new window
Start-Process powershell -ArgumentList "-NoExit", "-Command", ".\setup_api.ps1"

# 2. Download datasets
Write-Host "Downloading datasets..." -ForegroundColor Green
powershell -Command ".\download_datasets.ps1"

# 3. Prepare training data
Write-Host "Preparing training data..." -ForegroundColor Green
python prepare_training_data.py

# 4. Print training command
Write-Host "`n=== TRAINING COMMAND ===" -ForegroundColor Yellow
Write-Host @"
python train_lora.py `
  --pretrained_model_name_or_path="emilianJR/epiCRealism" `
  --dataset_dir="./datasets/cute_combined" `
  --output_dir="./loras/cute_baby_beast" `
  --rank=128 `
  --lora_alpha=128 `
  --max_train_steps=6000 `
  --snr_gamma=5.0 `
  --noise_offset=0.1 `
  --validation_prompt="A tiny baby bunny with big sparkly eyes and a pastel blue bow, sitting in a field of daisies, soft bokeh, storybook illustration, ultra cute" `
  --mixed_precision="bf16"
"@
Write-Host "Copy and run the above command to start training. Adjust rank/alpha for lower VRAM if needed." -ForegroundColor Cyan
