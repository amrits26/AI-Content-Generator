# ═══════════════════════════════════════════════════════════
# AniMate Studio - Tokenizer Reset Script
# ═══════════════════════════════════════════════════════════
# Run this if you ever get SentencePiece or tiktoken errors.
# Usage:  .\reset_tokenizer.ps1
# ═══════════════════════════════════════════════════════════

$ErrorActionPreference = "Continue"

$venvPip   = "c:/Users/amrit/OneDrive/Documents/AI Content Generator/.venv/Scripts/pip.exe"
$venvPython = "c:/Users/amrit/OneDrive/Documents/AI Content Generator/.venv/Scripts/python.exe"
$modelCache = "$env:USERPROFILE\.cache\huggingface\hub\models--THUDM--CogVideoX-5b"
$siteDir    = "c:\Users\amrit\OneDrive\Documents\AI Content Generator\.venv\Lib\site-packages"

Write-Host "`n=== AniMate Studio - Tokenizer Reset ===" -ForegroundColor Cyan

# 1. Stop any running Python processes for this project
Write-Host "`n[1/5] Stopping Python processes on port 7860..."
Get-Process python*, python3* -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -like "*AI Content Generator*" } |
    Stop-Process -Force -ErrorAction SilentlyContinue
Get-Process -Id (Get-NetTCPConnection -LocalPort 7860 -ErrorAction SilentlyContinue).OwningProcess `
    -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "  Done." -ForegroundColor Green

# 2. Uninstall sentencepiece completely
Write-Host "`n[2/5] Removing sentencepiece..."
& $venvPip uninstall sentencepiece -y 2>$null
# Remove any broken remnant directories
Get-ChildItem $siteDir -Filter "*entencepiece*" -ErrorAction SilentlyContinue |
    ForEach-Object { Remove-Item $_.FullName -Recurse -Force -ErrorAction SilentlyContinue }
Write-Host "  Done." -ForegroundColor Green

# 3. Upgrade tiktoken
Write-Host "`n[3/5] Upgrading tiktoken..."
& $venvPip install --upgrade "tiktoken>=0.7.0" --quiet
Write-Host "  Done." -ForegroundColor Green

# 4. Clear HuggingFace tokenizer cache for CogVideoX-5b
Write-Host "`n[4/5] Clearing CogVideoX-5b model cache..."
if (Test-Path $modelCache) {
    Remove-Item $modelCache -Recurse -Force
    Write-Host "  Deleted: $modelCache" -ForegroundColor Yellow
    Write-Host "  (Model will re-download on next run - this is expected)" -ForegroundColor DarkGray
} else {
    Write-Host "  Cache not found - already clean." -ForegroundColor Green
}

# 5. Verify
Write-Host "`n[5/5] Verifying environment..."
& $venvPython -c "import tiktoken; print(f'  tiktoken {tiktoken.__version__} OK')"
& $venvPython -c "import sentencepiece" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  sentencepiece correctly absent" -ForegroundColor Green
} else {
    Write-Host "  WARNING: sentencepiece still importable!" -ForegroundColor Red
}

Write-Host "`n=== Tokenizer reset complete ===" -ForegroundColor Cyan
Write-Host "Start the app:  python main.py`n" -ForegroundColor White
