# =========================
# setup_api.ps1
# =========================
<#
STEP-BY-STEP STRIPE SETUP (Read Carefully):

1. Go to https://dashboard.stripe.com/products and create THREE one-time products:
   - "10 Credits" (price: your choice, one-time)
   - "50 Credits" (price: your choice, one-time)
   - "100 Credits" (price: your choice, one-time)

2. For each product, click "Add price" and note the Price ID (e.g., price_1Nxxxx...).

3. Go to https://dashboard.stripe.com/apikeys and copy your "Secret Key" (starts with sk_live_...).

4. Open the generated .env file in the api/ folder and REPLACE the placeholder values:
   STRIPE_SECRET_KEY=sk_live_...
   STRIPE_PRICE_ID_10=price_...
   STRIPE_PRICE_ID_50=price_...
   STRIPE_PRICE_ID_100=price_...

5. Save the .env file.

6. To test the API after it starts, run:
   Invoke-RestMethod -Uri http://localhost:8000/v1/create-checkout-session -Method Post -ContentType "application/json" -Body '{"user_id":"test123","credits":10}'

#>

Write-Host "=== Animate Studio API Setup ===" -ForegroundColor Cyan

# --- Check if API is already running ---
$apiPort = 8000
try {
    $apiCheck = Invoke-WebRequest -Uri "http://localhost:$apiPort/docs" -UseBasicParsing -TimeoutSec 2
    if ($apiCheck.StatusCode -eq 200) {
        Write-Host "API server already running on port $apiPort. Exiting." -ForegroundColor Yellow
        exit 0
    }
} catch {}

# --- Ensure api/ folder exists ---
$apiDir = Join-Path $PSScriptRoot "api"
if (-not (Test-Path $apiDir)) {
    Write-Host "ERROR: api/ folder not found. Please ensure your FastAPI code is in 'api/'." -ForegroundColor Red
    exit 1
}

# --- Install dependencies using system Python ---
$requirements = @(
    "fastapi",
    "uvicorn[standard]",
    "stripe",
    "python-dotenv",
    "pydantic",
    "aiofiles"
)
Write-Host "Installing API dependencies with system Python..." -ForegroundColor Green
$installCmd = "python -m pip install " + ($requirements -join " ")
$installResult = Invoke-Expression $installCmd
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install dependencies. Check your Python installation." -ForegroundColor Red
    exit 1
}

# --- Generate .env template if missing ---
$envPath = Join-Path $apiDir ".env"
if (-not (Test-Path $envPath)) {
    @"
# Fill in your actual Stripe keys and price IDs below!
STRIPE_SECRET_KEY=sk_live_your_secret_key_here
STRIPE_PRICE_ID_10=price_10_credits_here
STRIPE_PRICE_ID_50=price_50_credits_here
STRIPE_PRICE_ID_100=price_100_credits_here
"@ | Set-Content -Encoding UTF8 $envPath
    Write-Host "Generated api/.env template. Please update it with your real Stripe keys and price IDs." -ForegroundColor Yellow
} else {
    Write-Host "api/.env already exists. Please verify your Stripe keys and price IDs." -ForegroundColor Yellow
}

# --- Launch API server ---
Write-Host "Starting FastAPI server on http://localhost:$apiPort ..." -ForegroundColor Green
Push-Location $PSScriptRoot
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m uvicorn api.main:app --reload --port $apiPort"
Pop-Location
Write-Host "API server launched. Use Ctrl+C in the server window to stop." -ForegroundColor Cyan
