# One-click launcher for AniMate Studio API
# Usage: .\run_animate_studio_api.ps1

$env:PYTHONPATH = "."
$env:STRIPE_SECRET_KEY = "<your_stripe_secret>"
$env:STRIPE_WEBHOOK_SECRET = "<your_stripe_webhook_secret>"
$env:API_KEY = "<your_api_key>"
python -m uvicorn api.animate_api:app --host 0.0.0.0 --port 8001
