# AniMate Studio One-Click Cloud Deploy (PowerShell)
# Usage: ./deploy_studio.ps1

Write-Host "[AniMate Studio] Starting One-Click Cloud Deployment..." -ForegroundColor Cyan

# Check Docker
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Host "Docker is not installed. Please install Docker Desktop." -ForegroundColor Red
    exit 1
}

# Check Dockerfile
if (-not (Test-Path "Dockerfile")) {
    Write-Host "Dockerfile not found in project root." -ForegroundColor Red
    exit 1
}

# Check .env
if (-not (Test-Path ".env")) {
    Write-Host ".env file not found. Please create and configure .env." -ForegroundColor Red
    exit 1
}

# Build Docker image
Write-Host "Building Docker image..." -ForegroundColor Yellow
$imgName = "animatestudio:latest"
docker build -t $imgName .
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker build failed." -ForegroundColor Red
    exit 1
}

# Run Docker container
Write-Host "Running Docker container (GPU enabled)..." -ForegroundColor Yellow
docker run --gpus all -d --env-file .env -p 7860:7860 --name animatestudio $imgName
if ($LASTEXITCODE -ne 0) {
    Write-Host "Docker run failed." -ForegroundColor Red
    exit 1
}

Write-Host "[SUCCESS] AniMate Studio deployed! Access at http://localhost:7860" -ForegroundColor Green
