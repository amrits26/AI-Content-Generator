# Deploy AniMate Studio and ClipStreams APIs to RunPod GPU instance
# Usage: .\deploy_to_runpod.ps1

docker-compose build
docker-compose up -d
Write-Host "Deployment complete! Visit your RunPod public endpoint."
