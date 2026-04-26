#!/bin/bash
# AniMate Studio One-Click Cloud Deploy (Bash)
# Usage: ./deploy_studio.sh

set -e

CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo -e "${CYAN}[AniMate Studio] Starting One-Click Cloud Deployment...${NC}"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Docker is not installed. Please install Docker.${NC}"
    exit 1
fi

# Check Dockerfile
if [ ! -f Dockerfile ]; then
    echo -e "${RED}Dockerfile not found in project root.${NC}"
    exit 1
fi

# Check .env
if [ ! -f .env ]; then
    echo -e "${RED}.env file not found. Please create and configure .env.${NC}"
    exit 1
fi

# Build Docker image
echo -e "${YELLOW}Building Docker image...${NC}"
imgName="animatestudio:latest"
docker build -t $imgName .

# Run Docker container
echo -e "${YELLOW}Running Docker container (GPU enabled)...${NC}"
docker run --gpus all -d --env-file .env -p 7860:7860 --name animatestudio $imgName

echo -e "${GREEN}[SUCCESS] AniMate Studio deployed! Access at http://localhost:7860${NC}"
