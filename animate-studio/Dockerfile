# AniMate Studio — Production Dockerfile (GPU Optimized)

FROM nvidia/cuda:12.2.0-cudnn8-runtime-ubuntu22.04

# System dependencies (ensure full FFmpeg with H.265/x265 and fonts for subtitles)
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip git \
    ffmpeg libx265-dev libx264-dev libvpx-dev libfreetype6-dev \
    libgl1-mesa-glx libglib2.0-0 fonts-dejavu-core fonts-liberation fonts-freefont-ttf \
    && rm -rf /var/lib/apt/lists/*

# Set up working directory
WORKDIR /app

# Copy requirements
COPY animate-studio/requirements-prod.txt ./requirements-prod.txt
RUN python3.10 -m venv /venv && \
    /venv/bin/pip install --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements-prod.txt

# Copy source
COPY animate-studio /app/animate-studio

# Expose Gradio port
EXPOSE 7860

# Entrypoint
CMD ["/venv/bin/python", "-u", "animate-studio/main.py"]
