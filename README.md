# AniMate Studio

AniMate Studio is a Windows-friendly Python app for generating short animated videos from a story prompt. It combines story generation, character management, text-to-video rendering, narration, safety checks, and platform-specific export in a Gradio UI.

The project is geared toward short-form kids animation workflows such as YouTube Kids, Facebook Reels, and TikTok. It also includes batch generation, LoRA training utilities, and script-based pipeline tests.

## What It Does

- Generate a storyboard from a theme using a local or OpenAI-compatible LLM.
- Render scene-based animation with Diffusers and optional AnimateDiff motion support.
- Keep character details consistent with saved profiles and LoRA adapters.
- Create narration with ElevenLabs or Edge TTS.
- Run safety and compliance checks before export.
- Export video variants for YouTube, Facebook Reels, and TikTok.
- Run unattended batch generation from a CSV of themes.

## Main Features

- Gradio UI with three tabs:
  - Quick Story for short hooks
  - Full Episode for longer multi-scene videos
  - Character Lab for managing character profiles
- Config-driven pipeline via `config.yaml`
- Scene caching and VRAM-aware generation settings
- FFmpeg-based audio mixing and export pipeline
- Optional LoRA training for custom styles or characters
- Test and smoke scripts for validating the pipeline outside the UI

## Project Layout

```text
animate-studio/
  main.py                  # Gradio application entry point
  config.yaml              # Central configuration
  batch_generate.py        # CSV-driven batch generation
  train_lora.py            # LoRA training utility
  run_e2e_test.py          # End-to-end script test
  test_pipeline.py         # Integration test script
  engine/
    story_engine.py        # Storyboard generation
    animator.py            # Video generation pipeline
    audio_engine.py        # Narration and audio mixing
    exporter.py            # Platform-specific export
    character_manager.py   # Character profiles and LoRA discovery
    safety_filter.py       # Safety and compliance checks
  utils/
    ffmpeg_utils.py        # FFmpeg helpers
    prompt_templates.py    # Prompt construction
    upscaler.py            # Upscaling helpers
  assets/
    audio/                 # Optional background music
    references/            # Character reference images
  datasets/                # Training datasets
  loras/                   # Character/style LoRA assets
  output/                  # Generated media
  tests/                   # Unit tests
```

## Requirements

- Windows with PowerShell
- Python 3.11+
- FFmpeg in `PATH`
- NVIDIA GPU strongly recommended
- CUDA-capable PyTorch build for practical generation speed

The current dependency set is defined in `requirements.txt`. The video stack is heavy; allow extra disk space for model weights, caches, and generated assets.

## Quick Start

### 1. Create and activate a virtual environment

```powershell
cd "c:\Users\amrit\OneDrive\Documents\AI-Content-Generator\animate-studio"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install PyTorch and project dependencies

If you want CUDA 12.1 wheels:

```powershell
.\.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 3. Verify FFmpeg and CUDA

```powershell
ffmpeg -version
.\.venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available())"
```

### 4. Launch the app

```powershell
.\.venv\Scripts\python.exe main.py
```

Then open `http://localhost:7860` in your browser.

## Configuration

Most behavior is controlled through `config.yaml`.

Key sections:

- `app`: output, assets, logs, and directory paths
- `models.story`: LLM provider and model settings
- `models.video`: video model, dtype, dimensions, steps, fps, and device
- `motion`: AnimateDiff motion settings
- `audio`: TTS provider, voices, and BGM levels
- `export`: platform presets, upscale settings, and color grading
- `performance`: cache and VRAM throttling controls
- `compliance`: export-time safety and disclosure checks

### Story model options

The story engine supports OpenAI-compatible endpoints, including local setups.

Default local configuration:

```yaml
models:
  story:
    provider: "local"
    model_name: "llama3.2"
    openai_base_url: "http://localhost:11434/v1"
```

If you use Ollama, make sure the configured model is available locally:

```powershell
ollama pull llama3.2
```

If you use OpenAI instead, set the provider and API settings in `config.yaml`, or preferably supply `OPENAI_API_KEY` as an environment variable.

### Audio options

Narration supports:

- ElevenLabs when an API key is configured

# AniMate Studio

## Cinematic, Agentic Animation Suite — Production-Grade

---

### 🚀 Overview
AniMate Studio is a luxury-grade, agentic animation engine for kids' stories, built for cinematic quality, reliability, and creative control. It features a manifest-driven, multi-agent workflow, 4K upscaling, robust fallback logic, and a Cupertino-inspired UI.

---

## 🏗️ Architecture

![Agentic Multi-Agent Flow](animate-studio/assets/references/architecture-diagram.png)

- **Manifest-Driven Workflow**: LLM-powered storyboard → manifest → animation → audio → export.
- **Multi-Agent Engine**: Modular agents for story, animation, audio, export, safety, and usage tracking.
- **Robust Fallbacks**: All API calls use tenacity retry and circuit breaker logic.
- **Luxury UI**: Gradio with Cupertino theme, glassmorphism, and micro-interactions.
- **AIOps Monitoring**: Latency tracking, predictive logging, and cost estimation.
- **Cold Storage**: Raw scene clips are archived after 4K export for a minimalist workspace.

---

## ⚡ Quickstart

```sh
# 1. Clone and enter the repo
$ git clone <your-repo-url>
$ cd AI-Content-Generator/animate-studio

# 2. (Recommended) Set up Python 3.10+ venv
$ python3 -m venv .venv
$ source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install production dependencies
$ pip install -r requirements-prod.txt

# 4. Launch the app
$ python main.py
```

---

## 🛠️ VS Code Setup & Env Vars
- Recommended: VS Code + Python extension
- Set environment variables in `.env` or via VS Code launch config:
  - `OPENAI_API_KEY`, `ELEVENLABS_API_KEY`, etc.
- GPU: NVIDIA CUDA 12+ required for 4K upscaling

---

## 💸 Cost Estimation
- **LLM**: $0.02 per 1K tokens
- **TTS**: $0.30 per 1K chars
- **Video**: $0 (local GPU)
- **Total**: See Usage & Costs tab in UI for live breakdown

---

## 🧩 Key Modules
- `engine/story_engine.py`: Manifest/LLM agent
- `engine/animator.py`: Animation agent (motion mastery, LoRA, temporal consistency)
- `engine/audio_engine.py`: TTS, BGM, robust fallback
- `engine/exporter.py`: 4K upscaling, cold storage, strict naming
- `engine/usage_tracker.py`: SQLite logging, latency, AIOps
- `main.py`: Gradio UI, Cupertino theme, Review Station

---

## 📝 Documentation
- See `setup_guide.md` for full setup and training LoRA/character adapters.
- API docs: See `animate-studio/api_docs.json` (Swagger/OpenAPI for FastAPI backend).

---

## 🏆 Credits
- Built by AniMate Studio contributors, 2026.
- Powered by OpenAI, ElevenLabs, ModelScope, Real-ESRGAN, and more.
- `test_pipeline.py`: integration-style checks for FFmpeg, story generation, audio, pipeline load, and one-scene generation
- `download_model.py`: helper for downloading CogVideoX model assets
- `generate_captions.py`: caption-related helper workflow
- `smoke_test_animatediff.py` and `debug_*.py`: targeted troubleshooting scripts

## Testing And Validation

Script-based validation:

```powershell
.\.venv\Scripts\python.exe test_pipeline.py
.\.venv\Scripts\python.exe run_e2e_test.py
```

If you have `pytest` installed in the environment, you can also run the unit tests in `tests/`.

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

## Outputs

Generated files are written under `output/`. Depending on the workflow, you may see:

- intermediate scene videos
- narration WAV files
- exported MP4 files
- thumbnails
- SRT caption files
- JSON metadata for platform uploads

Logs are written to:

- `animate_studio.log`
- `safety_audit.log`
- `batch_log.txt`

## Troubleshooting

### FFmpeg not found

Install FFmpeg and confirm that `ffmpeg -version` works in PowerShell.

### Story generation fails

Check:

- `models.story` settings in `config.yaml`
- whether Ollama is running for local mode
- whether the configured model exists locally
- API keys if using a hosted provider

### CUDA or VRAM issues

Reduce these values in `config.yaml`:

- `models.video.num_inference_steps`
- `models.video.height`
- `models.video.width`
- `quality_presets.*`

You can also keep scene caching enabled and use a lower quality preset for iteration.

### Audio generation fails

The app can fall back from ElevenLabs to Edge TTS. If narration still fails, verify that:

- your TTS provider settings are correct
- FFmpeg is installed
- the output directory is writable

## Notes

- The first successful run may download large model assets.
- Local generation is significantly slower without CUDA.
- Export compliance checks can block output if safety requirements fail.
- This repository is optimized around a local, scriptable pipeline rather than a hosted SaaS workflow.

## Recommended First Run

1. Configure `config.yaml` for your story provider.
2. Verify `ffmpeg`, CUDA, and any required API keys.
3. Start the UI with `main.py`.
4. Create or select a character in Character Lab.
5. Generate a short Quick Story before attempting longer exports.
