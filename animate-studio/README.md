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

- Gradio UI with four tabs:
  - Quick Story for short hooks
  - Full Episode for longer multi-scene videos
  - Character Lab for managing character profiles
  - Usage & Costs for tracking API usage and estimated spend
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
- `models.safety`: CLIP/NSFK models, blocked concepts, threshold, and frame scan rate
- `audio`: TTS provider, voices, and BGM levels
- `export`: platform presets, upscale settings, and color grading
- `performance`: cache and VRAM throttling controls
- `compliance`: export-time safety and disclosure checks
- `usage`: usage tracking database path and cost rate overrides

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

If you use OpenAI instead, set the provider in `config.yaml` and supply `OPENAI_API_KEY` as an environment variable. You can also set `OPENAI_BASE_URL` to switch between local and hosted endpoints without editing the repo config.

### Audio options

Narration supports:

- ElevenLabs when `ELEVENLABS_API_KEY` is set
- Edge TTS as the fallback path

Prefer environment variables for secrets:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:ELEVENLABS_API_KEY="your-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
```

`config.yaml` should remain free of secrets. Keep provider credentials in environment variables or another untracked local secret store.

## How To Use The UI

### Quick Story

Use this tab to generate a short hook, typically 15 to 60 seconds.

Typical flow:

1. Enter a theme.
2. Choose duration, character, style, and quality preset.
3. Generate the video.
4. Add narration and export the final result.

### Full Episode

Use this tab for longer multi-scene videos with a stronger story arc and export-ready output.

### Character Lab

Use this tab to:

- Save named character profiles
- Define animal type, color, accessory, and traits
- Attach reference images
- Associate matching LoRA assets

Character profiles are stored in `loras/character_profiles.json`.

### Usage & Costs

This tab shows aggregate usage across LLM calls, TTS calls, and video generation. It tracks token counts, character counts, video duration, and estimated costs. Usage data is stored in `output/usage.db` (SQLite). Cost rates can be customized in `config.yaml`:

```yaml
usage:
  cost_rates:
    llm_per_1k_tokens_cents: 2.0
    tts_per_1k_chars_cents: 30.0
    video_per_minute_cents: 0.0
```

### Safety Frame Scanning

The safety filter scans generated frames for blocked concepts. The scan rate is configurable:

```yaml
models:
  safety:
    frame_sample_rate: 1        # 1 = scan every frame (default, safest)
    frame_sample_rate_fast: 4   # used when fast_mode is enabled
```

### Input Sanitization

All user-supplied prompts (story themes, concepts, and edited storyboard fields) are checked through the ML-based safety text scanner before being passed to the LLM or video pipeline. Unsafe prompts are rejected with a clear message in the UI.

## Batch Generation

Use `batch_generate.py` to generate videos from `themes.csv` and resume after interruptions.

Examples:

```powershell
.\.venv\Scripts\python.exe batch_generate.py
.\.venv\Scripts\python.exe batch_generate.py --csv themes.csv --resume
.\.venv\Scripts\python.exe batch_generate.py --cooldown 30
```

The batch flow:

- loads themes from CSV
- generates a storyboard
- renders scenes
- adds narration if available
- exports a TikTok-formatted result
- writes progress to `batch_state.json`

## LoRA Training

Use `train_lora.py` to train a style or character LoRA from images in a dataset directory.

Example:

```powershell
.\.venv\Scripts\python.exe train_lora.py --name "kids_style" --images .\datasets\kids_style_v1 --steps 600 --rank 8
```

Expected output:

- weights under `loras/<name>/`
- checkpoints during training
- adapters discoverable by the character manager

The training dataset can include a `metadata.jsonl` file for captions.

## Helper Scripts

- `run_e2e_test.py`: full scripted path from story to narrated video
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