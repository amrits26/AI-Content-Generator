# AniMate Studio — Setup & Quick-Start Guide

## Prerequisites

| Requirement | Version | Check Command |
|---|---|---|
| Python | 3.11+ | `python --version` |
| CUDA | 12.1+ | `nvidia-smi` |
| FFmpeg | 6.0+ | `ffmpeg -version` |
| GPU VRAM | 8 GB+ (16 GB recommended) | `nvidia-smi` |
| Disk Space | ~30 GB (models + cache) | |

---

## 1. Environment Setup

```powershell
# Navigate to project
cd "C:\Users\amrit\OneDrive\Documents\AI Content Generator\animate-studio"

# Create virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install all dependencies
pip install -r requirements.txt
```

### Verify CUDA
```powershell
.\.venv\Scripts\python.exe -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0)}')"
```

---

## 2. FFmpeg Installation

If you don't have FFmpeg installed:

```powershell
# Option A: winget (recommended)
winget install Gyan.FFmpeg

# Option B: choco
choco install ffmpeg

# Verify
ffmpeg -version
```

---

## 3. API Keys Configuration

Edit `config.yaml`:

### ElevenLabs (Narration — Optional but Recommended)
1. Sign up at [elevenlabs.io](https://elevenlabs.io)
2. Go to Profile → API Keys
3. Copy your key into `config.yaml`:
```yaml
audio:
  tts:
    elevenlabs_api_key: "your-key-here"
```

> **Free tier:** 10,000 characters/month. Enough for ~10 episodes.  
> **Without ElevenLabs:** System auto-falls back to Edge TTS (free, lower quality).

### Story LLM (Required — Choose One)

**Option A: Ollama (Free, Local)**
```powershell
# Install Ollama from ollama.com, then:
ollama pull llama3.2
```
Config stays default:
```yaml
models:
  story:
    provider: "ollama"
    model_name: "llama3.2"
    openai_base_url: "http://localhost:11434/v1"
```

**Option B: OpenAI API**
```yaml
models:
  story:
    provider: "openai"
    model_name: "gpt-4o-mini"
    openai_base_url: "https://api.openai.com/v1"
    openai_api_key: "sk-your-key-here"
```

---

## 4. Add Background Music (Optional)

Place royalty-free `.mp3` or `.wav` files in:
```
animate-studio/assets/audio/
```

Recommended free sources:
- [Pixabay Music](https://pixabay.com/music/) — CC0
- [Uppbeat](https://uppbeat.io/) — Free tier with attribution
- [YouTube Audio Library](https://studio.youtube.com/channel/UC/music)

---

## 5. Launch AniMate Studio

```powershell
cd "C:\Users\amrit\OneDrive\Documents\AI Content Generator\animate-studio"
.\.venv\Scripts\Activate.ps1
.\.venv\Scripts\python.exe main.py
```

Open in browser: **http://localhost:7860**

---

## 6. Generate Your First Episode — "Billy Bunny Learns to Share"

### Step 1: Create the Character (Tab 3 — Character Lab)
1. **Name:** Billy
2. **Animal Type:** bunny
3. **Color:** soft blue
4. **Accessory:** red bowtie
5. **Traits:** friendly, curious, kind
6. Click **💾 Save Character**

### Step 2: Quick Test (Tab 1 — Quick Story)
1. **Theme:** "Billy Bunny learns to share his carrot with a lonely duckling"
2. **Duration:** 30s
3. **Character:** Billy
4. Click **🎬 Generate Hook**
5. Wait ~15 minutes for 3 scenes
6. Click **📱 Export for Reels** — instant Facebook-ready export

### Step 3: Full Episode (Tab 2 — Full Episode)
1. **Story Concept:** "Billy Bunny finds a lonely duckling in the meadow. He shares his carrot and they become best friends. They explore the magical garden together."
2. **Scenes:** 5
3. **Voice:** Rachel (Warm)
4. **Character:** Billy
5. Click **🎬 Generate Full Episode**
6. Expected time: ~45-60 minutes (RTX 4060/4070)

---

## 7. LoRA Training (Custom Characters)

For ultimate character consistency, train a LoRA:

### Prerequisites
- [Kohya SS GUI](https://github.com/bmaltais/kohya_ss) installed
- 10-20 reference images of your character (front, side, various poses)

### Quick Steps
1. Prepare images: 512x512 PNG, character centered, white background
2. Place in a training folder with captions (`.txt` files)
3. Open Kohya SS → LoRA Training
4. Settings:
   - **Learning rate:** 1e-4
   - **Epochs:** 20-40
   - **Batch size:** 1 (for 8GB VRAM)
   - **Resolution:** 512x512
5. Save the `.safetensors` file to `./loras/[CharacterName].safetensors`
6. In Character Lab, the LoRA auto-detects when names match

---

## 8. Performance Tips (HP Omen)

| Setting | Value | Why |
|---|---|---|
| `torch.bfloat16` | Enabled | 2x memory savings on Ampere+ |
| `enable_model_cpu_offload` | Enabled | Swaps unused model parts to RAM |
| `enable_vae_tiling` | Enabled | Processes VAE in tiles, saves VRAM |
| GPU temp limit | 82°C | Auto-pause prevents thermal throttling |
| Scene caching | Enabled | Skip regenerating unchanged scenes |

### Expected Generation Times (RTX 4060 8GB)
| Content | Duration | Generation Time |
|---|---|---|
| Quick Hook (3 scenes) | ~30s video | ~15 min |
| Full Episode (5 scenes) | ~60s video | ~45 min |
| Full Episode (8 scenes) | ~90s video | ~75 min |

---

## 9. Monetization Checklist

Before uploading content, verify:

- [ ] Safety filter passed (check `safety_audit.log`)
- [ ] Human creative input logged (story editing, character selection)
- [ ] AI disclosure in video description (auto-generated in metadata)
- [ ] Background music is royalty-free
- [ ] Video meets platform duration requirements
- [ ] Thumbnail is kid-safe (auto-scanned)

### Platform-Specific Notes

**YouTube Kids:**
- Upload 60-90s versions (16:9) — qualifies for mid-roll ads
- Add to a kids' playlist for algorithmic boost
- Include `.srt` captions for accessibility

**Facebook Reels:**
- Upload 9:16 vertical exports
- Post 15+ Reels in 10 days for Creator Fast Track eligibility
- AI content is explicitly allowed for Reels monetization

**TikTok:**
- Upload 9:16 vertical exports
- Label as "AI-generated" in settings
- Need 10K+ followers for Creator Rewards

---

## Troubleshooting

| Issue | Solution |
|---|---|
| CUDA out of memory | Reduce `num_inference_steps` to 30 in config.yaml |
| FFmpeg not found | Ensure FFmpeg is in PATH: `$env:PATH` |
| Models download slow | First run downloads ~15GB. Use metered WiFi workaround: pre-download from HuggingFace |
| Safety filter too strict | Lower `safety_threshold` in config.yaml (default 0.25) |
| Edge TTS fails | `pip install edge-tts --upgrade` |
| GPU overheating | Lower `gpu_temp_limit_c` to 78 in config.yaml |
