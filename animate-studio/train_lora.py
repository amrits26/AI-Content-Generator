"""
═══════════════════════════════════════════════════════════════
AniMate Studio — LoRA Style Training (UNet3D-compatible)
═══════════════════════════════════════════════════════════════
Trains a PEFT LoRA adapter on the text-to-video UNet3D model
using reference images. Optimized for 8GB VRAM GPUs.

Usage:
    python train_lora.py --name "kids_style" \
        --images ./datasets/kids_style_v1 \
        --steps 600 --rank 8

Produces .safetensors LoRA weights in ./loras/<name>/
═══════════════════════════════════════════════════════════════
"""

import argparse
import gc
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from torchvision import transforms

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("animate_studio.train_lora")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_captions(dataset_dir: str) -> dict:
    """Load captions from metadata.jsonl if available."""
    captions = {}
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                captions[entry["file_name"]] = entry["text"]
        logger.info("Loaded %d captions from metadata.jsonl", len(captions))
    return captions


def train_lora(
    character_name: str,
    image_dir: str,
    output_dir: str = "./loras",
    steps: int = 600,
    rank: int = 8,
    learning_rate: float = 1e-4,
    resolution: int = 256,
    batch_size: int = 1,
    gradient_accumulation_steps: int = 4,
    save_every: int = 200,
    config_path: str = "config.yaml",
    resume_from: str = "",
    start_step: int = 0,
):
    """
    Train a LoRA adapter on the UNet3D for style transfer.
    Uses single-frame inputs with the temporal dim squeezed to 1.
    """
    from peft import LoraConfig, get_peft_model
    from diffusers import DDPMScheduler, AutoencoderKL
    from diffusers.models import UNet3DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    from safetensors.torch import save_file

    config = load_config(config_path)
    model_name = config["models"]["video"]["name"]

    # ── Validate dataset ────────────────────────────────
    image_dir = os.path.abspath(image_dir)
    if not os.path.isdir(image_dir):
        logger.error("Image directory not found: %s", image_dir)
        raise SystemExit(1)

    image_files = [
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
    ]
    if not image_files:
        logger.error("No images found in %s", image_dir)
        raise SystemExit(1)

    captions = load_captions(image_dir)
    default_caption = (
        "cute adorable animated characters, bright colors, soft warm lighting, "
        "big expressive eyes, children cartoon style, Pixar quality 3D animation"
    )

    logger.info(
        "Training LoRA '%s' — %d images, %d steps, rank=%d, lr=%.0e, res=%d",
        character_name, len(image_files), steps, rank, learning_rate, resolution,
    )

    # ── Prepare output dir ──────────────────────────────
    safe_name = character_name.replace(" ", "_")
    lora_output_dir = os.path.join(output_dir, safe_name)
    os.makedirs(lora_output_dir, exist_ok=True)

    # ── Load model components separately (memory-efficient) ─
    logger.info("Loading model components from: %s", model_name)

    tokenizer = CLIPTokenizer.from_pretrained(
        model_name, subfolder="tokenizer", local_files_only=True,
    )
    text_encoder = CLIPTextModel.from_pretrained(
        model_name, subfolder="text_encoder",
        torch_dtype=torch.float16, variant="fp16", local_files_only=True,
    )
    vae = AutoencoderKL.from_pretrained(
        model_name, subfolder="vae",
        torch_dtype=torch.float16, variant="fp16", local_files_only=True,
    )
    # Load UNet fp16 (cached) and upcast to fp32 for training
    unet = UNet3DConditionModel.from_pretrained(
        model_name, subfolder="unet",
        torch_dtype=torch.float16, variant="fp16", local_files_only=True,
    )
    unet = unet.to(dtype=torch.float32)
    noise_scheduler = DDPMScheduler.from_pretrained(
        model_name, subfolder="scheduler", local_files_only=True,
    )

    # Freeze text_encoder + VAE (only UNet LoRA trains)
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    unet.requires_grad_(False)

    # ── Apply PEFT LoRA to UNet attention layers ────────
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        target_modules=["to_q", "to_k", "to_v", "to_out.0"],
        lora_dropout=0.05,
    )
    unet = get_peft_model(unet, lora_config)

    # Optionally resume adapter weights from a previous checkpoint directory.
    if resume_from:
        _load_lora_weights_into_peft(unet, resume_from)
        logger.info("Resumed LoRA weights from: %s", resume_from)

    # Enable gradient checkpointing to save VRAM
    # UNet3DConditionModel needs the flag set manually
    UNet3DConditionModel._supports_gradient_checkpointing = True
    unet.base_model.model.enable_gradient_checkpointing()

    # Count trainable params
    trainable = sum(p.numel() for p in unet.parameters() if p.requires_grad)
    total = sum(p.numel() for p in unet.parameters())
    logger.info(
        "LoRA params: %d trainable / %d total (%.2f%%)",
        trainable, total, 100 * trainable / total,
    )

    # ── Move to GPU ─────────────────────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)

    # ── Image transforms ────────────────────────────────
    transform = transforms.Compose([
        transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # ── Optimizer ───────────────────────────────────────
    trainable_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=learning_rate, weight_decay=1e-2,
    )

    # ── Training loop ───────────────────────────────────
    unet.train()
    text_encoder.eval()
    vae.eval()

    if start_step < 0 or start_step >= steps:
        logger.error("Invalid start_step=%d for steps=%d", start_step, steps)
        raise SystemExit(1)

    remaining_steps = steps - start_step
    logger.info(
        "Starting training — target=%d, start=%d, remaining=%d (effective batch = %d x %d = %d)",
        steps, start_step, remaining_steps, batch_size, gradient_accumulation_steps,
        batch_size * gradient_accumulation_steps,
    )

    global_step = start_step
    running_loss = 0.0

    for step in range(remaining_steps * gradient_accumulation_steps):
        # ── Sample a random image ───────────────────────
        img_file = random.choice(image_files)
        try:
            img = Image.open(os.path.join(image_dir, img_file)).convert("RGB")
        except Exception as e:
            logger.warning("Skipping corrupt image %s: %s", img_file, e)
            continue

        pixel_values = transform(img).unsqueeze(0).to(device, dtype=torch.float16)
        caption = captions.get(img_file, default_caption)

        # ── Encode image → latents ──────────────────────
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * vae.config.scaling_factor
            # Add temporal dim for UNet3D: (B, C, H, W) → (B, C, 1, H, W)
            latents = latents.unsqueeze(2).to(dtype=torch.float32)

        # ── Add noise ───────────────────────────────────
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device, dtype=torch.long,
        )
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # ── Text embeddings ─────────────────────────────
        with torch.no_grad():
            text_input = tokenizer(
                caption, padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True, return_tensors="pt",
            )
            encoder_hidden_states = text_encoder(
                text_input.input_ids.to(device)
            )[0].to(dtype=torch.float32)

        # ── UNet prediction ─────────────────────────────
        model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # ── Loss ────────────────────────────────────────
        loss = torch.nn.functional.mse_loss(
            model_pred.float(), noise.float(), reduction="mean",
        )
        loss = loss / gradient_accumulation_steps
        loss.backward()

        running_loss += loss.item()

        # ── Gradient accumulation step ──────────────────
        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % 10 == 0 or global_step == 1:
                avg_loss = running_loss / min(global_step, 10)
                vram_gb = 0.0
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.memory_allocated() / (1024**3)
                logger.info(
                    "Step %d/%d — loss: %.4f — VRAM: %.1f GB",
                    global_step, steps, avg_loss, vram_gb,
                )
                running_loss = 0.0

            # ── Checkpoint save ─────────────────────────
            if save_every > 0 and global_step % save_every == 0:
                ckpt_dir = os.path.join(lora_output_dir, f"checkpoint-{global_step}")
                _save_lora_weights(unet, ckpt_dir)
                logger.info("Checkpoint saved: %s", ckpt_dir)

    # ── Save final weights ──────────────────────────────
    _save_lora_weights(unet, lora_output_dir)
    final_path = os.path.join(lora_output_dir, "pytorch_lora_weights.safetensors")
    logger.info("Training complete! LoRA saved: %s", final_path)

    # ── Update character profile ────────────────────────
    _update_character_profile(
        character_name, final_path, safe_name.lower(), config_path,
    )

    # ── Free GPU ────────────────────────────────────────
    del unet, vae, text_encoder, optimizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return final_path


def _save_lora_weights(unet, output_dir):
    """Save LoRA weights as safetensors in diffusers-compatible format + PEFT adapter_config."""
    from peft.utils import get_peft_model_state_dict
    from safetensors.torch import save_file

    os.makedirs(output_dir, exist_ok=True)

    # Extract PEFT state dict
    peft_state = get_peft_model_state_dict(unet)

    # Save PEFT-format weights (for direct PEFT loading)
    peft_save_path = os.path.join(output_dir, "adapter_model.safetensors")
    peft_state_contiguous = {k: v.float().contiguous() for k, v in peft_state.items()}
    save_file(peft_state_contiguous, peft_save_path)

    # Save adapter config for PEFT loading
    adapter_config = {
        "peft_type": "LORA",
        "r": unet.peft_config["default"].r,
        "lora_alpha": unet.peft_config["default"].lora_alpha,
        "target_modules": list(unet.peft_config["default"].target_modules),
        "lora_dropout": unet.peft_config["default"].lora_dropout,
        "bias": "none",
        "task_type": None,
    }
    config_path = os.path.join(output_dir, "adapter_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(adapter_config, f, indent=2)

    # Also save diffusers-format weights for compatibility
    # PEFT:      base_model.model.<layer>.lora_A.default.weight
    # diffusers: unet.<layer>.lora_A.weight
    diffusers_state = {}
    for key, value in peft_state.items():
        new_key = key.replace("base_model.model.", "unet.")
        new_key = new_key.replace(".default.", ".")
        diffusers_state[new_key] = value.float().contiguous()

    save_path = os.path.join(output_dir, "pytorch_lora_weights.safetensors")
    save_file(diffusers_state, save_path)
    logger.info("Saved %d LoRA tensors → %s", len(diffusers_state), save_path)


def _load_lora_weights_into_peft(unet, resume_dir: str):
    """Load LoRA weights into a PEFT-wrapped UNet from checkpoint dir."""
    from safetensors.torch import load_file

    if not os.path.isdir(resume_dir):
        logger.error("Resume directory not found: %s", resume_dir)
        raise SystemExit(1)

    adapter_path = os.path.join(resume_dir, "adapter_model.safetensors")
    diffusers_path = os.path.join(resume_dir, "pytorch_lora_weights.safetensors")

    if os.path.exists(adapter_path):
        state_dict = load_file(adapter_path)
    elif os.path.exists(diffusers_path):
        state_dict = load_file(diffusers_path)
        # Convert diffusers keys to PEFT keys.
        converted = {}
        for key, value in state_dict.items():
            new_key = key.replace("unet.", "base_model.model.")
            new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
            new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
            converted[new_key] = value
        state_dict = converted
    else:
        logger.error("No resume weights found in: %s", resume_dir)
        raise SystemExit(1)

    missing, unexpected = unet.load_state_dict(state_dict, strict=False)
    if missing:
        logger.info("Resume missing keys: %d", len(missing))
    if unexpected:
        logger.info("Resume unexpected keys: %d", len(unexpected))


def _update_character_profile(
    name: str, lora_path: str, trigger_word: str, config_path: str,
):
    """Register the trained LoRA in the character profiles."""
    config = load_config(config_path)
    profiles_path = os.path.join(
        config["app"]["loras_dir"], "character_profiles.json",
    )

    profiles = {}
    if os.path.exists(profiles_path):
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)

    if name not in profiles:
        profiles[name] = {"name": name}

    profiles[name]["lora_path"] = os.path.abspath(lora_path)
    profiles[name]["lora_trigger_word"] = trigger_word
    profiles[name]["lora_strength"] = 0.7

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    logger.info("Character profile updated: %s", name)


def main():
    parser = argparse.ArgumentParser(
        description="AniMate Studio — LoRA Style Training (UNet3D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_lora.py --name "kids_style" --images ./datasets/kids_style_v1/
  python train_lora.py --name "kids_style" --images ./datasets/kids_style_v1/ --steps 600 --rank 8
        """,
    )
    parser.add_argument("--name", required=True, help="Style/character name")
    parser.add_argument("--images", required=True, help="Image directory")
    parser.add_argument("--output", default="./loras", help="Output dir")
    parser.add_argument("--steps", type=int, default=600, help="Training steps")
    parser.add_argument("--rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--resolution", type=int, default=256, help="Resolution")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Grad accum")
    parser.add_argument("--save-every", type=int, default=200, help="Checkpoint freq")
    parser.add_argument("--config", default="config.yaml", help="Config path")
    parser.add_argument("--resume-from", default="", help="Checkpoint directory to resume from")
    parser.add_argument("--start-step", type=int, default=0, help="Starting global step when resuming")

    args = parser.parse_args()
    train_lora(
        character_name=args.name,
        image_dir=args.images,
        output_dir=args.output,
        steps=args.steps,
        rank=args.rank,
        learning_rate=args.lr,
        resolution=args.resolution,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        save_every=args.save_every,
        config_path=args.config,
        resume_from=args.resume_from,
        start_step=args.start_step,
    )


if __name__ == "__main__":
    main()
