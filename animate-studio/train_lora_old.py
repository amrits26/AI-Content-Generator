"""
═══════════════════════════════════════════════════════════════
AniMate Studio — LoRA Training Script
═══════════════════════════════════════════════════════════════
CLI tool for training character-specific LoRA weights from
reference images. Produces .safetensors files for the loras/
directory.

Usage:
    python train_lora.py --name "Billy Bunny" \
        --images ./assets/references/billy/ \
        --steps 1000 --rank 16

Requires: diffusers, peft, accelerate, torch
═══════════════════════════════════════════════════════════════
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("animate_studio.train_lora")


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_lora(
    character_name: str,
    image_dir: str,
    output_dir: str = "./loras",
    steps: int = 1000,
    rank: int = 16,
    learning_rate: float = 1e-4,
    resolution: int = 512,
    batch_size: int = 1,
    config_path: str = "config.yaml",
):
    """
    Train a LoRA adapter for a character using DreamBooth-style fine-tuning.
    """
    try:
        import torch
        from diffusers import StableDiffusionPipeline
        from peft import LoraConfig, get_peft_model
    except ImportError as e:
        logger.error(
            "Missing dependencies for LoRA training. "
            "Run: pip install diffusers peft accelerate torch"
        )
        raise SystemExit(1) from e

    config = load_config(config_path)
    model_name = config["models"]["video"]["name"]

    # Validate image directory
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

    logger.info(
        "Training LoRA for '%s' — %d images, %d steps, rank=%d",
        character_name, len(image_files), steps, rank,
    )

    os.makedirs(output_dir, exist_ok=True)
    safe_name = character_name.replace(" ", "_")
    output_path = os.path.join(output_dir, f"{safe_name}.safetensors")

    # ── Build training command via accelerate ──────────
    # We shell out to diffusers' train_dreambooth_lora.py
    # which handles the training loop properly
    try:
        from diffusers.scripts import train_dreambooth_lora
        _HAS_SCRIPT = True
    except ImportError:
        _HAS_SCRIPT = False

    if _HAS_SCRIPT:
        logger.info("Using diffusers built-in DreamBooth LoRA trainer...")
        # This would call the built-in trainer
    else:
        logger.info(
            "diffusers training script not found. "
            "Using accelerate CLI approach..."
        )

    # Construct the accelerate launch command
    trigger_word = safe_name.lower()
    cmd_args = [
        sys.executable, "-m", "accelerate", "launch",
        "--mixed_precision=fp16",
        "-m", "diffusers.examples.dreambooth.train_dreambooth_lora",
        f"--pretrained_model_name_or_path={model_name}",
        f"--instance_data_dir={image_dir}",
        f"--instance_prompt=a photo of {trigger_word} character",
        f"--output_dir={output_dir}",
        f"--resolution={resolution}",
        f"--train_batch_size={batch_size}",
        f"--max_train_steps={steps}",
        f"--learning_rate={learning_rate}",
        f"--rank={rank}",
        "--gradient_accumulation_steps=1",
        "--lr_scheduler=constant",
        "--lr_warmup_steps=0",
        "--seed=42",
    ]

    logger.info("Launch command:\n  %s", " ".join(cmd_args))

    import subprocess
    result = subprocess.run(cmd_args, capture_output=False)

    if result.returncode != 0:
        logger.error("Training failed with exit code %d", result.returncode)
        logger.info(
            "\n--- MANUAL ALTERNATIVE ---\n"
            "If automated training fails, you can:\n"
            "1. Use Kohya_ss GUI: https://github.com/bmaltais/kohya_ss\n"
            "2. Use CivitAI online trainer\n"
            "3. Place pre-trained .safetensors in ./loras/\n"
        )
        raise SystemExit(result.returncode)

    # Update character profile
    logger.info("Training complete! LoRA saved to: %s", output_path)
    _update_character_profile(character_name, output_path, trigger_word, config_path)


def _update_character_profile(
    name: str, lora_path: str, trigger_word: str, config_path: str
):
    """Register the trained LoRA in the character profiles."""
    config = load_config(config_path)
    profiles_path = os.path.join(config["app"]["loras_dir"], "character_profiles.json")

    profiles = {}
    if os.path.exists(profiles_path):
        with open(profiles_path, "r", encoding="utf-8") as f:
            profiles = json.load(f)

    if name not in profiles:
        profiles[name] = {"name": name}

    profiles[name]["lora_path"] = os.path.abspath(lora_path)
    profiles[name]["lora_trigger_word"] = trigger_word

    with open(profiles_path, "w", encoding="utf-8") as f:
        json.dump(profiles, f, indent=2)
    logger.info("Character profile updated: %s", name)


def main():
    parser = argparse.ArgumentParser(
        description="AniMate Studio — LoRA Training CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_lora.py --name "Billy Bunny" --images ./assets/references/billy/
  python train_lora.py --name "Baby Pup" --images ./refs/ --steps 2000 --rank 32
        """,
    )
    parser.add_argument("--name", required=True, help="Character name")
    parser.add_argument("--images", required=True, help="Directory of reference images")
    parser.add_argument("--output", default="./loras", help="Output directory (default: ./loras)")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps (default: 1000)")
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank (default: 16)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate (default: 1e-4)")
    parser.add_argument("--resolution", type=int, default=512, help="Training resolution (default: 512)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--config", default="config.yaml", help="Config file path")

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
        config_path=args.config,
    )


if __name__ == "__main__":
    main()
