"""
AnimateDiff Engine — Real motion video generation using SD1.5 + motion modules.
Replaces the static-image-with-interpolation approach with true temporal generation.
"""

import gc
import logging
import os
from typing import Optional

import torch
from PIL import Image

from engine.config import load_config

logger = logging.getLogger("animate_studio.animate_diff")


class AnimateDiffEngine:
    """Generate real motion clips using AnimateDiff (SD1.5 + motion adapter)."""

    def __init__(self, config_path: str = "config.yaml", config: Optional[dict] = None):
        config = load_config(config_path=config_path, config=config)

        motion_cfg = config.get("motion", {})
        self.enabled = motion_cfg.get("use_animatediff", False)
        self.base_model_id = motion_cfg.get("base_model_id", "runwayml/stable-diffusion-v1-5")
        self.motion_module_path = motion_cfg.get(
            "motion_module_path", "./models/motion_module/mm_sd_v15_v3.ckpt"
        )
        self.motion_adapter_repo = motion_cfg.get(
            "motion_adapter_repo", "guoyww/animatediff-motion-adapter-v1-5-3"
        )
        self.num_frames = motion_cfg.get("num_frames", 16)
        self.gen_height = motion_cfg.get("height", 512)
        self.gen_width = motion_cfg.get("width", 512)
        self.steps = motion_cfg.get("steps", 25)
        self.guidance_scale = motion_cfg.get("guidance_scale", 7.5)
        self.fps = motion_cfg.get("fps", 8)

        lora_cfg = config.get("lora", {})
        self.lora_path = lora_cfg.get("path", "")
        self.lora_hf_repo = lora_cfg.get("hf_repo", "")
        self.lora_trigger = lora_cfg.get("trigger_token", "")
        self.lora_scale = motion_cfg.get("lora_scale", lora_cfg.get("scale", 0.7))

        self.pipe = None
        self._load_error = None
        self._lora_loaded = False

    def load(self):
        """Load the AnimateDiff pipeline (lazy, called on first generate)."""
        if self.pipe is not None:
            return True
        if not self.enabled:
            self._load_error = "AnimateDiff disabled in config"
            return False

        try:
            from diffusers import AnimateDiffPipeline, MotionAdapter, DDIMScheduler
        except ImportError as e:
            self._load_error = f"Missing diffusers classes: {e}"
            logger.error(self._load_error)
            return False

        try:
            # 1. Load motion adapter
            adapter = None
            ckpt = os.path.abspath(self.motion_module_path)
            if os.path.isfile(ckpt):
                try:
                    adapter = MotionAdapter.from_single_file(ckpt, torch_dtype=torch.float16)
                    logger.info("Motion adapter loaded from checkpoint: %s", ckpt)
                except Exception as e:
                    logger.warning("from_single_file failed (%s), trying repo...", e)

            if adapter is None:
                adapter = MotionAdapter.from_pretrained(
                    self.motion_adapter_repo, torch_dtype=torch.float16
                )
                logger.info("Motion adapter loaded from repo: %s", self.motion_adapter_repo)

            # 2. Build AnimateDiff pipeline
            self.pipe = AnimateDiffPipeline.from_pretrained(
                self.base_model_id,
                motion_adapter=adapter,
                torch_dtype=torch.float16,
            )

            # 3. Use DDIM scheduler (good for AnimateDiff)
            self.pipe.scheduler = DDIMScheduler.from_config(
                self.pipe.scheduler.config,
                clip_sample=False,
                timestep_spacing="linspace",
                beta_schedule="linear",
                steps_offset=1,
            )

            # 4. VRAM optimizations
            try:
                self.pipe.enable_vae_slicing()
            except Exception:
                pass
            try:
                self.pipe.enable_attention_slicing()
            except Exception:
                pass

            # 5. Use CPU offload for 8GB cards
            if torch.cuda.is_available():
                try:
                    self.pipe.enable_model_cpu_offload()
                    logger.info("AnimateDiff: CPU offload enabled")
                except Exception:
                    self.pipe = self.pipe.to("cuda")
                    logger.info("AnimateDiff: moved to CUDA directly")
            else:
                self.pipe = self.pipe.to("cpu")

            # 6. Try loading LoRA (may fail if architecture mismatch — that's OK)
            self._try_load_lora()

            logger.info("AnimateDiff pipeline ready.")
            return True

        except Exception as e:
            self._load_error = str(e)
            logger.error("AnimateDiff load failed: %s", e)
            self.pipe = None
            return False

    def _try_load_lora(self):
        """Attempt to load LoRA weights from HF repo or local file. Non-fatal if it fails."""
        if not self.pipe:
            return

        # Priority 1: HuggingFace repo (SD1.5-compatible LoRAs)
        if self.lora_hf_repo:
            try:
                self.pipe.load_lora_weights(
                    self.lora_hf_repo, adapter_name="style_lora"
                )
                self.pipe.set_adapters(["style_lora"], adapter_weights=[self.lora_scale])
                self._lora_loaded = True
                logger.info(
                    "LoRA loaded from HF repo: %s (scale=%.2f, trigger=%s)",
                    self.lora_hf_repo, self.lora_scale, self.lora_trigger or "none",
                )
                return
            except Exception as e:
                logger.warning("HF LoRA load failed (%s): %s", self.lora_hf_repo, e)

        # Priority 2: Local .safetensors file (with architecture pre-check)
        if not self.lora_path:
            return
        lora_abs = os.path.abspath(self.lora_path)
        if not os.path.isfile(lora_abs):
            logger.info("LoRA file not found at %s, skipping.", lora_abs)
            return

        # Check architecture compatibility: SD1.5 uses 768-dim cross-attn,
        # ModelScope LoRA uses 1024-dim. Quick check before loading.
        try:
            from safetensors.torch import load_file
            weights = load_file(lora_abs)
            for k, v in weights.items():
                if "attn2" in k and "to_k" in k and v.shape[-1] in (768, 1024):
                    if v.shape[-1] != 768:
                        logger.info(
                            "LoRA cross-attn dim is %d (need 768 for SD1.5) — skipping LoRA.",
                            v.shape[-1],
                        )
                        return
                    break
            del weights
        except Exception as e:
            logger.warning("Could not pre-check LoRA compatibility: %s", e)

        try:
            lora_dir = os.path.dirname(lora_abs)
            weight_name = os.path.basename(lora_abs)
            self.pipe.load_lora_weights(
                lora_dir, weight_name=weight_name, adapter_name="style_lora"
            )
            self.pipe.set_adapters(["style_lora"], adapter_weights=[self.lora_scale])
            self._lora_loaded = True
            logger.info("LoRA loaded from local file: %s (scale=%.2f)", lora_abs, self.lora_scale)
        except Exception as e:
            logger.warning("LoRA load failed, unloading partial state: %s", e)
            try:
                self.pipe.unload_lora_weights()
            except Exception:
                pass

    def generate_clip(
        self,
        prompt: str,
        negative_prompt: str = "",
        num_frames: Optional[int] = None,
        fps: Optional[int] = None,
        steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        seed: Optional[int] = None,
        gen_height: Optional[int] = None,
        gen_width: Optional[int] = None,
        lora_strength: Optional[float] = None,
    ) -> list[Image.Image]:
        """
        Generate a motion video clip as a list of PIL Image frames.
        Returns empty list on failure so caller can fall back to legacy pipeline.
        """
        if not self.load():
            logger.error("AnimateDiff unavailable: %s", self._load_error)
            return []

        nf = num_frames if num_frames is not None else self.num_frames
        st = steps if steps is not None else self.steps
        gs = guidance_scale if guidance_scale is not None else self.guidance_scale
        height = gen_height if gen_height is not None else self.gen_height
        width = gen_width if gen_width is not None else self.gen_width

        # Apply dynamic LoRA strength override if LoRA is loaded
        if self._lora_loaded and lora_strength is not None:
            try:
                self.pipe.set_adapters(["style_lora"], adapter_weights=[lora_strength])
            except Exception as e:
                logger.warning("Dynamic LoRA strength override failed: %s", e)

        # Prepend LoRA trigger token if LoRA is active
        if self._lora_loaded and self.lora_trigger and self.lora_trigger not in prompt:
            prompt = f"{self.lora_trigger}, {prompt}"

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        logger.info(
            "AnimateDiff generating %d frames (%dx%d, steps=%d, gs=%.1f, seed=%d): %s...",
            nf, width, height, st, gs, seed, prompt[:80],
        )

        try:
            output = self.pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_frames=nf,
                guidance_scale=gs,
                num_inference_steps=st,
                height=height,
                width=width,
                generator=generator,
            )
        except Exception as e:
            logger.error("AnimateDiff generation failed: %s", e)
            return []

        # Extract PIL frames from output
        raw = output.frames
        if isinstance(raw, list) and len(raw) > 0:
            # AnimateDiffPipeline returns list[list[PIL.Image]]
            inner = raw[0] if isinstance(raw[0], list) else raw
        else:
            inner = raw

        frames = []
        for f in inner:
            if isinstance(f, Image.Image):
                frames.append(f)
            else:
                try:
                    frames.append(Image.fromarray(f))
                except Exception:
                    pass

        logger.info("AnimateDiff produced %d frames.", len(frames))
        return frames

    def unload(self):
        """Free VRAM."""
        self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
