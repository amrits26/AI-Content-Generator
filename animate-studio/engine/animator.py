"""
═══════════════════════════════════════════════════════════════
AniMate Studio — Animation Pipeline (ModelScope / CogVideoX)
═══════════════════════════════════════════════════════════════
Scene-by-scene video generation using ModelScope text-to-video
or CogVideoX with character LoRA support, frame-level
transitions, and progressive caching for efficient regeneration.

Optimized for HP Omen RTX 4060/4070 with VRAM-aware settings.
═══════════════════════════════════════════════════════════════
"""

import gc
import glob
import hashlib
import logging
import math
import os
import re
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from PIL import Image, ImageFilter

try:
    import accelerate
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False

from engine.animate_diff_engine import AnimateDiffEngine
from engine.character_manager import CharacterManager
from engine.safety_filter import SafetyFilter
from engine.story_engine import Scene, Storyboard
from utils.ffmpeg_utils import concat_videos, frames_to_video, add_crossfade, interpolate_frames, enhance_video
from utils.prompt_templates import (
    NEGATIVE_PROMPT,
    STYLE_PREFIX,
    build_scene_prompt,
    get_style_preset,
    get_negative_prompt,
    get_camera_motion,
    CAMERA_MOTIONS,
    _AUTO_CAMERA_MAP,
)

logger = logging.getLogger("animate_studio.animator")


class GPUMonitor:
    """Monitor GPU temperature and pause if overheating."""

    def __init__(self, temp_limit: int = 82, cooldown_s: int = 60):
        self.temp_limit = temp_limit
        self.cooldown_s = cooldown_s

    def get_gpu_temp(self) -> Optional[float]:
        """Read GPU temperature via nvidia-smi."""
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True,
            )
            return float(result.stdout.strip().split("\n")[0])
        except Exception:
            return None

    def check_and_wait(self) -> bool:
        """Check GPU temp. If too hot, log warning and wait. Returns True if had to wait."""
        temp = self.get_gpu_temp()
        if temp is None:
            return False
        if temp > self.temp_limit:
            logger.warning(
                "GPU temp %.0f°C exceeds limit %d°C — cooling down for %ds...",
                temp, self.temp_limit, self.cooldown_s,
            )
            time.sleep(self.cooldown_s)
            return True
        return False


class Animator:
    """
    Video generation pipeline (ModelScope / CogVideoX) with scene-by-scene
    generation, frame transitions, character consistency, and safety filtering.
    """

    def __init__(self, config_path: str = "config.yaml", fast_mode: Optional[bool] = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        video_cfg = self.config["models"]["video"]
        perf_cfg = self.config["performance"]

        self.model_name = video_cfg["name"]
        self.dtype = getattr(torch, video_cfg["dtype"])
        self.device = video_cfg["device"]
        self.enable_cpu_offload = video_cfg["enable_cpu_offload"]
        self.enable_vae_tiling = video_cfg["enable_vae_tiling"]
        self.enable_vae_slicing = video_cfg.get("enable_vae_slicing", True)
        self.enable_attn_slicing = video_cfg.get("enable_attention_slicing", True)
        self.num_inference_steps = video_cfg["num_inference_steps"]
        self.guidance_scale = video_cfg["guidance_scale"]
        self.frames_per_scene = video_cfg["frames_per_scene"]
        self.fps = video_cfg["fps"]
        self.gen_height = video_cfg.get("height", 256)
        self.gen_width = video_cfg.get("width", 256)
        self.vram_safety_margin_gb = video_cfg.get("vram_safety_margin_gb", 1.0)

        # Style / camera
        self.style = self.config["models"].get("style", "pixar_cute")
        self.camera_motion = self.config["models"].get("camera_motion", "auto")

        # fast_mode: param overrides config; config overrides default (False)
        if fast_mode is not None:
            self.fast_mode = fast_mode
        else:
            self.fast_mode = video_cfg.get("fast_mode", False)

        self.output_dir = self.config["app"]["output_dir"]
        self.cache_dir = perf_cfg["cache_dir"]
        self.enable_caching = perf_cfg["enable_scene_caching"]
        self.vram_pause_threshold = perf_cfg.get("vram_pause_threshold", 0.90)
        self.empty_cache_between_scenes = perf_cfg.get("torch_empty_cache_between_scenes", True)

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

        self.gpu_monitor = GPUMonitor(
            temp_limit=perf_cfg["gpu_temp_limit_c"],
            cooldown_s=perf_cfg["cooldown_seconds"],
        )

        self.character_manager = CharacterManager(config_path)
        self.safety_filter = SafetyFilter(config_path)

        # AnimateDiff real-motion engine
        self._motion_cfg = self.config.get("motion", {})
        self.use_animatediff = self._motion_cfg.get("use_animatediff", False)
        self._animatediff = AnimateDiffEngine(config_path) if self.use_animatediff else None

        self._pipeline = None

    # ── Pipeline Management ──────────────────────────────
    def load_pipeline(self):
        """Load video generation pipeline with VRAM optimizations."""
        if self._pipeline is not None:
            return

        # Auto-detect pipeline class based on model name
        model_lower = self.model_name.lower()
        if "cogvideo" in model_lower:
            from diffusers import CogVideoXPipeline as PipelineClass
        else:
            from diffusers import DiffusionPipeline as PipelineClass

        logger.info("Loading video pipeline: %s", self.model_name)

        load_kwargs = dict(
            torch_dtype=self.dtype,
            low_cpu_mem_usage=True,
        )
        # Use fp16 variant if available (ModelScope provides fp16 safetensors)
        if "cogvideo" not in model_lower:
            load_kwargs["variant"] = "fp16"

        self._pipeline = PipelineClass.from_pretrained(
            self.model_name,
            **load_kwargs,
        )

        # ── Upgrade scheduler for better quality ─────────
        # DPMSolverMultistep produces significantly cleaner output
        # than the default PNDM scheduler at the same step count
        try:
            from diffusers import DPMSolverMultistepScheduler
            self._pipeline.scheduler = DPMSolverMultistepScheduler.from_config(
                self._pipeline.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
            logger.info("Scheduler upgraded to DPMSolver++ with Karras sigmas.")
        except Exception as e:
            logger.info("Keeping default scheduler: %s", e)

        if self.enable_cpu_offload:
            if not _HAS_ACCELERATE:
                logger.warning(
                    "accelerate not installed -- CPU offload may be slow. "
                    "Run: pip install accelerate"
                )
                self._pipeline.enable_sequential_cpu_offload()
                logger.info("Sequential CPU offload enabled (no accelerate).")
            else:
                try:
                    self._pipeline.enable_model_cpu_offload()
                    logger.info("CPU offload enabled.")
                except (RuntimeError, AttributeError) as e:
                    if "accelerate" in str(e).lower():
                        logger.warning(
                            "enable_model_cpu_offload requires accelerate -- "
                            "falling back to sequential offloading. Error: %s", e
                        )
                        self._pipeline.enable_sequential_cpu_offload()
                        logger.info("Sequential CPU offload enabled (fallback).")
                    else:
                        raise
        else:
            self._pipeline = self._pipeline.to(self.device)

        # ── VRAM optimizations for 8GB cards ─────────────
        if self.enable_vae_tiling:
            try:
                self._pipeline.vae.enable_tiling()
                logger.info("VAE tiling enabled.")
            except AttributeError:
                logger.info("VAE tiling not supported by this pipeline.")

        if self.enable_vae_slicing:
            try:
                self._pipeline.vae.enable_slicing()
                logger.info("VAE slicing enabled.")
            except AttributeError:
                logger.info("VAE slicing not supported by this pipeline.")

        if self.enable_attn_slicing:
            try:
                self._pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled.")
            except AttributeError:
                logger.info("Attention slicing not supported by this pipeline.")

        logger.info("Video pipeline loaded.")

    def _check_vram_pressure(self):
        """Pause if VRAM usage exceeds threshold. Clear cache if needed."""
        if not torch.cuda.is_available():
            return
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        usage_pct = allocated / total if total > 0 else 0
        if usage_pct > self.vram_pause_threshold:
            logger.warning(
                "VRAM usage %.0f%% exceeds threshold %.0f%% — clearing cache...",
                usage_pct * 100, self.vram_pause_threshold * 100,
            )
            gc.collect()
            torch.cuda.empty_cache()

    def _flush_vram(self):
        """Free VRAM between scene generations."""
        if self.empty_cache_between_scenes and torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()

    def unload_pipeline(self):
        """Free VRAM by unloading the pipeline."""
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipeline unloaded — VRAM freed.")

    # ── Scene Generation ─────────────────────────────────
    def generate_scene(
        self,
        scene: Scene,
        character_name: Optional[str] = None,
        character_prompt: str = "",
        prev_last_frame: Optional[Image.Image] = None,
        seed: Optional[int] = None,
        style: Optional[str] = None,
        camera_motion: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        gen_height: Optional[int] = None,
        gen_width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        lora_strength: Optional[float] = None,
    ) -> dict:
        """
        Generate a single scene as a sequence of video frames.

        Args:
            scene: Scene object with visual description and metadata
            character_name: Character to load LoRA for
            character_prompt: Full character description for prompt
            prev_last_frame: Last frame of previous scene (for transitions)
            seed: Random seed for reproducibility
            style: Style preset override (default: use config)
            camera_motion: Camera motion override (default: use config)
            num_inference_steps: Override inference steps (default: use config)
            gen_height: Override generation height (default: use config)
            gen_width: Override generation width (default: use config)
            guidance_scale: Override guidance scale (default: use config)
            lora_strength: Override LoRA strength (default: use character profile)

        Returns:
            dict with "frames" (list[PIL.Image]), "video_path", "seed"
        """
        # Resolve per-request overrides
        effective_steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        effective_height = gen_height if gen_height is not None else self.gen_height
        effective_width = gen_width if gen_width is not None else self.gen_width
        effective_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        # ── AnimateDiff path: real motion clips ──────────
        if self.use_animatediff and self._animatediff:
            result = self._generate_scene_animatediff(
                scene, character_prompt, seed, style, camera_motion,
                num_inference_steps=effective_steps,
                guidance_scale=effective_guidance,
                gen_height=effective_height,
                gen_width=effective_width,
                lora_strength=lora_strength,
            )
            if result and result.get("video_path"):
                return result
            logger.warning(
                "AnimateDiff failed for scene %d — falling back to legacy pipeline.",
                scene.scene_id,
            )

        # ── Legacy ModelScope / CogVideoX path ──────────
        self.load_pipeline()
        self.gpu_monitor.check_and_wait()
        self._check_vram_pressure()

        # Resolve style and camera
        active_style = style or self.style
        active_camera = camera_motion or self.camera_motion
        preset = get_style_preset(active_style)
        style_prefix = preset["prefix"]
        negative = preset["negative"]

        # Camera motion descriptor
        if active_camera == "auto":
            cam_key = _AUTO_CAMERA_MAP.get(scene.emotion_tone, "static")
        else:
            cam_key = active_camera
        cam_desc = get_camera_motion(cam_key)

        # Build the full prompt
        prompt_parts = [style_prefix]
        if cam_desc:
            prompt_parts.append(cam_desc)
        if character_prompt:
            prompt_parts.append(character_prompt)
        prompt_parts.append(scene.visual_description)
        prompt = ", ".join(prompt_parts)

        logger.info("Generating scene %d [%s/%s]: %s...", scene.scene_id, active_style, cam_key, prompt[:80])

        # Load character LoRA if available
        if character_name:
            self.character_manager.load_lora_into_pipeline(
                self._pipeline, character_name,
                strength_override=lora_strength,
            )

        # Set seed for reproducibility
        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()
        generator = torch.Generator(device="cpu").manual_seed(seed)

        # Build generation kwargs — include height/width if pipeline supports it
        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_frames=self.frames_per_scene,
            num_inference_steps=effective_steps,
            guidance_scale=effective_guidance,
            generator=generator,
        )
        # Add height/width for pipelines that support it
        try:
            import inspect
            call_sig = inspect.signature(self._pipeline.__call__)
            if "height" in call_sig.parameters:
                gen_kwargs["height"] = effective_height
            if "width" in call_sig.parameters:
                gen_kwargs["width"] = effective_width
        except Exception:
            pass  # fallback: let pipeline use its defaults

        # Generate video frames
        output = self._pipeline(**gen_kwargs)

        # Handle different output formats:
        # CogVideoX returns list of PIL Images in output.frames[0]
        # ModelScope returns numpy array with shape (batch, frames, H, W, C)
        raw_frames = output.frames
        if isinstance(raw_frames, np.ndarray):
            # ModelScope: shape (1, num_frames, H, W, 3) float32 [0..1] or uint8
            arr = np.squeeze(raw_frames)  # remove batch dim → (num_frames, H, W, 3)
            if arr.ndim == 4:
                frame_list = [arr[i] for i in range(arr.shape[0])]
            else:
                frame_list = [arr]
            frames = []
            for fr in frame_list:
                if fr.dtype != np.uint8:
                    fr = np.clip(fr * 255, 0, 255).astype(np.uint8)
                frames.append(Image.fromarray(fr))
        elif isinstance(raw_frames, list) and len(raw_frames) > 0:
            # CogVideoX: list of lists of PIL Images
            inner = raw_frames[0] if isinstance(raw_frames[0], list) else raw_frames
            frames = []
            for fr in inner:
                if isinstance(fr, Image.Image):
                    frames.append(fr)
                elif isinstance(fr, np.ndarray):
                    if fr.dtype != np.uint8:
                        fr = np.clip(fr * 255, 0, 255).astype(np.uint8)
                    frames.append(Image.fromarray(np.squeeze(fr)))
                else:
                    frames.append(fr)
        else:
            frames = list(raw_frames)

        # Safety check on generated frames
        # Scale sample rate relative to frame count for consistent coverage
        num_generated = len(frames)
        sample_rate = max(1, num_generated // 4) if self.fast_mode else max(1, num_generated // 8)
        safety_result = self.safety_filter.scan_frames_batch(frames, sample_rate=sample_rate)
        # Unload CLIP from GPU immediately to free VRAM for next scene
        self.safety_filter.unload_clip()
        if not safety_result.passed:
            logger.warning(
                "Scene %d failed safety — flagged: %s",
                scene.scene_id, safety_result.flagged_concepts,
            )
            return {
                "frames": frames,
                "video_path": "",
                "seed": seed,
                "safety_passed": False,
                "safety_result": safety_result,
            }

        # Save frames to disk
        scene_dir = os.path.join(
            self.output_dir, f"scene_{scene.scene_id:03d}_seed{seed}"
        )
        os.makedirs(scene_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            frame.save(os.path.join(scene_dir, f"frame_{i:04d}.png"))

        # Convert frames to video
        scene_video = os.path.join(scene_dir, f"scene_{scene.scene_id:03d}.mp4")
        frames_to_video(scene_dir, scene_video, fps=self.fps)

        # Post-processing: frame interpolation → sharpen/denoise → enhance color
        try:
            # Interpolate to 2x fps for smoother motion
            interp_video = os.path.join(scene_dir, f"scene_{scene.scene_id:03d}_interp.mp4")
            interpolate_frames(scene_video, interp_video, target_fps=self.fps * 2)
            # Enhance: denoise AI artifacts, sharpen, boost saturation
            enhanced_video = os.path.join(scene_dir, f"scene_{scene.scene_id:03d}_enhanced.mp4")
            enhance_video(interp_video, enhanced_video, sharpen=True, denoise=True, boost_saturation=1.15)
            scene_video = enhanced_video
            logger.info("Scene %d post-processed (interpolation + enhancement).", scene.scene_id)
        except Exception as e:
            logger.warning("Post-processing failed for scene %d, using raw video: %s", scene.scene_id, e)

        # Flush VRAM between scenes to prevent OOM
        self._flush_vram()

        return {
            "frames": frames,
            "video_path": scene_video,
            "seed": seed,
            "safety_passed": True,
            "safety_result": safety_result,
        }

    def _generate_scene_animatediff(
        self,
        scene: Scene,
        character_prompt: str,
        seed: Optional[int],
        style: Optional[str],
        camera_motion: Optional[str],
        num_inference_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        gen_height: Optional[int] = None,
        gen_width: Optional[int] = None,
        lora_strength: Optional[float] = None,
    ) -> Optional[dict]:
        """Generate a scene using AnimateDiff with multi-clip looping."""
        # Use compact AnimateDiff-optimized style to stay under 77 CLIP tokens
        active_style = style if style is not None else "animatediff_cartoon"
        active_camera = camera_motion or self.camera_motion
        preset = get_style_preset(active_style)
        style_prefix = preset["prefix"]
        negative = preset["negative"]

        if active_camera == "auto":
            cam_key = _AUTO_CAMERA_MAP.get(scene.emotion_tone, "static")
        else:
            cam_key = active_camera
        cam_desc = get_camera_motion(cam_key)

        # Build compact prompt: style + camera + visual_description only
        # Skip character_prompt (already embedded in visual_description for E2E)
        prompt_parts = [style_prefix]
        if cam_desc:
            prompt_parts.append(cam_desc)
        # Use visual_description directly (compact scene details)
        prompt_parts.append(scene.visual_description)
        prompt = ", ".join(prompt_parts)

        if seed is None:
            seed = torch.randint(0, 2**32, (1,)).item()

        motion_cfg = self._motion_cfg
        clip_frames = motion_cfg.get("num_frames", 16)
        clip_fps = motion_cfg.get("fps", 8)
        effective_steps = num_inference_steps if num_inference_steps is not None else motion_cfg.get("steps", 25)
        effective_guidance = guidance_scale if guidance_scale is not None else motion_cfg.get("guidance_scale", 7.5)
        clip_seconds = clip_frames / max(clip_fps, 1)
        target_seconds = max(getattr(scene, "duration_s", clip_seconds) or clip_seconds, clip_seconds)
        clips_needed = max(1, math.ceil(target_seconds / clip_seconds))

        logger.info(
            "AnimateDiff scene %d: %d clip(s) x %d frames @ %d fps [%s/%s]",
            scene.scene_id, clips_needed, clip_frames, clip_fps, active_style, cam_key,
        )

        scene_dir = os.path.join(self.output_dir, f"scene_{scene.scene_id:03d}_seed{seed}")
        os.makedirs(scene_dir, exist_ok=True)

        clip_paths = []
        all_frames = []

        for ci in range(clips_needed):
            clip_seed = (seed + ci * 7919) % (2**32)
            frames = self._animatediff.generate_clip(
                prompt=prompt,
                negative_prompt=negative,
                num_frames=clip_frames,
                fps=clip_fps,
                steps=effective_steps,
                guidance_scale=effective_guidance,
                seed=clip_seed,
                gen_height=gen_height,
                gen_width=gen_width,
                lora_strength=lora_strength,
            )
            if not frames:
                return None  # signal fallback

            all_frames.extend(frames)

            # Save frames + encode clip
            for fi, fr in enumerate(frames):
                fr.save(os.path.join(scene_dir, f"clip{ci:02d}_frame_{fi:04d}.png"))
            clip_path = os.path.join(scene_dir, f"clip_{ci:02d}.mp4")
            frames_to_video(frames, clip_path, fps=clip_fps)
            clip_paths.append(clip_path)

            self._flush_vram()

        # Safety check
        num_generated = len(all_frames)
        sample_rate = max(1, num_generated // 4) if self.fast_mode else max(1, num_generated // 8)
        safety_result = self.safety_filter.scan_frames_batch(all_frames, sample_rate=sample_rate)
        self.safety_filter.unload_clip()
        if not safety_result.passed:
            logger.warning("Scene %d failed safety (AnimateDiff).", scene.scene_id)
            return {
                "frames": all_frames, "video_path": "", "seed": seed,
                "safety_passed": False, "safety_result": safety_result,
            }

        # Merge clips into scene video
        scene_video = os.path.join(scene_dir, f"scene_{scene.scene_id:03d}.mp4")
        if len(clip_paths) == 1:
            shutil.copy2(clip_paths[0], scene_video)
        else:
            concat_videos(clip_paths, scene_video)

        # Enhance
        try:
            enhanced = os.path.join(scene_dir, f"scene_{scene.scene_id:03d}_enhanced.mp4")
            enhance_video(scene_video, enhanced, sharpen=True, denoise=True, boost_saturation=1.15)
            scene_video = enhanced
        except Exception as e:
            logger.warning("Enhancement failed for scene %d: %s", scene.scene_id, e)

        return {
            "frames": all_frames,
            "video_path": scene_video,
            "seed": seed,
            "safety_passed": True,
            "safety_result": safety_result,
        }

    def generate_scene_with_retry(
        self,
        scene: Scene,
        character_name: Optional[str] = None,
        character_prompt: str = "",
        prev_last_frame: Optional[Image.Image] = None,
        max_attempts: int = 3,
        style: Optional[str] = None,
        camera_motion: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        gen_height: Optional[int] = None,
        gen_width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        lora_strength: Optional[float] = None,
    ) -> dict:
        """
        Generate a scene with automatic retry on safety failure.
        Mutates seed on each retry.
        """
        for attempt in range(max_attempts):
            seed = torch.randint(0, 2**32, (1,)).item()
            logger.info(
                "Scene %d — attempt %d/%d (seed=%d)",
                scene.scene_id, attempt + 1, max_attempts, seed,
            )

            result = self.generate_scene(
                scene=scene,
                character_name=character_name,
                character_prompt=character_prompt,
                prev_last_frame=prev_last_frame,
                seed=seed,
                style=style,
                camera_motion=camera_motion,
                num_inference_steps=num_inference_steps,
                gen_height=gen_height,
                gen_width=gen_width,
                guidance_scale=guidance_scale,
                lora_strength=lora_strength,
            )

            if result.get("safety_passed", False):
                return result

            logger.warning("Attempt %d failed safety — retrying with new seed...", attempt + 1)

        logger.error(
            "Scene %d failed all %d safety attempts.",
            scene.scene_id, max_attempts,
        )
        return result  # Return last attempt even if failed

    # ── Full Episode Generation ──────────────────────────
    def generate_episode(
        self,
        storyboard: Storyboard,
        character_name: Optional[str] = None,
        progress_callback=None,
        style: Optional[str] = None,
        camera_motion: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        gen_height: Optional[int] = None,
        gen_width: Optional[int] = None,
        fps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        lora_strength: Optional[float] = None,
    ) -> dict:
        """
        Generate a complete episode from a storyboard.
        Scene-by-scene with transitions and safety checks.

        Args:
            storyboard: Complete storyboard with scenes
            character_name: Character to maintain consistency for
            progress_callback: fn(scene_idx, total, status_msg) for UI updates
            style: Style preset override (default: use config)
            camera_motion: Camera motion override (default: use config)
            num_inference_steps: Override inference steps (default: use config)
            gen_height: Override generation height (default: use config)
            gen_width: Override generation width (default: use config)
            fps: Override frames per second (default: use config)
            guidance_scale: Override guidance scale (default: use config)
            lora_strength: Override LoRA strength (default: use character profile)

        Returns:
            dict with "video_path", "scenes", "safety_results", "duration_s"
        """
        # Resolve per-request overrides
        effective_style = style if style is not None else self.style
        effective_camera = camera_motion if camera_motion is not None else self.camera_motion
        effective_steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
        effective_height = gen_height if gen_height is not None else self.gen_height
        effective_width = gen_width if gen_width is not None else self.gen_width
        effective_fps = fps if fps is not None else self.fps
        effective_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        effective_lora_strength = lora_strength  # None means use character profile default
        # Only load legacy pipeline if AnimateDiff is disabled
        if not self.use_animatediff:
            self.load_pipeline()

        character_prompt = ""
        if character_name:
            character_prompt = self.character_manager.get_character_prompt(character_name)

        scene_videos = []
        scene_results = []
        all_safety_results = []
        prev_last_frame = None
        total = len(storyboard.scenes)

        for i, scene in enumerate(storyboard.scenes):
            if progress_callback:
                progress_callback(
                    i, total,
                    f"Generating scene {i+1}/{total}: {scene.narration[:40]}...",
                )

            # Check cache
            cache_key = self._scene_cache_key(
                scene, character_name,
                style=effective_style, camera_motion=effective_camera,
                num_inference_steps=effective_steps, gen_height=effective_height,
                gen_width=effective_width, guidance_scale=effective_guidance,
                lora_strength=effective_lora_strength,
            )
            cached = self._get_cached_scene(cache_key)
            if cached:
                logger.info("Scene %d loaded from cache.", scene.scene_id)
                scene_videos.append(cached["video_path"])
                scene_results.append(cached)
                continue

            result = self.generate_scene_with_retry(
                scene=scene,
                character_name=character_name,
                character_prompt=character_prompt,
                prev_last_frame=prev_last_frame,
                style=effective_style,
                camera_motion=effective_camera,
                num_inference_steps=effective_steps,
                gen_height=effective_height,
                gen_width=effective_width,
                guidance_scale=effective_guidance,
                lora_strength=effective_lora_strength,
            )

            if result.get("video_path"):
                scene_videos.append(result["video_path"])
                scene_results.append(result)
                all_safety_results.append(result.get("safety_result"))

                # Get last frame for transition continuity
                if result.get("frames"):
                    prev_last_frame = result["frames"][-1]

                # Cache the scene
                if self.enable_caching:
                    self._cache_scene(cache_key, result)

            if progress_callback:
                progress_callback(i + 1, total, f"Scene {i+1}/{total} complete.")

        if not scene_videos:
            return {"video_path": "", "error": "No scenes generated successfully."}

        # Concatenate all scene videos with crossfades
        if progress_callback:
            progress_callback(total, total, "Concatenating scenes...")

        safe_title = re.sub(r'[<>:"/\\|?*]', '_', storyboard.title).replace(' ', '_')[:80]
        episode_path = os.path.join(
            self.output_dir,
            f"{safe_title}_episode.mp4",
        )

        if len(scene_videos) == 1:
            shutil.copy2(scene_videos[0], episode_path)
        else:
            # Apply crossfade transitions between scenes
            try:
                merged = scene_videos[0]
                for j in range(1, len(scene_videos)):
                    temp_out = os.path.join(self.output_dir, f"_merge_step_{j}.mp4")
                    try:
                        add_crossfade(merged, scene_videos[j], temp_out, fade_duration=0.3)
                        merged = temp_out
                    except Exception as e:
                        logger.warning("Crossfade failed, using hard cut: %s", e)
                        concat_videos([merged, scene_videos[j]], temp_out)
                        merged = temp_out

                shutil.move(merged, episode_path)
            finally:
                # Clean up temp merge files even on exception
                for tmp in glob.glob(os.path.join(self.output_dir, "_merge_step_*.mp4")):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

        return {
            "video_path": episode_path,
            "scenes": scene_results,
            "safety_results": all_safety_results,
            "duration_s": storyboard.total_duration_s,
        }

    # ── Scene Caching ────────────────────────────────────
    def _scene_cache_key(
        self,
        scene: Scene,
        character_name: Optional[str],
        style: Optional[str] = None,
        camera_motion: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        gen_height: Optional[int] = None,
        gen_width: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        lora_strength: Optional[float] = None,
    ) -> str:
        """Generate a cache key from scene content + all generation parameters."""
        parts = [
            scene.visual_description,
            scene.emotion_tone,
            scene.setting,
            str(character_name),
            str(style if style is not None else self.style),
            str(camera_motion if camera_motion is not None else self.camera_motion),
            str(num_inference_steps if num_inference_steps is not None else self.num_inference_steps),
            str(gen_height if gen_height is not None else self.gen_height),
            str(gen_width if gen_width is not None else self.gen_width),
            str(guidance_scale if guidance_scale is not None else self.guidance_scale),
            str(lora_strength) if lora_strength is not None else "default",
            "animatediff" if self.use_animatediff else "legacy",
        ]
        content = "|".join(parts)
        return hashlib.sha256(content.encode()).hexdigest()

    def _get_cached_scene(self, cache_key: str) -> Optional[dict]:
        """Check if a scene is cached and return its data."""
        if not self.enable_caching:
            return None
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        if os.path.exists(cache_file):
            import json
            with open(cache_file, "r") as f:
                data = json.load(f)
            if os.path.exists(data.get("video_path", "")):
                return data
        return None

    def _cache_scene(self, cache_key: str, result: dict):
        """Save scene result to cache."""
        import json
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.json")
        cache_data = {
            "video_path": result.get("video_path", ""),
            "seed": result.get("seed", 0),
            "safety_passed": result.get("safety_passed", False),
        }
        with open(cache_file, "w") as f:
            json.dump(cache_data, f)
