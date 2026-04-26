import datetime
from datetime import datetime, timedelta
import json
import time
import os
import re
import math
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
import datetime
from datetime import datetime
import json
import time
import shutil
import time
from pathlib import Path
from typing import Optional

import threading
import threading
import time

# Add tenacity for retry/circuit breaker
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, before_sleep_log

import numpy as np
import torch
from PIL import Image, ImageFilter

try:
    import accelerate
    _HAS_ACCELERATE = True
except ImportError:
    _HAS_ACCELERATE = False

from engine.animate_diff_engine import AnimateDiffEngine
from engine.character_manager import CharacterManager
from engine.config import load_config
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

    _shared_pipeline = None
    _shared_pipeline_key: Optional[str] = None
    _shared_animatediff: Optional[AnimateDiffEngine] = None
    _shared_animatediff_key: Optional[str] = None

    def __init__(self, config_path: str = "config.yaml", fast_mode: Optional[bool] = None, config: Optional[dict] = None):
        self.config_path = config_path
        self.config = load_config(config_path=config_path, config=config)

        # Motion Mastery: Advanced Motion Control
        self.motion_smoothness = self.config.get("motion", {}).get("motion_smoothness", 0.85)

        # Beast Mode config
        self.beast_mode_cfg = self.config.get("beast_mode", {})
        self.beast_mode_enabled = self.beast_mode_cfg.get("enabled", False)

        video_cfg = self.config["models"]["video"]
        perf_cfg = self.config["performance"]
        # Inject Beast Mode upgrades if enabled
        if self.beast_mode_enabled:
            self._enable_beast_mode(self.beast_mode_cfg)

    def _enable_beast_mode(self, beast_cfg):
        video_cfg = self.config["models"]["video"]
        perf_cfg = self.config.get("performance", {})
        fast_mode = beast_cfg.get("fast_mode", False)

        # FreeU
        try:
            if beast_cfg.get("freeu", {}).get("enabled", False):
                if hasattr(self, "_pipeline") and self._pipeline is not None:
                    self._pipeline.enable_freeu(
                        s1=beast_cfg["freeu"].get("s1", 0.9),
                        s2=beast_cfg["freeu"].get("s2", 0.2),
                        b1=beast_cfg["freeu"].get("b1", 1.2),
                        b2=beast_cfg["freeu"].get("b2", 1.4),
                    )
        except Exception as e:
            logger.warning(f"FreeU not enabled: {e}")

    def generate_episode(
        self,
        manifest: dict,
        style_lock: Optional[dict] = None,
        progress_callback=None,
        style: Optional[str] = None,
        camera_motion: Optional[str] = None,
        num_inference_steps: Optional[int] = None,
        gen_height: Optional[int] = None,
        gen_width: Optional[int] = None,
        fps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        lora_strength: Optional[float] = None,
        motion_smoothness: Optional[float] = None,
    ) -> dict:
        """
        Generate a complete episode from a manifest (agentic workflow).
        Scene-by-scene with transitions, style lock, and safety checks.
        """
        # Add audio_sync flag to manifest for lip-sync readiness
        manifest["audio_sync"] = True

        # Use provided or default motion_smoothness
        msmooth = motion_smoothness if motion_smoothness is not None else self.motion_smoothness
        effective_style = style if style is not None else getattr(self, "style", None)
        effective_camera = camera_motion if camera_motion is not None else getattr(self, "camera_motion", None)
        effective_steps = num_inference_steps if num_inference_steps is not None else getattr(self, "num_inference_steps", None)
        effective_height = gen_height if gen_height is not None else getattr(self, "gen_height", None)
        effective_width = gen_width if gen_width is not None else getattr(self, "gen_width", None)
        effective_fps = fps if fps is not None else getattr(self, "fps", None)
        effective_guidance = guidance_scale if guidance_scale is not None else getattr(self, "guidance_scale", None)
        effective_lora_strength = lora_strength
        if not getattr(self, "use_animatediff", False):
            self.load_pipeline()

        # Style lock: inject color/texture/accessory into prompt
        style_lock_str = ""
        if style_lock:
            style_lock_str = ", ".join(f"{k}: {v}" for k, v in style_lock.items())

        character_name = manifest["character"]["name"]
        character_type = manifest["character"]["type"]
        character_prompt = f"{character_name} the {character_type}"
        if style_lock_str:
            character_prompt += f", {style_lock_str}"

        scene_videos = []
        scene_results = []
        all_safety_results = []
        prev_last_frame = None
        prev_noise_seed = None  # For temporal consistency
        scenes = manifest["scenes"]
        total = len(scenes)

        for i, scene in enumerate(scenes):
            if progress_callback:
                progress_callback(
                    i, total,
                    f"Generating scene {i+1}/{total}: {scene.get('action','')[:40]}...",
                )

            # Dynamic Pacing Agent: adjust smoothness/camera for long narration
            narration_text = scene.get("narration") or scene.get("action", "")
            narration_len = len(narration_text.split())
            pacing_smoothness = msmooth
            pacing_camera = scene.get("camera", effective_camera)
            if narration_len > 30:
                pacing_smoothness = min(1.0, msmooth + 0.1)
                pacing_camera = "slow-zoom"

            # Build masterpiece prompt
            prompt_parts = [effective_style, character_prompt, scene.get("action", ""), pacing_camera]
            masterpiece_prompt = ", ".join([p for p in prompt_parts if p])

            # Sampler/CFG lock
            sampler = "DPMSolver++"  # locked for cinematic
            cfg_scale = effective_guidance

            # LoRA injection (if available)
            # (Assume character_manager handles LoRA by name)

            # Generate scene using the masterpiece prompt
            # (Reuse generate_scene_with_retry, but adapt for dict scene)
            scene_obj = Scene(
                scene_id=scene.get("scene_id", i+1),
                narration=narration_text,
                visual_description=masterpiece_prompt,
                emotion_tone=scene.get("emotion_tone", ""),
                setting=scene.get("setting", ""),
                duration_s=scene.get("duration_s", 5.0),
            )
            # Temporal Consistency: lock noise seed for frame-to-frame stability
            if prev_noise_seed is not None:
                locked_seed = prev_noise_seed
            else:
                locked_seed = None
            result = self.generate_scene_with_retry(
                scene=scene_obj,
                character_name=character_name,
                character_prompt=character_prompt,
                prev_last_frame=prev_last_frame,
                style=effective_style,
                camera_motion=pacing_camera,
                num_inference_steps=effective_steps,
                gen_height=effective_height,
                gen_width=effective_width,
                guidance_scale=cfg_scale,
                lora_strength=effective_lora_strength,
                motion_smoothness=pacing_smoothness,
                locked_seed=locked_seed,
            )

            # Color Grade Master: apply LUT based on mood
            if result.get("video_path"):
                # Determine LUT by mood
                mood = scene.get("emotion_tone", "").lower()
                lut = None
                if "warm" in mood or "happy" in mood or "vintage" in mood:
                    lut = "warm_gold.cube"
                elif "mystery" in mood or "cool" in mood or "sad" in mood:
                    lut = "cool_blue.cube"
                # Only apply LUT if found
                if lut:
                    try:
                        from utils.ffmpeg_utils import apply_lut
                        apply_lut(result["video_path"], result["video_path"], lut)
                        logger.info(f"Applied LUT {lut} to {result['video_path']}")
                    except Exception as e:
                        logger.warning(f"LUT application failed: {e}")
                scene_videos.append(result["video_path"])
                scene_results.append(result)
                all_safety_results.append(result.get("safety_result"))
                if result.get("frames"):
                    prev_last_frame = result["frames"][-1]
                # Asset Binder: lock noise seed for next scene
                if result.get("seed") is not None:
                    prev_noise_seed = result["seed"]
            if getattr(self, "enable_caching", False):
                self._cache_scene(f"scene_{scene_obj.scene_id}", result)
            if progress_callback:
                progress_callback(i + 1, total, f"Scene {i+1}/{total} complete.")

        if not scene_videos:
            return {"video_path": "", "error": "No scenes generated successfully."}

        if progress_callback:
            progress_callback(total, total, "Concatenating scenes...")

        safe_title = re.sub(r'[<>:"/\\|?*]', '_', manifest["title"]).replace(' ', '_')[:80]
        episode_path = os.path.join(
            self.output_dir,
            f"{safe_title}_episode.mp4",
        )

        if len(scene_videos) == 1:
            shutil.copy2(scene_videos[0], episode_path)
        else:
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
                for tmp in glob.glob(os.path.join(self.output_dir, "_merge_step_*.mp4")):
                    try:
                        os.remove(tmp)
                    except Exception:
                        pass

        duration_s = sum(scene.get("duration_s", 0) for scene in scenes)
        return {
            "video_path": episode_path,
            "scenes": scene_results,
            "safety_results": all_safety_results,
            "duration_s": duration_s,
        }

        if self.enable_attn_slicing:
            try:
                self._pipeline.enable_attention_slicing()
                logger.info("Attention slicing enabled.")
            except AttributeError:
                logger.info("Attention slicing not supported by this pipeline.")

        logger.info("Video pipeline loaded.")
        self.__class__._shared_pipeline = self._pipeline
        self.__class__._shared_pipeline_key = self.model_version

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
        if self.__class__._shared_pipeline is self._pipeline:
            self.__class__._shared_pipeline = None
            self.__class__._shared_pipeline_key = None
        self._pipeline = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Pipeline unloaded — VRAM freed.")

    def cleanup_request_state(self):
        """Reset mutable request-scoped state from the shared pipeline."""
        pipeline = self._pipeline or self.__class__._shared_pipeline
        if pipeline is not None:
            self.character_manager.unload_lora(pipeline)

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
        locked_seed: Optional[int] = None,

        character_ref_image: Optional[Image.Image] = None,
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
        # Dynamic CFG rescaling (Beast Mode)
        if self.beast_mode_enabled:
            effective_guidance = 8.5
            if scene and ("action" in scene.visual_description.lower() or "running" in scene.visual_description.lower()):
                effective_guidance = 9.5
        else:
            effective_guidance = guidance_scale if guidance_scale is not None else self.guidance_scale
        # ── AnimateDiff path: real motion clips ──────────
        if self.use_animatediff and self._animatediff:
            if getattr(self, "beast_mode_enabled", False):
                effective_steps = self.beast_mode_steps
            else:
                effective_steps = num_inference_steps if num_inference_steps is not None else self.num_inference_steps
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

        # Seed locking (Beast Mode)
        if self.beast_mode_enabled and locked_seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(locked_seed)
        else:
            if seed is None:
                seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device="cpu").manual_seed(seed)

        if getattr(self, "beast_mode_enabled", False):
            prompt = scene.visual_description  # Only the raw theme string
        else:
            prompt_parts = [style_prefix]
            if cam_desc:
                prompt_parts.append(cam_desc)
            if character_prompt:
                prompt_parts.append(character_prompt)
            prompt_parts.append(scene.visual_description)
            prompt = ", ".join(prompt_parts)
        # Build generation kwargs
        gen_kwargs = dict(
            prompt=prompt,
            negative_prompt=negative,
            num_frames=self.frames_per_scene,
            num_inference_steps=effective_steps,
            guidance_scale=effective_guidance,
            generator=generator,
        )
        # Generate video frames
        output = self._pipeline(**gen_kwargs)
        # IP-Adapter logic (Beast Mode)
        if self.beast_mode_enabled and hasattr(self, 'ip_adapter') and character_ref_image is not None:
            try:
                self._pipeline.set_ip_adapter_scale(self.beast_mode_cfg.get("ip_adapter", {}).get("scale", 0.7))
                # Assume encode_reference_image exists or is handled elsewhere
            except Exception as e:
                logger.warning(f"IP-Adapter scale not set: {e}")
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
        safety_cfg = self.config.get("models", {}).get("safety", {})
        if self.fast_mode:
            sample_rate = safety_cfg.get("frame_sample_rate_fast", 4)
        else:
            sample_rate = safety_cfg.get("frame_sample_rate", 1)
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

        # Track usage (best-effort)
        try:
            from engine.usage_tracker import get_tracker
            get_tracker(self.config).log_video_generation(
                duration_s=scene.duration_s,
                resolution=f"{gen_width}x{gen_height}",
                fps=self.fps,
                pipeline_type="legacy",
            )
        except Exception:
            logger.debug("Usage tracking failed for video generation", exc_info=True)

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
        safety_cfg = self.config.get("models", {}).get("safety", {})
        if self.fast_mode:
            sample_rate = safety_cfg.get("frame_sample_rate_fast", 4)
        else:
            sample_rate = safety_cfg.get("frame_sample_rate", 1)
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

        # Track usage (best-effort)
        try:
            from engine.usage_tracker import get_tracker
            get_tracker(self.config).log_video_generation(
                duration_s=scene.duration_s,
                resolution=f"{gen_width}x{gen_height}",
                fps=clip_fps,
                pipeline_type="animatediff",
            )
        except Exception:
            logger.debug("Usage tracking failed for video generation", exc_info=True)

        return {
            "frames": all_frames,
            "video_path": scene_video,
            "seed": seed,
            "safety_passed": True,
            "safety_result": safety_result,
        }

    # Manual circuit breaker: open after 3 consecutive failures, reset after 60s
    class ManualCircuitBreaker:
        def __init__(self, fail_max=3, reset_timeout=60):
            self.fail_max = fail_max
            self.reset_timeout = reset_timeout
            self.failure_count = 0
            self.lock = threading.Lock()
            self.opened_at = None
        def __call__(self, func):
            def wrapper(*args, **kwargs):
                with self.lock:
                    if self.opened_at and (time.time() - self.opened_at < self.reset_timeout):
                        raise RuntimeError("Circuit breaker is open")
                    if self.opened_at and (time.time() - self.opened_at >= self.reset_timeout):
                        self.failure_count = 0
                        self.opened_at = None
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    with self.lock:
                        self.failure_count += 1
                        if self.failure_count >= self.fail_max:
                            self.opened_at = time.time()
                            logger.warning(f"Manual circuit breaker tripped for {func.__name__}")
                    raise
                return result
            return wrapper
    _scene_circuit_breaker = ManualCircuitBreaker(fail_max=3, reset_timeout=60)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @_scene_circuit_breaker
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
        Generate a scene with automatic retry on safety failure, with retry/circuit breaker.
        Mutates seed on each retry.
        """
        last_result = None
        for attempt in range(max_attempts):
            seed = torch.randint(0, 2**32, (1,)).item()
            logger.info(
                "Scene %d — attempt %d/%d (seed=%d)",
                scene.scene_id, attempt + 1, max_attempts, seed,
            )
            try:
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
                last_result = result
                if result.get("safety_passed", False):
                    return result
                logger.warning("Attempt %d failed safety — retrying with new seed...", attempt + 1)
            except Exception as e:
                logger.warning(f"Scene generation failed (attempt %d/%d): %s", attempt + 1, max_attempts, e)
                last_result = {"error": str(e)}
        logger.error(
            "Scene %d failed all %d safety attempts.",
            scene.scene_id, max_attempts,
        )
        return last_result  # Return last attempt even if failed

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
                gen_width=effective_width, fps=effective_fps,
                guidance_scale=effective_guidance,
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
        fps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        lora_strength: Optional[float] = None,
    ) -> str:
        """Generate a cache key from scene content + all generation parameters."""
        parts = [
            scene.narration,
            scene.visual_description,
            scene.emotion_tone,
            scene.setting,
            str(scene.duration_s),
            str(character_name),
            str(style if style is not None else self.style),
            str(camera_motion if camera_motion is not None else self.camera_motion),
            str(num_inference_steps if num_inference_steps is not None else self.num_inference_steps),
            str(gen_height if gen_height is not None else self.gen_height),
            str(gen_width if gen_width is not None else self.gen_width),
            str(fps if fps is not None else self.fps),
            str(guidance_scale if guidance_scale is not None else self.guidance_scale),
            str(lora_strength) if lora_strength is not None else "default",
            str(self.fast_mode),
            self.model_version,
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
            with open(cache_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if os.path.exists(data.get("video_path", "")):
                return data
            try:
                os.remove(cache_file)
            except OSError:
                pass
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
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(cache_data, f)
