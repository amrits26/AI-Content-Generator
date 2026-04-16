"""
═══════════════════════════════════════════════════════════════
AniMate Studio — AI Upscaler (Real-ESRGAN)
═══════════════════════════════════════════════════════════════
Upscales generated video frames using Real-ESRGAN for crisp
1080p+ output from lower-resolution diffusion models.

Falls back to Lanczos bicubic if Real-ESRGAN is unavailable.
═══════════════════════════════════════════════════════════════
"""

import logging
import os
import subprocess
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

logger = logging.getLogger("animate_studio.upscaler")

_REALESRGAN_AVAILABLE = False
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    _REALESRGAN_AVAILABLE = True
except ImportError:
    logger.info("Real-ESRGAN not installed — will use Lanczos fallback.")


class Upscaler:
    """
    AI-powered frame upscaler using Real-ESRGAN.
    Falls back to Lanczos (PIL) when the library is unavailable.
    """

    def __init__(
        self,
        model_name: str = "realesrgan-x4plus-anime",
        scale: int = 2,
        device: str = "cuda",
        half: bool = True,
    ):
        self.model_name = model_name
        self.scale = scale
        self.device = device
        self.half = half
        self._upsampler: Optional[object] = None

    def _load_model(self):
        """Lazy-load the Real-ESRGAN model."""
        if self._upsampler is not None:
            return
        if not _REALESRGAN_AVAILABLE:
            logger.warning("Real-ESRGAN not available; upscale_frame will use Lanczos.")
            return

        # Select architecture based on model name
        if "x4plus-anime" in self.model_name:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4,
            )
            netscale = 4
        elif "x4plus" in self.model_name:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=23, num_grow_ch=32, scale=4,
            )
            netscale = 4
        else:
            model = RRDBNet(
                num_in_ch=3, num_out_ch=3, num_feat=64,
                num_block=6, num_grow_ch=32, scale=4,
            )
            netscale = 4

        self._upsampler = RealESRGANer(
            scale=netscale,
            model_path=None,  # auto-download
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=self.half,
            device=self.device,
        )
        logger.info("Real-ESRGAN model loaded: %s (scale=%d)", self.model_name, netscale)

    def upscale_frame(self, image: Image.Image) -> Image.Image:
        """
        Upscale a single PIL Image.

        Returns upscaled image at self.scale × original resolution.
        """
        target_w = image.width * self.scale
        target_h = image.height * self.scale

        if _REALESRGAN_AVAILABLE:
            try:
                self._load_model()
                if self._upsampler is not None:
                    img_np = np.array(image)
                    output, _ = self._upsampler.enhance(img_np, outscale=self.scale)
                    return Image.fromarray(output)
            except Exception as e:
                logger.warning("Real-ESRGAN failed, falling back to Lanczos: %s", e)

        # Fallback: high-quality Lanczos resize
        return image.resize((target_w, target_h), Image.LANCZOS)

    def upscale_video_ffmpeg(
        self,
        input_path: str,
        output_path: str,
        target_width: int = 1920,
        target_height: int = 1080,
    ) -> str:
        """
        Upscale a video file using FFmpeg's Lanczos scaler.
        Use this as a fast alternative when Real-ESRGAN is too slow
        for full video processing.
        """
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "warning",
            "-i", input_path,
            "-vf", f"scale={target_width}:{target_height}:flags=lanczos",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-c:a", "copy",
            output_path,
        ]
        logger.info("FFmpeg upscale: %dx%d → %s", target_width, target_height, output_path)
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return output_path

    def unload(self):
        """Free GPU memory by unloading the model."""
        if self._upsampler is not None:
            del self._upsampler
            self._upsampler = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
            logger.info("Upscaler model unloaded.")
