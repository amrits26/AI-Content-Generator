"""
═══════════════════════════════════════════════════════════════
AniMate Studio — Multi-Platform Exporter
═══════════════════════════════════════════════════════════════
Handles final export to YouTube, Facebook Reels, and TikTok
with platform-specific encoding, thumbnails, captions, and
monetization compliance metadata.
═══════════════════════════════════════════════════════════════
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import yaml

from engine.safety_filter import SafetyFilter, SafetyResult
from utils.ffmpeg_utils import (
    apply_final_encode,
    crop_aspect_ratio,
    extract_best_thumbnail,
    get_media_duration,
    run_ffmpeg,
)
from utils.upscaler import Upscaler

logger = logging.getLogger("animate_studio.exporter")


@dataclass
class ExportResult:
    """Result of an export operation."""
    success: bool
    platform: str
    video_path: str = ""
    thumbnail_path: str = ""
    caption_path: str = ""
    metadata_path: str = ""
    duration_s: float = 0.0
    resolution: str = ""
    compliance: dict = field(default_factory=dict)
    error: str = ""
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class Exporter:
    """
    Multi-platform video exporter with compliance checks.

    Supports:
      - youtube: 16:9, 60-90s, H.264, high bitrate
      - facebook_reels: 9:16, 15-60s, H.264
      - tiktok: 9:16, 15-60s, H.264
    """

    PLATFORMS = {"youtube", "facebook_reels", "tiktok"}

    def __init__(self, config_path: str = "config.yaml"):
        self._config_path = config_path
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.output_dir = self.config["app"]["output_dir"]
        self.export_presets = self.config["export"]
        self.compliance_cfg = self.config["compliance"]
        self.ai_disclosure = self.compliance_cfg["ai_disclosure_text"]

        # Upscaling & color grading settings
        self.upscale_enabled = self.export_presets.get("upscale", False)
        self.upscale_factor = self.export_presets.get("upscale_factor", 2)
        self.upscaler_model = self.export_presets.get("upscaler_model", "realesrgan-x4plus-anime")
        self.color_grading = self.export_presets.get("color_grading", {})
        self.burn_captions = self.export_presets.get("burn_captions", False)

        self._upscaler: Optional[Upscaler] = None

        os.makedirs(self.output_dir, exist_ok=True)

    def export(
        self,
        source_video: str,
        platform: str,
        title: str,
        narration_texts: list[str],
        safety_result: SafetyResult,
        human_input_logged: bool = True,
        audio_royalty_free: bool = True,
        hashtags: Optional[list[str]] = None,
    ) -> ExportResult:
        """
        Full export pipeline for a single platform:
        1. Compliance check
        2. Aspect ratio crop/scale
        3. Final encode with platform settings
        4. Thumbnail extraction + safety scan
        5. Caption (.srt) generation
        6. Metadata file with AI disclosure
        """
        if platform not in self.PLATFORMS:
            return ExportResult(
                success=False,
                platform=platform,
                error=f"Unknown platform: {platform}. Must be one of {self.PLATFORMS}",
            )

        preset = self.export_presets[platform]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = self._sanitize_filename(title)
        export_dir = os.path.join(self.output_dir, f"{safe_title}_{platform}_{timestamp}")
        os.makedirs(export_dir, exist_ok=True)

        try:
            # ── Step 1: Check source video duration ──────
            source_duration = get_media_duration(source_video)
            logger.info(
                "Exporting for %s — source: %.1fs, target: %d-%ds",
                platform, source_duration,
                preset["min_duration_s"], preset["max_duration_s"],
            )

            # ── Step 2: Thumbnail (before encoding to save time) ─
            thumbnail_path = os.path.join(export_dir, f"{safe_title}_thumbnail.png")
            extract_best_thumbnail(source_video, thumbnail_path)
            logger.info("Thumbnail extracted: %s", thumbnail_path)

            # ── Step 3: Compliance check ─────────────────
            safety_filter = SafetyFilter(config_path=self._config_path)
            compliance = safety_filter.run_compliance_checklist(
                safety_result=safety_result,
                human_input_logged=human_input_logged,
                audio_royalty_free=audio_royalty_free,
                duration_s=source_duration,
                platform=platform,
                thumbnail_path=thumbnail_path,
            )

            if self.compliance_cfg["require_safety_pass"] and not compliance.get("_all_passed"):
                failed_checks = [k for k, v in compliance.items() if not v and k != "_all_passed"]
                return ExportResult(
                    success=False,
                    platform=platform,
                    compliance=compliance,
                    error=f"Compliance checks failed: {failed_checks}",
                )

            # ── Step 4: Crop to target aspect ratio ──────
            cropped_path = os.path.join(export_dir, "_cropped.mp4")
            crop_aspect_ratio(
                source_video,
                cropped_path,
                target_width=preset["width"],
                target_height=preset["height"],
            )

            # ── Step 4b: AI Upscale (if enabled) ─────────
            upscale_input = cropped_path
            if self.upscale_enabled:
                upscaled_path = os.path.join(export_dir, "_upscaled.mp4")
                try:
                    if self._upscaler is None:
                        self._upscaler = Upscaler(
                            model_name=self.upscaler_model,
                            scale=self.upscale_factor,
                        )
                    self._upscaler.upscale_video_ffmpeg(
                        cropped_path, upscaled_path,
                        target_width=preset["width"],
                        target_height=preset["height"],
                    )
                    upscale_input = upscaled_path
                    logger.info("Upscaled to %dx%d", preset["width"], preset["height"])
                except Exception as e:
                    logger.warning("Upscale failed, using cropped: %s", e)

            # ── Step 4c: Color grading (FFmpeg filter) ───
            grading_input = upscale_input
            grading_cfg = self.color_grading.get(platform, {})
            grading_filter = grading_cfg.get("filter", "")
            if grading_filter:
                graded_path = os.path.join(export_dir, "_graded.mp4")
                try:
                    run_ffmpeg(
                        [
                            "-i", grading_input,
                            "-vf", grading_filter,
                            "-c:v", "libx264",
                            "-pix_fmt", "yuv420p",
                            "-crf", "18",
                            "-c:a", "copy",
                            graded_path,
                        ],
                        description=f"color_grade_{platform}",
                    )
                    grading_input = graded_path
                    logger.info("Color grading applied for %s", platform)
                except Exception as e:
                    logger.warning("Color grading failed, skipping: %s", e)

            # ── Step 5: Final encode ─────────────────────
            video_filename = f"{safe_title}_{platform}.mp4"
            final_video_path = os.path.join(export_dir, video_filename)

            apply_final_encode(
                input_path=grading_input,
                output_path=final_video_path,
                width=preset["width"],
                height=preset["height"],
                video_codec=preset["video_codec"],
                video_bitrate=preset["video_bitrate"],
                audio_codec=preset["audio_codec"],
                audio_bitrate=preset["audio_bitrate"],
                crf=preset["crf"],
                preset=preset["preset"],
                pixel_format=preset["pixel_format"],
            )

            # Clean up intermediate files
            for tmp in [cropped_path, upscale_input, grading_input]:
                if tmp != final_video_path and os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

            final_duration = get_media_duration(final_video_path)

            # ── Step 6: Generate captions (.srt) ─────────
            caption_path = os.path.join(export_dir, f"{safe_title}.srt")
            self._generate_srt(narration_texts, final_duration, caption_path)

            # ── Step 7: Metadata with AI disclosure ──────
            metadata_path = os.path.join(export_dir, f"{safe_title}_metadata.json")
            metadata = self._generate_metadata(
                title=title,
                platform=platform,
                duration_s=final_duration,
                resolution=f"{preset['width']}x{preset['height']}",
                hashtags=hashtags or self._default_hashtags(platform),
                narration_texts=narration_texts,
                compliance=compliance,
            )
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(
                "Export complete: %s [%s] %.1fs %dx%d",
                final_video_path, platform, final_duration,
                preset["width"], preset["height"],
            )

            return ExportResult(
                success=True,
                platform=platform,
                video_path=final_video_path,
                thumbnail_path=thumbnail_path,
                caption_path=caption_path,
                metadata_path=metadata_path,
                duration_s=final_duration,
                resolution=f"{preset['width']}x{preset['height']}",
                compliance=compliance,
            )

        except Exception as e:
            logger.exception("Export failed for %s: %s", platform, e)
            return ExportResult(
                success=False,
                platform=platform,
                error=str(e),
            )

    def export_all_platforms(
        self,
        source_video: str,
        title: str,
        narration_texts: list[str],
        safety_result: SafetyResult,
        human_input_logged: bool = True,
        audio_royalty_free: bool = True,
    ) -> dict[str, ExportResult]:
        """Export for all three platforms in sequence."""
        results = {}
        for platform in self.PLATFORMS:
            results[platform] = self.export(
                source_video=source_video,
                platform=platform,
                title=title,
                narration_texts=narration_texts,
                safety_result=safety_result,
                human_input_logged=human_input_logged,
                audio_royalty_free=audio_royalty_free,
            )
        return results

    # ── Caption Generation ───────────────────────────────
    def _generate_srt(
        self,
        narration_texts: list[str],
        total_duration: float,
        output_path: str,
    ):
        """
        Generate SRT caption file from narration texts.
        Distributes timing evenly across scenes.
        """
        if not narration_texts:
            # Write empty SRT
            with open(output_path, "w", encoding="utf-8") as f:
                f.write("")
            return

        scene_duration = total_duration / len(narration_texts)
        lines = []

        for i, text in enumerate(narration_texts):
            start_s = i * scene_duration
            end_s = start_s + scene_duration

            start_ts = self._seconds_to_srt_time(start_s)
            end_ts = self._seconds_to_srt_time(end_s)

            lines.append(str(i + 1))
            lines.append(f"{start_ts} --> {end_ts}")
            lines.append(text.strip())
            lines.append("")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Captions saved: %s (%d entries)", output_path, len(narration_texts))

    @staticmethod
    def _seconds_to_srt_time(seconds: float) -> str:
        """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

    # ── Metadata Generation ──────────────────────────────
    def _generate_metadata(
        self,
        title: str,
        platform: str,
        duration_s: float,
        resolution: str,
        hashtags: list[str],
        narration_texts: list[str],
        compliance: dict,
    ) -> dict:
        """Generate metadata JSON for platform upload."""
        description_parts = [
            f"{title}\n",
            self.ai_disclosure,
            "",
            "---",
            f"Created with AniMate Studio | {platform.replace('_', ' ').title()}",
            f"Duration: {duration_s:.1f}s | Resolution: {resolution}",
            "",
            " ".join(hashtags),
        ]

        return {
            "title": title,
            "description": "\n".join(description_parts),
            "platform": platform,
            "duration_seconds": round(duration_s, 2),
            "resolution": resolution,
            "hashtags": hashtags,
            "ai_disclosure": self.ai_disclosure,
            "content_rating": "kids",
            "age_range": "2-8",
            "scenes": len(narration_texts),
            "narration": narration_texts,
            "compliance_checks": compliance,
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "tool": "AniMate Studio v1.0",
        }

    def _default_hashtags(self, platform: str) -> list[str]:
        """Platform-specific default hashtags for kids' animation content."""
        base = [
            "#KidsAnimation", "#AnimatedStory", "#ChildrenContent",
            "#BedtimeStory", "#AIAnimation", "#KidsCartoon",
        ]
        platform_tags = {
            "youtube": ["#YouTubeKids", "#AnimatedEpisode", "#KidsYouTube"],
            "facebook_reels": ["#FacebookReels", "#ReelsAnimation", "#KidsReels"],
            "tiktok": ["#TikTokKids", "#AnimationTikTok", "#AIGenerated"],
        }
        return base + platform_tags.get(platform, [])

    # ── Utility ──────────────────────────────────────────
    @staticmethod
    def _sanitize_filename(name: str) -> str:
        """Remove characters unsafe for filenames."""
        keepchars = (" ", "_", "-")
        cleaned = "".join(c for c in name if c.isalnum() or c in keepchars).strip()
        return cleaned.replace(" ", "_")[:80]

    def _find_config_path(self) -> str:
        """Locate config.yaml relative to this module."""
        # Try several locations
        candidates = [
            "config.yaml",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml"),
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return "config.yaml"
