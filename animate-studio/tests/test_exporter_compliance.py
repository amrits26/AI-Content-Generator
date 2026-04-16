"""Unit tests for Exporter compliance and metadata generation."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from engine.exporter import Exporter, ExportResult


@pytest.fixture
def config_path(tmp_path):
    cfg = {
        "app": {
            "output_dir": str(tmp_path / "output"),
            "safety_audit_log": str(tmp_path / "audit.log"),
            "assets_dir": str(tmp_path / "assets"),
            "loras_dir": str(tmp_path / "loras"),
        },
        "models": {
            "safety": {
                "clip_model": "openai/clip-vit-large-patch14",
                "nsfk_model": "yasserrmd/nsfk-detection",
                "blocked_concepts": ["gore"],
                "safety_threshold": 0.25,
                "max_regeneration_attempts": 3,
            }
        },
        "export": {
            "upscale": False,
            "upscale_factor": 2,
            "upscaler_model": "realesrgan-x4plus-anime",
            "color_grading": {},
            "burn_captions": False,
            "youtube": {
                "width": 1920,
                "height": 1080,
                "aspect": "16:9",
                "min_duration_s": 60,
                "max_duration_s": 90,
                "video_codec": "libx264",
                "video_bitrate": "8M",
                "audio_codec": "aac",
                "audio_bitrate": "192k",
                "crf": 18,
                "preset": "slow",
                "pixel_format": "yuv420p",
            },
            "facebook_reels": {
                "width": 1080,
                "height": 1920,
                "aspect": "9:16",
                "min_duration_s": 15,
                "max_duration_s": 60,
                "video_codec": "libx264",
                "video_bitrate": "6M",
                "audio_codec": "aac",
                "audio_bitrate": "192k",
                "crf": 20,
                "preset": "medium",
                "pixel_format": "yuv420p",
            },
            "tiktok": {
                "width": 1080,
                "height": 1920,
                "aspect": "9:16",
                "min_duration_s": 15,
                "max_duration_s": 60,
                "video_codec": "libx264",
                "video_bitrate": "6M",
                "audio_codec": "aac",
                "audio_bitrate": "192k",
                "crf": 20,
                "preset": "medium",
                "pixel_format": "yuv420p",
            },
        },
        "compliance": {
            "require_safety_pass": True,
            "require_human_input_log": True,
            "require_ai_disclosure": True,
            "require_royalty_free_audio": True,
            "ai_disclosure_text": "AI generated content.",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return str(path)


@pytest.fixture
def exp(config_path):
    return Exporter(config_path)


def test_unknown_platform_returns_error(exp):
    from engine.safety_filter import SafetyResult
    result = exp.export(
        source_video="fake.mp4",
        platform="snapchat",
        title="Test",
        narration_texts=[],
        safety_result=SafetyResult(passed=True, scan_type="text"),
    )
    assert result.success is False
    assert "snapchat" in result.error.lower()


def test_export_result_has_timestamp():
    r = ExportResult(success=True, platform="youtube")
    assert r.timestamp != ""
    assert "T" in r.timestamp  # ISO format


def test_valid_platforms_accepted(exp):
    assert "youtube" in exp.PLATFORMS
    assert "facebook_reels" in exp.PLATFORMS
    assert "tiktok" in exp.PLATFORMS


def test_export_presets_loaded(exp):
    assert exp.export_presets["youtube"]["width"] == 1920
    assert exp.export_presets["facebook_reels"]["height"] == 1920
