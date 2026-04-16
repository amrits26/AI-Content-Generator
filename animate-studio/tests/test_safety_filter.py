"""Unit tests for SafetyFilter text scanning and compliance checks (mocked models)."""
import os
import sys
from unittest.mock import MagicMock, patch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from engine.safety_filter import SafetyFilter, SafetyResult


@pytest.fixture
def config_path(tmp_path):
    cfg = {
        "app": {
            "output_dir": str(tmp_path / "output"),
            "safety_audit_log": str(tmp_path / "safety_audit.log"),
        },
        "models": {
            "safety": {
                "clip_model": "openai/clip-vit-large-patch14",
                "nsfk_model": "yasserrmd/nsfk-detection",
                "blocked_concepts": ["gore", "horror", "nudity", "violence"],
                "safety_threshold": 0.25,
                "max_regeneration_attempts": 3,
            }
        },
        "export": {
            "youtube": {
                "min_duration_s": 60,
                "max_duration_s": 90,
            },
            "facebook_reels": {
                "min_duration_s": 15,
                "max_duration_s": 60,
            },
        },
        "compliance": {
            "require_safety_pass": True,
            "require_human_input_log": True,
            "require_ai_disclosure": True,
            "require_royalty_free_audio": True,
            "ai_disclosure_text": "AI generated",
        },
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.dump(cfg), encoding="utf-8")
    return str(path)


@pytest.fixture
def sf(config_path):
    return SafetyFilter(config_path)


# ── Text scanning (mocked NSFK pipeline) ─────────────────

def test_scan_text_safe(sf):
    mock_pipeline = MagicMock(return_value=[{"label": "sfw", "score": 0.98}])
    sf._nsfk_pipeline = mock_pipeline
    result = sf.scan_text("Billy Bunny shares his carrots with friends in the meadow.")
    assert result.passed is True
    assert result.scan_type == "text"


def test_scan_text_unsafe(sf):
    mock_pipeline = MagicMock(return_value=[{"label": "nsfw", "score": 0.95}])
    sf._nsfk_pipeline = mock_pipeline
    result = sf.scan_text("Violent scary content with weapons.")
    assert result.passed is False
    assert len(result.flagged_concepts) > 0


def test_scan_empty_text(sf):
    result = sf.scan_text("")
    assert result.passed is True


def test_scan_text_borderline_safe(sf):
    """Low-confidence unsafe label should still pass (below 0.6 threshold)."""
    mock_pipeline = MagicMock(return_value=[{"label": "nsfw", "score": 0.3}])
    sf._nsfk_pipeline = mock_pipeline
    result = sf.scan_text("Mildly ambiguous text.")
    assert result.passed is True


# ── Compliance checklist ──────────────────────────────────

def test_compliance_all_pass(sf, tmp_path):
    # Create a fake thumbnail
    thumb = tmp_path / "thumb.png"
    from PIL import Image
    Image.new("RGB", (100, 100), "white").save(str(thumb))

    # Mock the frame scan for thumbnail check
    sf.scan_frame = MagicMock(return_value=SafetyResult(passed=True, scan_type="visual"))

    safety = SafetyResult(passed=True, scan_type="combined")
    checks = sf.run_compliance_checklist(
        safety_result=safety,
        human_input_logged=True,
        audio_royalty_free=True,
        duration_s=70.0,
        platform="youtube",
        thumbnail_path=str(thumb),
    )
    assert checks["_all_passed"] is True
    assert checks["duration_in_range"] is True
    assert checks["safety_filter_passed"] is True


def test_compliance_duration_too_short(sf):
    safety = SafetyResult(passed=True, scan_type="combined")
    checks = sf.run_compliance_checklist(
        safety_result=safety,
        human_input_logged=True,
        audio_royalty_free=True,
        duration_s=5.0,
        platform="youtube",
    )
    assert checks["duration_in_range"] is False
    assert checks["_all_passed"] is False


def test_compliance_duration_too_long(sf):
    safety = SafetyResult(passed=True, scan_type="combined")
    checks = sf.run_compliance_checklist(
        safety_result=safety,
        human_input_logged=True,
        audio_royalty_free=True,
        duration_s=120.0,
        platform="youtube",
    )
    assert checks["duration_in_range"] is False


def test_compliance_no_human_input(sf):
    safety = SafetyResult(passed=True, scan_type="combined")
    checks = sf.run_compliance_checklist(
        safety_result=safety,
        human_input_logged=False,
        audio_royalty_free=True,
        duration_s=70.0,
        platform="youtube",
    )
    assert checks["human_creative_input_logged"] is False
    assert checks["_all_passed"] is False


def test_compliance_safety_failed(sf):
    safety = SafetyResult(passed=False, scan_type="combined", flagged_concepts=["violence"])
    checks = sf.run_compliance_checklist(
        safety_result=safety,
        human_input_logged=True,
        audio_royalty_free=True,
        duration_s=70.0,
        platform="youtube",
    )
    assert checks["safety_filter_passed"] is False
    assert checks["_all_passed"] is False


def test_compliance_facebook_reels_duration(sf):
    safety = SafetyResult(passed=True, scan_type="combined")
    checks = sf.run_compliance_checklist(
        safety_result=safety,
        human_input_logged=True,
        audio_royalty_free=True,
        duration_s=30.0,
        platform="facebook_reels",
    )
    assert checks["duration_in_range"] is True
