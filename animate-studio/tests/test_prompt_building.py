"""Unit tests for prompt template functions."""
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from utils.prompt_templates import (
    get_style_preset,
    get_camera_motion,
    build_character_description,
    build_scene_prompt,
    get_negative_prompt,
    STYLE_PRESETS,
    CAMERA_MOTIONS,
    EMOTION_TONES,
    SCENE_SETTINGS,
)


# ── Style presets ─────────────────────────────────────────

def test_get_known_style_preset():
    p = get_style_preset("pixar_cute")
    assert "prefix" in p
    assert "negative" in p
    assert "Pixar" in p["prefix"]


def test_get_animatediff_style_preset():
    p = get_style_preset("animatediff_cartoon")
    assert "cartoon" in p["prefix"].lower()
    # Should be compact — under 77 tokens (rough heuristic: < 200 chars)
    assert len(p["prefix"]) < 200


def test_unknown_style_falls_back_to_pixar():
    p = get_style_preset("nonexistent_style")
    assert p == STYLE_PRESETS["pixar_cute"]


# ── Camera motion ─────────────────────────────────────────

def test_get_known_camera_motion():
    m = get_camera_motion("slow_zoom_in")
    assert "zoom" in m.lower()


def test_get_static_camera():
    m = get_camera_motion("static")
    assert "static" in m.lower()


def test_unknown_camera_returns_empty():
    m = get_camera_motion("flying_backwards_spin")
    assert m == ""


# ── Character descriptions ────────────────────────────────

def test_build_bunny_description():
    desc = build_character_description("bunny", "Billy", "soft blue", "red bowtie")
    assert "Billy" in desc
    assert "blue" in desc
    assert "bowtie" in desc
    assert "bunny" in desc.lower()


def test_build_duckling_description():
    desc = build_character_description("duckling", "Daisy", "yellow", "pink ribbon")
    assert "Daisy" in desc
    assert "duckling" in desc.lower()


def test_unknown_animal_falls_back_to_bunny():
    desc = build_character_description("dragon", "Draco", "green", "scarf")
    # Should still produce a description using bunny template
    assert "Draco" in desc


# ── Scene prompts ─────────────────────────────────────────

def test_build_scene_prompt_contains_style():
    prompt = build_scene_prompt(
        character_desc="A fluffy blue bunny",
        action="sharing a carrot",
        emotion="happy",
        setting="meadow",
    )
    assert "Pixar" in prompt or "3D" in prompt


def test_build_scene_prompt_contains_setting():
    prompt = build_scene_prompt(
        character_desc="A kitten",
        action="playing",
        emotion="excited",
        setting="playground",
    )
    assert "playground" in prompt.lower()


def test_build_scene_prompt_auto_camera():
    prompt = build_scene_prompt(
        character_desc="A duckling",
        action="exploring",
        emotion="curious",
        setting="forest",
        camera_motion="auto",
    )
    # curious → gentle_pan_right
    assert "pan" in prompt.lower()


def test_negative_prompt_not_empty():
    neg = get_negative_prompt("pixar_cute")
    assert len(neg) > 50
    assert "nsfw" in neg.lower()


# ── Data completeness ────────────────────────────────────

def test_all_emotions_have_tones():
    for key in ["happy", "sad", "excited", "curious", "brave", "shy", "surprised", "loving", "sleepy", "proud"]:
        assert key in EMOTION_TONES, f"Missing emotion: {key}"


def test_all_settings_defined():
    for key in ["meadow", "forest", "bedroom", "kitchen", "playground", "beach", "classroom", "garden"]:
        assert key in SCENE_SETTINGS, f"Missing setting: {key}"
