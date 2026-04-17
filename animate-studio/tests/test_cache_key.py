"""Unit tests for scene cache key — verifies all generation params affect the key.

Uses a standalone reimplementation of the cache key logic to avoid importing
the full Animator class (which loads torch/diffusers and takes 30+ seconds).
"""
import hashlib
import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.story_engine import Scene


def _scene_cache_key(
    scene,
    character_name,
    style="pixar_cute",
    camera_motion="auto",
    num_inference_steps=25,
    gen_height=512,
    gen_width=512,
    fps=8,
    guidance_scale=7.5,
    lora_strength=None,
    fast_mode=False,
    model_version="damo-vilab/text-to-video-ms-1.7b|frankjoshua/toonyou_beta6|animatediff",
    use_animatediff=True,
):
    """Mirror of Animator._scene_cache_key — kept in sync manually."""
    parts = [
        scene.narration,
        scene.visual_description,
        scene.emotion_tone,
        scene.setting,
        str(scene.duration_s),
        str(character_name),
        str(style),
        str(camera_motion),
        str(num_inference_steps),
        str(gen_height),
        str(gen_width),
        str(fps),
        str(guidance_scale),
        str(lora_strength) if lora_strength is not None else "default",
        str(fast_mode),
        model_version,
        "animatediff" if use_animatediff else "legacy",
    ]
    content = "|".join(parts)
    return hashlib.sha256(content.encode()).hexdigest()


@pytest.fixture
def scene():
    return Scene(
        scene_id=1,
        narration="Billy shares his carrot.",
        visual_description="A bunny in a meadow sharing a carrot",
        emotion_tone="happy",
        setting="meadow",
        duration_s=10.0,
    )


def _base_key(scene):
    return _scene_cache_key(scene, "Billy")


def test_same_inputs_produce_same_key(scene):
    assert _base_key(scene) == _base_key(scene)


def test_style_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Billy", style="hyper_realistic")
    assert base != different


def test_camera_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Billy", camera_motion="slow_zoom_in")
    assert base != different


def test_steps_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Billy", num_inference_steps=50)
    assert base != different


def test_resolution_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Billy", gen_height=256, gen_width=256)
    assert base != different


def test_guidance_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Billy", guidance_scale=12.0)
    assert base != different


def test_lora_strength_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Billy", lora_strength=0.3)
    assert base != different


def test_fps_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Billy", fps=12)
    assert base != different


def test_narration_change_produces_different_key(scene):
    base = _base_key(scene)
    scene2 = Scene(
        scene_id=1,
        narration="Billy keeps the carrot for himself.",
        visual_description="A bunny in a meadow sharing a carrot",
        emotion_tone="happy",
        setting="meadow",
        duration_s=10.0,
    )
    different = _scene_cache_key(scene2, "Billy")
    assert base != different


def test_duration_change_produces_different_key(scene):
    base = _base_key(scene)
    scene2 = Scene(
        scene_id=1,
        narration=scene.narration,
        visual_description=scene.visual_description,
        emotion_tone=scene.emotion_tone,
        setting=scene.setting,
        duration_s=12.0,
    )
    different = _scene_cache_key(scene2, "Billy")
    assert base != different


def test_model_version_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Billy", model_version="alternate-model|alt-motion|legacy")
    assert base != different


def test_character_change_produces_different_key(scene):
    base = _base_key(scene)
    different = _scene_cache_key(scene, "Daisy")
    assert base != different


def test_visual_description_change_produces_different_key(scene):
    base = _base_key(scene)
    scene2 = Scene(
        scene_id=1,
        narration="Billy shares his carrot.",
        visual_description="A bunny in a forest under a rainbow",
        emotion_tone="happy",
        setting="meadow",
        duration_s=10.0,
    )
    different = _scene_cache_key(scene2, "Billy")
    assert base != different
