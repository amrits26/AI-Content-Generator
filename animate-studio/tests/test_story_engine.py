"""Unit tests for StoryEngine._parse_response() and StoryParseError."""
import json
import os
import sys
import pytest

# Ensure project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from engine.story_engine import StoryEngine, StoryParseError


@pytest.fixture
def engine(tmp_path):
    """Create a StoryEngine with a minimal config file."""
    cfg = {
        "models": {
            "story": {
                "provider": "ollama",
                "model_name": "llama3.2",
                "openai_base_url": "http://localhost:11434/v1",
                "openai_api_key": "",
                "temperature": 0.8,
                "max_tokens": 2048,
            }
        }
    }
    cfg_path = tmp_path / "config.yaml"
    import yaml
    cfg_path.write_text(yaml.dump(cfg), encoding="utf-8")
    return StoryEngine(str(cfg_path))


# ── Valid JSON ────────────────────────────────────────────

def test_parse_valid_json(engine):
    raw = json.dumps({
        "title": "Test Story",
        "moral": "Be kind.",
        "scenes": [
            {
                "scene_id": 1,
                "narration": "Once upon a time...",
                "visual_description": "A sunny meadow",
                "emotion_tone": "happy",
                "setting": "meadow",
            }
        ],
    })
    result = engine._parse_response(raw)
    assert result["title"] == "Test Story"
    assert len(result["scenes"]) == 1


def test_parse_json_in_markdown_fence(engine):
    raw = '```json\n{"title": "Fenced", "moral": "Test", "scenes": []}\n```'
    result = engine._parse_response(raw)
    assert result["title"] == "Fenced"


def test_parse_json_with_leading_text(engine):
    raw = 'Here is the story:\n\n{"title": "Embedded", "moral": "Yes", "scenes": [{"scene_id": 1, "narration": "Hi", "visual_description": "Test", "emotion_tone": "happy", "setting": "meadow"}]}'
    result = engine._parse_response(raw)
    assert result["title"] == "Embedded"
    assert len(result["scenes"]) == 1


# ── Invalid / unparseable ─────────────────────────────────

def test_parse_garbage_raises_story_parse_error(engine):
    with pytest.raises(StoryParseError):
        engine._parse_response("This is not JSON at all, just plain text rambling.")


def test_parse_empty_raises_story_parse_error(engine):
    with pytest.raises(StoryParseError):
        engine._parse_response("")


def test_parse_truncated_json_raises_story_parse_error(engine):
    with pytest.raises(StoryParseError):
        engine._parse_response('{"title": "Broken", "moral": "oops", "scenes": [')


# ── Edge cases ────────────────────────────────────────────

def test_parse_valid_json_fewer_scenes(engine):
    """LLM returns valid JSON but fewer scenes than requested — still parseable."""
    raw = json.dumps({
        "title": "Short",
        "moral": "Short moral.",
        "scenes": [
            {
                "scene_id": 1,
                "narration": "One scene only.",
                "visual_description": "A meadow",
                "emotion_tone": "happy",
                "setting": "meadow",
            }
        ],
    })
    result = engine._parse_response(raw)
    assert len(result["scenes"]) == 1  # Parser returns what it got


def test_parse_json_with_extra_keys(engine):
    """LLM adds unexpected keys — parser should still return valid dict."""
    raw = json.dumps({
        "title": "Extra",
        "moral": "Flexibility.",
        "author": "LLM",
        "scenes": [],
    })
    result = engine._parse_response(raw)
    assert result["title"] == "Extra"


def test_generate_storyboard_surfaces_story_parse_error(engine):
    engine._call_llm = lambda system_prompt, user_prompt: "not-json"

    with pytest.raises(StoryParseError):
        engine.generate_storyboard(
            theme="Billy Bunny learns to share",
            character_name="Billy",
            character_type="bunny",
            num_scenes=3,
            scene_duration=10.0,
        )
