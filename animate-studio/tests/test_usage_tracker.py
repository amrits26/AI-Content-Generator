"""Tests for engine.usage_tracker — SQLite usage logging and cost estimation."""

import json
import os
import tempfile

import pytest

from engine.usage_tracker import UsageTracker, get_tracker


@pytest.fixture()
def tracker(tmp_path):
    """Return a UsageTracker backed by a temporary database."""
    db = str(tmp_path / "test_usage.db")
    return UsageTracker(db_path=db)


@pytest.fixture()
def tracker_with_rates(tmp_path):
    """Return a tracker with custom cost rates."""
    db = str(tmp_path / "test_usage.db")
    config = {
        "usage": {
            "cost_rates": {
                "llm_per_1k_tokens_cents": 10.0,
                "tts_per_1k_chars_cents": 50.0,
                "video_per_minute_cents": 5.0,
            }
        }
    }
    return UsageTracker(db_path=db, config=config)


# ── Basic logging ──────────────────────────────────────


def test_log_llm_call_creates_row(tracker):
    tracker.log_llm_call(
        provider="ollama", model="llama3.2",
        prompt_tokens=100, completion_tokens=50, total_tokens=150,
    )
    rows = tracker.get_recent(limit=1)
    assert len(rows) == 1
    assert rows[0]["operation"] == "llm_call"
    assert rows[0]["tokens_used"] == 150
    assert rows[0]["provider"] == "ollama"


def test_log_tts_call_creates_row(tracker):
    tracker.log_tts_call(provider="elevenlabs", characters=500, voice_id="abc123")
    rows = tracker.get_recent(limit=1)
    assert len(rows) == 1
    assert rows[0]["operation"] == "tts_call"
    assert rows[0]["characters_generated"] == 500


def test_log_video_generation_creates_row(tracker):
    tracker.log_video_generation(
        duration_s=12.5, resolution="512x512", fps=8, pipeline_type="animatediff",
    )
    rows = tracker.get_recent(limit=1)
    assert len(rows) == 1
    assert rows[0]["operation"] == "video_generation"
    assert rows[0]["video_duration_seconds"] == 12.5


# ── Summary and queries ───────────────────────────────


def test_get_summary_returns_totals(tracker):
    tracker.log_llm_call("openai", "gpt-4o-mini", 200, 100, 300)
    tracker.log_llm_call("openai", "gpt-4o-mini", 150, 80, 230)
    tracker.log_tts_call("elevenlabs", 1000, "v1")

    summary = tracker.get_summary()
    assert "llm_call" in summary["operations"]
    assert summary["operations"]["llm_call"]["count"] == 2
    assert summary["operations"]["llm_call"]["total_tokens"] == 530
    assert "tts_call" in summary["operations"]
    assert summary["operations"]["tts_call"]["count"] == 1
    assert summary["total_cost_cents"] > 0


def test_get_recent_respects_limit(tracker):
    for i in range(10):
        tracker.log_llm_call("ollama", "llama3.2", 10, 10, 20)
    rows = tracker.get_recent(limit=5)
    assert len(rows) == 5


# ── Cost estimation with custom rates ─────────────────


def test_cost_estimation_uses_config_rates(tracker_with_rates):
    # 1000 tokens at 10¢ per 1k → 10 cents
    tracker_with_rates.log_llm_call("openai", "gpt-4o", 500, 500, 1000)
    rows = tracker_with_rates.get_recent(limit=1)
    assert rows[0]["estimated_cost_cents"] == pytest.approx(10.0, abs=0.01)

    # 1000 chars at 50¢ per 1k → 50 cents
    tracker_with_rates.log_tts_call("elevenlabs", 1000, "v1")
    rows = tracker_with_rates.get_recent(limit=1)
    assert rows[0]["estimated_cost_cents"] == pytest.approx(50.0, abs=0.01)

    # 60s video at 5¢ per minute → 5 cents
    tracker_with_rates.log_video_generation(60.0, "512x512", 8, "legacy")
    rows = tracker_with_rates.get_recent(limit=1)
    assert rows[0]["estimated_cost_cents"] == pytest.approx(5.0, abs=0.01)


# ── Singleton ─────────────────────────────────────────


def test_tracker_singleton_returns_same_instance():
    import engine.usage_tracker as mod
    # Reset module-level state for isolation
    mod._tracker = None
    try:
        t1 = get_tracker()
        t2 = get_tracker()
        assert t1 is t2
    finally:
        mod._tracker = None


# ── Metadata stored correctly ─────────────────────────


def test_llm_metadata_contains_model(tracker):
    tracker.log_llm_call("openai", "gpt-4o-mini", 100, 50, 150)
    rows = tracker.get_recent(limit=1)
    meta = json.loads(rows[0]["metadata"])
    assert meta["model"] == "gpt-4o-mini"
    assert meta["prompt_tokens"] == 100
    assert meta["completion_tokens"] == 50
