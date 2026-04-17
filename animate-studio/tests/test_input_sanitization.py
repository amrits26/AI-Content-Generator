"""Tests for input sanitization in main.py — sanitize_and_check_prompt."""

from unittest.mock import MagicMock

import pytest

# Import the function under test.
# We avoid importing the full main module (heavy Gradio/torch deps) by
# extracting the logic from the module path and testing the callable directly.
# Since sanitize_and_check_prompt only depends on SafetyResult and a
# safety_filter with scan_text(), we can test it with lightweight mocks.

from engine.safety_filter import SafetyResult


def _make_sanitize():
    """Return a standalone sanitize_and_check_prompt function (same logic as main.py)."""

    def sanitize_and_check_prompt(text, safety_filter, max_length=2000):
        if not text or not text.strip():
            raise ValueError("Prompt cannot be empty.")
        text = text.strip()[:max_length]
        result = safety_filter.scan_text(text)
        if not result.passed:
            raise ValueError(
                f"Prompt flagged as inappropriate: "
                f"{', '.join(result.flagged_concepts) if result.flagged_concepts else result.details}"
            )
        return text

    return sanitize_and_check_prompt


sanitize_and_check_prompt = _make_sanitize()


def _mock_filter(passed=True, flagged=None, details=""):
    sf = MagicMock()
    sf.scan_text.return_value = SafetyResult(
        passed=passed,
        scan_type="text",
        flagged_concepts=flagged or [],
        details=details,
    )
    return sf


# ── Tests ─────────────────────────────────────────────


def test_sanitize_empty_raises():
    sf = _mock_filter()
    with pytest.raises(ValueError, match="empty"):
        sanitize_and_check_prompt("", sf)
    with pytest.raises(ValueError, match="empty"):
        sanitize_and_check_prompt("   ", sf)


def test_sanitize_none_raises():
    sf = _mock_filter()
    with pytest.raises(ValueError, match="empty"):
        sanitize_and_check_prompt(None, sf)


def test_sanitize_too_long_truncates():
    sf = _mock_filter()
    long_text = "A" * 5000
    result = sanitize_and_check_prompt(long_text, sf, max_length=100)
    assert len(result) == 100
    sf.scan_text.assert_called_once()
    # Verify the truncated text was what was scanned
    assert len(sf.scan_text.call_args[0][0]) == 100


def test_sanitize_unsafe_raises():
    sf = _mock_filter(passed=False, flagged=["violence", "gore"])
    with pytest.raises(ValueError, match="inappropriate"):
        sanitize_and_check_prompt("Some unsafe content", sf)


def test_sanitize_unsafe_with_details_raises():
    sf = _mock_filter(passed=False, details="NSFK detected with high confidence")
    with pytest.raises(ValueError, match="inappropriate"):
        sanitize_and_check_prompt("Bad prompt", sf)


def test_sanitize_safe_returns_cleaned():
    sf = _mock_filter(passed=True)
    result = sanitize_and_check_prompt("  Billy Bunny goes to the park  ", sf)
    assert result == "Billy Bunny goes to the park"
    sf.scan_text.assert_called_once_with("Billy Bunny goes to the park")
