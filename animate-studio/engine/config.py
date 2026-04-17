"""Shared configuration loading and environment secret resolution."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import yaml


class ConfigError(RuntimeError):
    """Raised when runtime configuration is invalid."""


def load_config(config_path: str = "config.yaml", config: Optional[dict[str, Any]] = None) -> dict[str, Any]:
    """Load config.yaml, overlay config.local.yaml, and resolve env-backed secrets."""
    if config is not None:
        resolved = deepcopy(config)
        config_file = Path(config_path).resolve()
    else:
        config_file = Path(config_path).resolve()
        with config_file.open("r", encoding="utf-8") as handle:
            resolved = yaml.safe_load(handle) or {}

        local_path = config_file.with_name("config.local.yaml")
        if local_path.exists():
            with local_path.open("r", encoding="utf-8") as handle:
                local_config = yaml.safe_load(handle) or {}
            resolved = _deep_merge(resolved, local_config)

    _resolve_app_paths(resolved, config_file.parent)
    _resolve_env_overrides(resolved)
    return resolved


def require_env_secret(config: dict[str, Any], env_var: str, purpose: str) -> str:
    """Return a required env-backed secret or raise a clear configuration error."""
    value = os.getenv(env_var, "")
    if value:
        return value

    if env_var == "OPENAI_API_KEY":
        value = config.get("models", {}).get("story", {}).get("openai_api_key", "")
    elif env_var == "ELEVENLABS_API_KEY":
        value = config.get("audio", {}).get("tts", {}).get("elevenlabs_api_key", "")

    if value:
        return value

    raise ConfigError(f"Missing required secret for {purpose}. Set the {env_var} environment variable.")


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_app_paths(config: dict[str, Any], base_dir: Path) -> None:
    app_cfg = config.setdefault("app", {})
    for key in ("output_dir", "assets_dir", "loras_dir", "log_file", "safety_audit_log"):
        raw_value = app_cfg.get(key)
        if not raw_value:
            continue
        path_value = Path(raw_value)
        if not path_value.is_absolute():
            path_value = (base_dir / path_value).resolve()
        app_cfg[key] = str(path_value)

    perf_cfg = config.setdefault("performance", {})
    cache_dir = perf_cfg.get("cache_dir")
    if cache_dir:
        cache_path = Path(cache_dir)
        if not cache_path.is_absolute():
            cache_path = (base_dir / cache_path).resolve()
        perf_cfg["cache_dir"] = str(cache_path)

    motion_cfg = config.setdefault("motion", {})
    motion_module_path = motion_cfg.get("motion_module_path")
    if motion_module_path:
        motion_path = Path(motion_module_path)
        if not motion_path.is_absolute():
            motion_path = (base_dir / motion_path).resolve()
        motion_cfg["motion_module_path"] = str(motion_path)

    lora_cfg = config.setdefault("lora", {})
    lora_path = lora_cfg.get("path")
    if lora_path:
        lora_file = Path(lora_path)
        if not lora_file.is_absolute():
            lora_file = (base_dir / lora_file).resolve()
        lora_cfg["path"] = str(lora_file)


def _resolve_env_overrides(config: dict[str, Any]) -> None:
    story_cfg = config.setdefault("models", {}).setdefault("story", {})
    audio_tts_cfg = config.setdefault("audio", {}).setdefault("tts", {})

    story_cfg["openai_base_url"] = os.getenv(
        "OPENAI_BASE_URL",
        story_cfg.get("openai_base_url", ""),
    )
    story_cfg["openai_api_key"] = os.getenv("OPENAI_API_KEY", "")
    audio_tts_cfg["elevenlabs_api_key"] = os.getenv("ELEVENLABS_API_KEY", "")
