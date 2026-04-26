import datetime
from datetime import datetime, timedelta
import json
import time
import os
import re
import math
"""
═══════════════════════════════════════════════════════════════
AniMate Studio — Story Engine
═══════════════════════════════════════════════════════════════
LLM-powered storyboard generator that creates structured
5-scene stories with emotional arcs for kids' animation.

Supports: Local (Ollama), OpenAI API, or compatible endpoints.
═══════════════════════════════════════════════════════════════
"""


import json
import logging
import os
import re
import datetime
from datetime import datetime
import time
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import threading
import time

# Add tenacity for retry/circuit breaker
from tenacity import retry, stop_after_attempt, wait_exponential, RetryError, before_sleep_log

from engine.config import ConfigError, load_config, require_env_secret
from utils.prompt_templates import (
    get_story_system_prompt,
    get_story_user_prompt,
    SCENE_SETTINGS,
)

logger = logging.getLogger("animate_studio.story")


class StoryParseError(Exception):
    """Raised when the LLM response cannot be parsed into a valid storyboard."""
    pass


@dataclass
class Scene:
    """A single scene in the storyboard."""
    scene_id: int
    narration: str
    visual_description: str
    emotion_tone: str = "happy"
    setting: str = "meadow"
    duration_s: float = 10.0


@dataclass
class Storyboard:
    """Complete storyboard with metadata."""
    title: str
    moral: str
    scenes: list[Scene] = field(default_factory=list)
    character_name: str = ""
    character_type: str = ""
    theme: str = ""
    total_duration_s: float = 0.0

    def __post_init__(self):
        self.total_duration_s = sum(s.duration_s for s in self.scenes)



class StoryEngine:
    """
    Generates structured storyboards from a one-sentence theme.

    Uses an LLM (local or API) to create scenes with emotional arcs:
    setup → curiosity → challenge → resolution → happy ending
    """

    def _connectivity_guard(self):
        """
        Validate base_url and diagnose connectivity before LLM call.
        """
        import socket
        import requests
        # Ensure base_url is valid
        if not (self.base_url.startswith("http://") or self.base_url.startswith("https://")):
            logger.error(f"Invalid base_url: {self.base_url}. Must start with http:// or https://")
            raise ValueError(f"Invalid base_url: {self.base_url}. Must start with http:// or https://")
        # Try to connect to base_url host
        try:
            host = self.base_url.split('//')[1].split('/')[0]
            host_only = host.split(':')[0]
            port = int(host.split(':')[1]) if ':' in host else 80
            with socket.create_connection((host_only, port), timeout=3):
                logger.info(f"Connectivity check: {host_only}:{port} reachable.")
        except Exception as e:
            logger.warning(f"Connectivity check failed for {self.base_url}: {e}")
        # Ping Ollama and OpenAI endpoints
        endpoints = ["http://localhost:11434", "https://api.openai.com"]
        for url in endpoints:
            try:
                resp = requests.get(url, timeout=2)
                logger.info(f"Ping {url}: {resp.status_code}")
            except Exception as e:
                logger.warning(f"Ping {url} failed: {e}")
        # Log user-facing fix if any endpoint is unreachable
        logger.info("If you see connection errors, check your base_url and ensure Ollama or OpenAI is running.")

    def __init__(self, config_path: str = "config.yaml", config: Optional[dict] = None):
        self.config = load_config(config_path=config_path, config=config)

        # Beast Mode config
        self.beast_mode_cfg = self.config.get("beast_mode", {})
        self.beast_mode_enabled = self.beast_mode_cfg.get("enabled", False)

        story_cfg = self.config.get("models", {}).get("story", {})
        self.provider = story_cfg.get("provider", "local")
        self.model_name = story_cfg.get("model_name", "llama3.2")
        self.base_url = story_cfg.get("openai_base_url", "http://localhost:11434/v1")
        self.api_key = story_cfg.get("openai_api_key", "")
        self.temperature = story_cfg.get("temperature", 0.8)
        self.max_tokens = story_cfg.get("max_tokens", 2048)


    def generate_manifest(
        self,
        theme: str,
        character_name: str = "Billy",
        character_type: str = "bunny",
        num_scenes: int = 5,
        scene_duration: float = 10.0,
    ) -> dict:
        """
        Agentic workflow: Story Architect → Cinematographer → Visual Continuity.
        Returns manifest dict (to be saved as manifest.json) and style lock dict.
        """
        # 1. Story Architect: Slice input into action/narrative blocks
        architect_prompt = (
            f"You are a Story Architect. Given the theme '{theme}', "
            f"character '{character_name}' ({character_type}), and {num_scenes} scenes, "
            "break the story into a sequence of 5-15s action blocks with clear narrative flow. "
            "Output a JSON list of scenes, each with: scene_id, action, emotion_tone, setting, and intended camera movement."
        )
        architect_response = self._call_llm(architect_prompt, "Respond in JSON only.")
        try:
            architect_scenes = json.loads(architect_response)
        except Exception as e:
            logger.error(f"Failed to parse Story Architect response: {e}\n{architect_response}")
            raise StoryParseError("Story Architect failed.")

        # 2. Cinematographer Agent: Inject camera metadata
        for scene in architect_scenes:
            cam_prompt = (
                f"You are a Cinematographer. For the following scene, suggest a camera movement and angle.\n"
                f"Scene: {scene['action']}\n"
                "Respond with a @camera: directive (e.g., @camera: tracking-shot, low-angle-dolly)."
            )
            cam_response = self._call_llm(cam_prompt, "Respond with a single @camera: directive.")
            scene['camera'] = cam_response.strip()

        # 3. Visual Continuity Agent: Style Lock for each character
        style_lock_prompt = (
            f"You are a Visual Continuity Agent. For character '{character_name}' ({character_type}), "
            "define a style lock JSON with exact hex color codes, fur/skin texture, and unique accessories. "
            "Example: {\"fur_color\": \"#F7C8D0\", \"bow_color\": \"#FF69B4\", \"fur_texture\": \"soft, plush\", \"accessory\": \"pink bow\"}"
        )
        style_lock_response = self._call_llm(style_lock_prompt, "Respond in JSON only.")
        try:
            style_lock = json.loads(style_lock_response)
        except Exception as e:
            logger.error(f"Failed to parse Style Lock: {e}\n{style_lock_response}")
            style_lock = {}

        # Compose manifest
        manifest = {
            "title": f"{character_name} — {theme}",
            "theme": theme,
            "character": {
                "name": character_name,
                "type": character_type,
                "style_lock": style_lock,
            },
            "scenes": architect_scenes,
            "created_at": datetime.now().isoformat(),
        }
        # Save manifest.json (optional: can be handled by caller)
        with open("manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Manifest generated with {len(architect_scenes)} scenes and style lock.")
        return manifest, style_lock


    # Manual circuit breaker: open after 3 consecutive failures, reset after 60s
    class ManualCircuitBreaker:
        def __init__(self, fail_max=3, reset_timeout=60):
            self.fail_max = fail_max
            self.reset_timeout = reset_timeout
            self.failure_count = 0
            self.lock = threading.Lock()
            self.opened_at = None
        def __call__(self, func):
            def wrapper(*args, **kwargs):
                with self.lock:
                    if self.opened_at and (time.time() - self.opened_at < self.reset_timeout):
                        raise RuntimeError("Circuit breaker is open")
                    if self.opened_at and (time.time() - self.opened_at >= self.reset_timeout):
                        self.failure_count = 0
                        self.opened_at = None
                try:
                    result = func(*args, **kwargs)
                except Exception as e:
                    with self.lock:
                        self.failure_count += 1
                        if self.failure_count >= self.fail_max:
                            self.opened_at = time.time()
                            logger.warning(f"Manual circuit breaker tripped for {func.__name__}")
                    raise
                return result
            return wrapper
    _llm_circuit_breaker = ManualCircuitBreaker(fail_max=3, reset_timeout=60)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=2, min=2, max=10),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )
    @_llm_circuit_breaker
    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the configured LLM provider with retry and circuit breaker."""
        self._connectivity_guard()
        if self.provider in ("openai", "ollama", "local"):
            try:
                return self._call_openai_compatible(system_prompt, user_prompt)
            except Exception as e:
                logger.warning(f"LLM call failed: {type(e).__name__}: {e}")
                raise
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI-compatible API (works with Ollama, LM Studio, vLLM, etc.)."""
        from openai import OpenAI

        if self.provider == "openai":
            self.api_key = require_env_secret(self.config, "OPENAI_API_KEY", "story generation")
        elif not self.api_key:
            self.api_key = os.getenv("OPENAI_API_KEY", "")

        client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key or "not-needed",
        )

        response = client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            timeout=60,
        )

        content = response.choices[0].message.content
        logger.debug("LLM response length: %d chars", len(content))

        # Track usage (best-effort — never break generation)
        try:
            from engine.usage_tracker import get_tracker
            usage = getattr(response, "usage", None)
            if usage:
                get_tracker(self.config).log_llm_call(
                    provider=self.provider,
                    model=self.model_name,
                    prompt_tokens=getattr(usage, "prompt_tokens", 0),
                    completion_tokens=getattr(usage, "completion_tokens", 0),
                    total_tokens=getattr(usage, "total_tokens", 0),
                )
        except Exception:
            logger.debug("Usage tracking failed for LLM call", exc_info=True)

        return content

    def _parse_response(self, raw: str) -> dict:
        """
        Parse LLM response into structured dict.
        Handles JSON wrapped in markdown code blocks.
        """
        # Try to extract JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw, re.DOTALL)
        if json_match:
            raw = json_match.group(1)

        # Try direct JSON parse
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass

        # Try to find the first valid JSON object in the text
        decoder = json.JSONDecoder()
        for i, ch in enumerate(raw):
            if ch == '{':
                try:
                    obj, _ = decoder.raw_decode(raw, i)
                    return obj
                except json.JSONDecodeError:
                    continue

        logger.error("Failed to parse LLM response as JSON.")
        # Save raw response for debugging
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "failed_story_responses.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"--- {datetime.now().isoformat()} ---\n")
            f.write(f"Raw response ({len(raw)} chars):\n{raw[:2000]}\n\n")
        raise StoryParseError(
            f"LLM returned unparseable response ({len(raw)} chars). "
            f"Raw text saved to logs/failed_story_responses.log. "
            f"Try regenerating or simplifying the theme."
        )

    def modify_scene(
        self,
        storyboard: Storyboard,
        scene_id: int,
        new_narration: Optional[str] = None,
        new_visual: Optional[str] = None,
        new_emotion: Optional[str] = None,
        new_setting: Optional[str] = None,
    ) -> Storyboard:
        """
        Modify a specific scene in the storyboard.
        This counts as "human creative input" for monetization compliance.
        """
        for scene in storyboard.scenes:
            if scene.scene_id == scene_id:
                if new_narration:
                    scene.narration = new_narration
                if new_visual:
                    scene.visual_description = new_visual
                if new_emotion:
                    scene.emotion_tone = new_emotion
                if new_setting:
                    scene.setting = new_setting
                logger.info("Scene %d modified (human creative input)", scene_id)
                break
        return storyboard

    def storyboard_to_dict(self, storyboard: Storyboard) -> dict:
        """Convert storyboard to serializable dict."""
        return {
            "title": storyboard.title,
            "moral": storyboard.moral,
            "character_name": storyboard.character_name,
            "character_type": storyboard.character_type,
            "theme": storyboard.theme,
            "total_duration_s": storyboard.total_duration_s,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "narration": s.narration,
                    "visual_description": s.visual_description,
                    "emotion_tone": s.emotion_tone,
                    "setting": s.setting,
                    "duration_s": s.duration_s,
                }
                for s in storyboard.scenes
            ],
        }
