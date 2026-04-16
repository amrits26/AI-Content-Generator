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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import yaml

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

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        story_cfg = self.config["models"]["story"]
        self.provider = story_cfg["provider"]
        self.model_name = story_cfg["model_name"]
        self.base_url = story_cfg.get("openai_base_url", "")
        self.api_key = os.environ.get("OPENAI_API_KEY") or story_cfg.get("openai_api_key", "")
        self.temperature = story_cfg.get("temperature", 0.8)
        self.max_tokens = story_cfg.get("max_tokens", 2048)

    def generate_storyboard(
        self,
        theme: str,
        character_name: str = "Billy",
        character_type: str = "bunny",
        num_scenes: int = 5,
        scene_duration: float = 10.0,
    ) -> Storyboard:
        """
        Generate a complete storyboard from a theme.

        Args:
            theme: One-sentence story concept (e.g., "Billy Bunny learns to share")
            character_name: Main character name
            character_type: Animal type (bunny, duckling, kitten, etc.)
            num_scenes: Number of scenes (3-8)
            scene_duration: Target duration per scene in seconds
        """
        num_scenes = max(3, min(8, num_scenes))

        system_prompt = get_story_system_prompt()
        user_prompt = get_story_user_prompt(
            theme=theme,
            character_name=character_name,
            character_type=character_type,
            num_scenes=num_scenes,
        )

        logger.info("Generating storyboard: '%s' (%d scenes)", theme, num_scenes)

        raw_response = self._call_llm(system_prompt, user_prompt)
        story_data = self._parse_response(raw_response)

        scenes = []
        for s in story_data.get("scenes", []):
            setting = s.get("setting", "meadow")
            visual = s.get("visual_description", "")
            narration = s.get("narration", "")
            emotion_tone = s.get("emotion_tone", "happy")

            if isinstance(setting, dict):
                setting = setting.get("name") or setting.get("value") or "meadow"
            if isinstance(visual, dict):
                visual = visual.get("description") or visual.get("text") or json.dumps(visual)
            if isinstance(narration, dict):
                narration = narration.get("text") or narration.get("narration") or json.dumps(narration)
            if isinstance(emotion_tone, dict):
                emotion_tone = emotion_tone.get("value") or emotion_tone.get("tone") or "happy"

            setting = str(setting)
            visual = str(visual)
            narration = str(narration)
            emotion_tone = str(emotion_tone)

            # Enrich visual description with SCENE_SETTINGS background
            setting_desc = SCENE_SETTINGS.get(setting, SCENE_SETTINGS.get("meadow", ""))
            if setting_desc and setting_desc.lower() not in visual.lower():
                visual = f"{visual}, {setting_desc}"
            scenes.append(Scene(
                scene_id=s.get("scene_id", len(scenes) + 1),
                narration=narration,
                visual_description=visual,
                emotion_tone=emotion_tone,
                setting=setting,
                duration_s=scene_duration,
            ))

        # Fallback: if LLM returned fewer scenes, pad with defaults
        while len(scenes) < num_scenes:
            scenes.append(Scene(
                scene_id=len(scenes) + 1,
                narration=f"{character_name} smiles happily.",
                visual_description=f"{character_name} the {character_type} standing in a sunny meadow, smiling",
                emotion_tone="happy",
                setting="meadow",
                duration_s=scene_duration,
            ))

        storyboard = Storyboard(
            title=story_data.get("title", theme),
            moral=story_data.get("moral", "Be kind to others."),
            scenes=scenes[:num_scenes],
            character_name=character_name,
            character_type=character_type,
            theme=theme,
        )

        logger.info(
            "Storyboard ready: '%s' — %d scenes, %.1fs total",
            storyboard.title, len(storyboard.scenes), storyboard.total_duration_s,
        )
        return storyboard

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call the configured LLM provider."""
        if self.provider in ("openai", "ollama", "local"):
            return self._call_openai_compatible(system_prompt, user_prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI-compatible API (works with Ollama, LM Studio, vLLM, etc.)."""
        from openai import OpenAI

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
        )

        content = response.choices[0].message.content
        logger.debug("LLM response length: %d chars", len(content))
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
