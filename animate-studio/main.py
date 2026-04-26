
import datetime
from datetime import datetime, timedelta
import json
import time
import os
import re
import math
# psutil availability check for _kill_port
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# Ensure datetime is imported and not shadowed
from datetime import datetime

import datetime
from datetime import datetime
import json
import time
import math
import re

import gradio as gr
import logging
import os
import time
import json
import psutil

# Set up logger
logger = logging.getLogger("animate_studio")
logging.basicConfig(level=logging.INFO)

# AniMate Studio — Main Application (Gradio UI)
# Pastel-themed Gradio interface with 4 tabs:
#   Tab 1: Quick Story (TikTok/Reels Hook)
#   Tab 2: Full Episode (YouTube Kids)
#   Tab 3: Character Lab

from engine.animator import Animator
from engine.story_engine import StoryEngine
from engine.usage_tracker import UsageTracker

# Load config
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
from engine.config import load_config
CONFIG = load_config(CONFIG_PATH)
def build_ui() -> gr.Blocks:
    """Construct the full Gradio interface with pastel kid-friendly theme."""
    audio_engine = create_audio_engine()

    # Cupertino luxury theme
    theme = gr.themes.Monochrome(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("SF Pro Display"),
    ).set(
        body_background_fill="linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%)",
        block_background_fill="rgba(255,255,255,0.95)",
        block_border_width="0px",
        block_shadow="0 8px 32px rgba(0,0,0,0.12)",
        block_radius="24px",
        button_primary_background_fill="linear-gradient(90deg, #007aff 0%, #00c6fb 100%)",
        button_primary_text_color="#fff",
        input_radius="16px",
    )

    custom_css = """
        body, .gradio-container {
            background: #1C1C1E !important;
            backdrop-filter: blur(10px) !important;
        }
        .gradio-container {
            max-width: 100vw !important;
            font-family: 'SF Pro Display', 'Nunito', sans-serif;
            border-radius: 24px !important;
            box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37) !important;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #E0D5B0 !important;
            font-weight: 800;
            letter-spacing: -1px;
        }
        label, .gr-input-label, .gradio-label, .gr-text-label {
            color: #E0D5B0 !important;
            font-size: 1.08em;
            font-weight: 600;
            letter-spacing: 0.01em;
        }
        .gr-textbox textarea, .gr-textbox input {
            padding: 20px !important;
            font-size: 1.1em;
            border-radius: 16px !important;
            background: rgba(255,255,255,0.08) !important;
            color: #fff !important;
        }
        .gr-button {
            border-radius: 20px !important;
            font-weight: 700;
            font-size: 1.1em;
            padding: 14px 32px !important;
            background: linear-gradient(135deg, #D4AF37 0%, #F5E6BE 100%) !important;
            color: #fff !important;
            box-shadow: 0 2px 12px 0 rgba(212,175,55,0.18) !important;
            transition: box-shadow 0.2s, filter 0.2s;
        }
        .gr-button:hover {
            filter: drop-shadow(0 0 8px #FFD70088);
            box-shadow: 0 4px 24px 0 #D4AF37cc !important;
        }
        .lux-glass {
            background: rgba(28,28,30,0.85) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.10) !important;
            border-radius: 24px !important;
            backdrop-filter: blur(10px) !important;
        }
        .status-micro-card {
            background: rgba(255,255,255,0.10) !important;
            border-radius: 14px !important;
            padding: 12px 18px !important;
            margin: 0 8px 12px 0 !important;
            color: #E0D5B0 !important;
            font-size: 1em;
            font-weight: 600;
            display: inline-block;
            min-width: 160px;
            box-shadow: 0 2px 8px 0 rgba(212,175,55,0.10) !important;
        }
        .aspect-toggle {
            background: rgba(255,255,255,0.08) !important;
            color: #E0D5B0 !important;
            border-radius: 12px !important;
            font-weight: 600;
            margin-right: 8px;
            padding: 8px 18px !important;
            border: 2px solid #D4AF37 !important;
        }
        .aspect-toggle.selected {
            background: linear-gradient(135deg, #D4AF37 0%, #F5E6BE 100%) !important;
            color: #222 !important;
        }
    """

    with gr.Blocks(title="AniMate Studio", theme=theme, css=custom_css) as app:
        gr.Markdown(
            """
            # AniMate Studio
            ### Cinematic Animation Engine — Manifest-Driven, 4K, Agentic, Luxury UI
            """
        )

        with gr.Accordion("Advanced Settings", open=False):
            beast_mode = gr.Checkbox(label="Beast Mode (Ultra Quality)", value=True)

        with gr.Tabs():
            pass  # Placeholder to ensure valid block; replace with actual tab UI code
    return app, theme, custom_css

## _check_ollama_model()  # Disabled: function not defined in this file


# ── Tokenizer check ──────────────────────────────────────
def _check_tokenizer():
    """Verify tokenizer dependencies are available."""
    ok = True
    try:
        import tiktoken  # noqa: F401
        logger.info("tiktoken tokenizer available.")
    except ImportError:
        logger.warning("tiktoken not installed — run: pip install tiktoken>=0.7.0")
        ok = False
    try:
        import sentencepiece  # noqa: F401
        logger.info("sentencepiece tokenizer available.")
    except ImportError:
        logger.warning("sentencepiece not installed — run: pip install sentencepiece>=0.2.0")
        ok = False
    if not ok:
        print(
            "\n*** MISSING TOKENIZER DEPENDENCY ***\n"
            "CogVideoX requires both tiktoken and sentencepiece.\n"
            "Run:  pip install tiktoken>=0.7.0 sentencepiece>=0.2.0\n",
            file=sys.stderr,
        )

_check_tokenizer()

# ── Import engine modules ────────────────────────────────
from engine.story_engine import StoryEngine, StoryParseError, Storyboard
from engine.character_manager import CharacterManager, CharacterProfile
from engine.animator import Animator
from engine.audio_engine import AudioEngine
from engine.safety_filter import SafetyFilter
from engine.exporter import Exporter, ExportResult
from utils.prompt_templates import (
    EMOTION_TONES,
    SCENE_SETTINGS,
    CHARACTER_TEMPLATES,
    ACCESSORIES,
    STYLE_PRESETS,
    CAMERA_MOTIONS,
    build_character_description,
)

# ── CUDA availability check ──────────────────────────────
try:
    import torch as _torch
    if not _torch.cuda.is_available():
        logger.warning(
            "CUDA not available — video generation will be extremely slow. "
            "Install PyTorch with CUDA support."
        )
    else:
        logger.info("CUDA available: %s", _torch.cuda.get_device_name(0))
except Exception as _e:
    logger.warning("CUDA check failed: %s", _e)

# ── Shared character library + request-scoped engine factories ─
character_manager = CharacterManager(CONFIG_PATH, config=CONFIG)

    # Add missing imports
    import sys
    try:
        import psutil
        _HAS_PSUTIL = True
    except ImportError:
        _HAS_PSUTIL = False


    import gradio as gr
    import logging
    import pandas as pd

    # Exception for config errors
    class ConfigError(Exception):
        pass


def create_audio_engine() -> AudioEngine:
    return AudioEngine(CONFIG_PATH, config=CONFIG)


def create_safety_filter() -> SafetyFilter:
    return SafetyFilter(CONFIG_PATH, config=CONFIG)


def create_exporter() -> Exporter:
    return Exporter(CONFIG_PATH, config=CONFIG)

# ── Human input tracking for monetization compliance ─────
_human_inputs_log = []


def log_human_input(action: str):
    """Log a human creative decision for monetization compliance."""
    entry = {"action": action, "timestamp": datetime.now().isoformat()}
    _human_inputs_log.append(entry)
    if len(_human_inputs_log) > 1000:
        del _human_inputs_log[:-1000]
    logger.info("Human input logged: %s", action)


# ── Input sanitization ───────────────────────────────────
def sanitize_and_check_prompt(text: str, safety_filter: SafetyFilter, max_length: int = 2000) -> str:
    """Strip, truncate, and safety-check a user-supplied prompt.

    Raises ValueError if the prompt is empty or flagged as unsafe.
    """
    if not text or not text.strip():
        raise ValueError("Prompt cannot be empty.")
    text = text.strip()[:max_length]
    result = safety_filter.scan_text(text)
    if not result.passed:
        raise ValueError(
            f"Prompt flagged as inappropriate: {', '.join(result.flagged_concepts) if result.flagged_concepts else result.details}"
        )
    return text


# ═════════════════════════════════════════════════════════
# TAB 1: QUICK STORY (TikTok / Reels Hook)
# ═════════════════════════════════════════════════════════

def quick_generate(theme, duration, character_choice, style_choice, quality_preset, beast_mode=True, progress=gr.Progress()):
    """Generate a quick 15-60s hook for TikTok/Reels.

    beast_mode: received from UI, currently not used in engine call.
    """
    import traceback
    try:
        # Disable button immediately
        yield None, "⏳ Generating...", "", gr.update(interactive=False)
        animator = create_animator()
        animator.beast_mode_enabled = beast_mode
        audio_engine = create_audio_engine()
        # Input sanitization
        try:
            theme = sanitize_and_check_prompt(theme, create_safety_filter(), max_length=500)
        except ValueError as e:
            yield None, f"**Input rejected:** {e}", "", gr.update(interactive=True)
            return
        log_human_input(f"Quick story theme: {theme}")
        log_human_input(f"Duration choice: {duration}s")
        log_human_input(f"Character: {character_choice}")
        log_human_input(f"Style: {style_choice}, Quality: {quality_preset}")
        progress(0.05, desc="Preparing generation...")
        # ...existing code...
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        yield None, "**An error occurred. Please check your internet connection or try again.**", "", gr.update(interactive=True)
        return

    log_human_input(f"Quick story theme: {theme}")
    log_human_input(f"Duration choice: {duration}s")
    log_human_input(f"Character: {character_choice}")
    log_human_input(f"Style: {style_choice}, Quality: {quality_preset}")

    progress(0.05, desc="Preparing generation...")

    # Determine character
    char_name = character_choice if character_choice != "None" else "Billy"
    profile = character_manager.get_profile(char_name)
    char_type = profile.animal_type if profile else "bunny"

    if beast_mode:
        # Direct single-scene generation: prompt is ONLY the raw theme string
        fps = CONFIG["models"]["video"].get("fps", 12)
        num_frames = int(duration * fps)
        from engine.story_engine import Scene, Storyboard
        scene = Scene(
            scene_id=1,
            narration="",  # No narration, just the raw theme as visual prompt
            visual_description=theme,  # Only the theme string as prompt
            emotion_tone="happy",
            setting="meadow",
            duration_s=duration,
        )
        storyboard = Storyboard(
            title=theme,
            moral="",
            scenes=[scene],
            character_name=char_name,
            character_type=char_type,
            theme=theme,
            total_duration_s=duration,
        )
        progress(0.15, desc="Generating animation...")
    else:
        story_engine = create_story_engine()
        # Map duration to scene count
        if duration <= 15:
            num_scenes = 2
        elif duration <= 30:
            num_scenes = 3
        else:
            num_scenes = 5

        scene_duration = duration / num_scenes

        try:
            manifest, style_lock = story_engine.generate_manifest(
                theme=theme,
                character_name=char_name,
                character_type=char_type,
                num_scenes=num_scenes,
                scene_duration=scene_duration,
            )
        except Exception as e:
            logger = logging.getLogger("animate_studio")
            logger.warning(f"StoryEngine failed ({e}); using fallback manifest with raw theme.")
            manifest = {
                "title": theme,
                "theme": theme,
                "character": {
                    "name": char_name,
                    "type": char_type,
                    "style_lock": {},
                },
                "scenes": [
                    {
                        "scene_id": 1,
                        "action": theme,
                        "emotion_tone": "neutral",
                        "setting": "meadow",
                        "camera": "static-wide",
                    }
                ],
                "created_at": datetime.now().isoformat(),
            }
            style_lock = {}
        progress(0.15, desc="Generating animation...")

    # Build per-request generation overrides (never mutate global animator)
    gen_kwargs = {}
    if quality_preset and quality_preset != "default":
        presets = CONFIG.get("quality_presets", {})
        if quality_preset in presets:
            qp = presets[quality_preset]
            gen_kwargs["num_inference_steps"] = qp.get("num_inference_steps")
            gen_kwargs["gen_height"] = qp.get("height")
            gen_kwargs["gen_width"] = qp.get("width")
            gen_kwargs["fps"] = qp.get("fps")
            gen_kwargs["guidance_scale"] = qp.get("guidance_scale")
    if style_choice and style_choice != "default":
        gen_kwargs["style"] = style_choice
    # Remove None values so generate_episode uses its own defaults
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    # Generate video
    def anim_progress(idx, total, msg):
        progress((0.15 + 0.6 * idx / total), desc=msg)

    try:
        episode = animator.generate_episode(
            storyboard=storyboard,
            character_name=char_name if character_manager.get_profile(char_name) else None,
            progress_callback=anim_progress,
            **gen_kwargs,
        )
    except ConfigError as e:
        yield None, f"Configuration error: {e}", "", gr.update(interactive=True)
        return
    except Exception as e:
        yield None, f"Animation failed: {e}", "", gr.update(interactive=True)
        return
    finally:
        animator.cleanup_request_state()

    if not episode.get("video_path"):
        yield None, "No video generated — all scenes failed safety.", "", gr.update(interactive=True)
        return

    progress(0.8, desc="Adding narration...")

    # Generate narration
    narration_texts = [s.narration for s in storyboard.scenes]
    audio_warning = ""
    try:
        audio_dir = os.path.join(CONFIG["app"]["output_dir"], "_quick_audio")
        narration = audio_engine.generate_episode_narration(narration_texts, audio_dir)
        final_video = audio_engine.apply_audio_to_video(
            video_path=episode["video_path"],
            narration_path=narration["full_narration_path"],
        )
    except ConfigError as e:
        logger.warning("Audio configuration failed, returning video without narration: %s", e)
        final_video = episode["video_path"]
        audio_warning = f"\n\n⚠️ **Audio configuration failed:** {e}. Video exported without narration.\n"
    except Exception as e:
        logger.warning("Audio failed, returning video without narration: %s", e)
        final_video = episode["video_path"]
        audio_warning = f"\n\n⚠️ **Audio failed:** {e}. Video exported without narration.\n"

    progress(0.95, desc="Finalizing...")

    # Build info text
    story_info = f"**{storyboard.title}**\n\n"
    story_info += f"*Moral: {storyboard.moral}*\n\n"
    for s in storyboard.scenes:
        story_info += f"**Scene {s.scene_id}** ({s.emotion_tone}): {s.narration}\n\n"
    if audio_warning:
        story_info += audio_warning

    # Fix: story_engine may not be defined in beast_mode path
    try:
        storyboard_json = json.dumps(story_engine.storyboard_to_dict(storyboard), indent=2)
    except UnboundLocalError:
        # Minimal dict for beast_mode
        storyboard_json = json.dumps({
            "title": storyboard.title,
            "moral": storyboard.moral,
            "scenes": [
                {
                    "scene_id": s.scene_id,
                    "narration": s.narration,
                    "visual_description": s.visual_description,
                    "emotion_tone": s.emotion_tone,
                    "setting": s.setting,
                    "duration_s": s.duration_s,
                } for s in storyboard.scenes
            ],
            "character_name": storyboard.character_name,
            "character_type": storyboard.character_type,
            "theme": storyboard.theme,
            "total_duration_s": storyboard.total_duration_s,
        }, indent=2)
    yield final_video, story_info, storyboard_json, gr.update(interactive=True)


def export_for_reels(video_path, story_json):
    """Quick export optimized for Facebook Reels."""
    import traceback
    try:
        if not video_path:
            return "No video to export."
        safety_filter = create_safety_filter()
        exporter = create_exporter()
        story_data = json.loads(story_json) if story_json else {}
        title = story_data.get("title", "Quick_Hook")
        # Use narration_texts from story_data if available, else empty
        narration_texts = [s.get("narration", "") for s in story_data.get("scenes", [])]
        safety_result = safety_filter.scan_text(" ".join(narration_texts))
        result = exporter.export(
            source_video=video_path,
            platform="facebook_reels",
            title=title,
            narration_texts=narration_texts,
            safety_result=safety_result,
            human_input_logged=True,  # Quick Story always has human theme input
        )
        if result.success:
            return f"Exported: {result.video_path}\nThumbnail: {result.thumbnail_path}\nCaptions: {result.caption_path}"
        else:
            return f"Export failed: {result.error}"
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return "**An error occurred. Please check your internet connection or try again.**"


# ═════════════════════════════════════════════════════════
# TAB 2: FULL EPISODE (YouTube Kids) — Two-step workflow
# ═════════════════════════════════════════════════════════

def generate_storyboard_preview(
    story_concept, num_scenes, character_choice, progress=gr.Progress()
):
    """Step 1: Generate storyboard and return an editable Dataframe + JSON state."""
    empty_df = pd.DataFrame(columns=["Scene", "Narration", "Visual Description", "Emotion", "Setting"])

    # Disable button immediately
    yield empty_df, "{}", "⏳ Generating storyboard...", gr.update(interactive=False)

    story_engine = create_story_engine()

    # Input sanitization
    import traceback
    try:
        story_concept = sanitize_and_check_prompt(story_concept, create_safety_filter(), max_length=2000)
    except ValueError as e:
        yield empty_df, "{}", f"**Input rejected:** {e}", gr.update(interactive=True)
        return

    log_human_input(f"Storyboard concept: {story_concept}")
    progress(0.1, desc="Crafting storyboard...")

    char_name = character_choice if character_choice != "None" else "Billy"
    profile = character_manager.get_profile(char_name)
    char_type = profile.animal_type if profile else "bunny"
    target_duration = CONFIG.get("motion", {}).get("final_duration_s", 60.0)
    scene_duration = target_duration / num_scenes

    try:
        manifest, style_lock = story_engine.generate_manifest(
            theme=story_concept,
            character_name=char_name,
            character_type=char_type,
            num_scenes=int(num_scenes),
            scene_duration=scene_duration,
        )
    except Exception as e:
        yield empty_df, "{}", f"**Story generation failed:** {e}", gr.update(interactive=True)
        return

    progress(1.0, desc="Storyboard ready!")

    # Build editable Dataframe rows from manifest
    rows = []
    for s in manifest["scenes"]:
        rows.append({
            "Scene": s.get("scene_id"),
            "Action": s.get("action"),
            "Emotion": s.get("emotion_tone"),
            "Setting": s.get("setting"),
            "Camera": s.get("camera"),
        })
    df = pd.DataFrame(rows)

    # Store manifest as JSON state for Step 2
    sb_json = json.dumps(manifest, indent=2)

    info = f"# {manifest['title']}\n**Theme:** {manifest['theme']}\n\n"
    info += f"**{len(manifest['scenes'])} scenes** — review & edit below, then click **Render Episode**."


    yield df, sb_json, info, gr.update(interactive=True)

    # Outer error handler for any unexpected errors
    # (This should be the last statement in the function, not outside)



def render_episode_from_storyboard(
    storyboard_df, storyboard_json,
    voice_choice, bgm_choice, character_choice,
    style_choice, quality_preset, camera_motion,
    lora_strength, progress=gr.Progress()
):
    """Step 2: Render the (possibly edited) storyboard into a full episode."""
    import traceback
    try:
        # Disable button immediately
        yield None, None, "⏳ Rendering episode...", None, gr.update(interactive=False)
        animator = create_animator()
        audio_engine = create_audio_engine()
        safety_filter = create_safety_filter()
        exporter = create_exporter()
        # Per-request input tracking for monetization compliance
        run_inputs = []
        def _log(action):
            run_inputs.append(action)
            log_human_input(action)
        _log(f"Render episode — Voice: {voice_choice}, Character: {character_choice}")
        _log(f"Style: {style_choice}, Quality: {quality_preset}, Camera: {camera_motion}")
        # Rebuild storyboard from JSON state, applying any Dataframe edits
        try:
            sb_dict = json.loads(storyboard_json)
        except Exception:
            yield None, None, "**Error:** No storyboard to render. Generate one first.", None, gr.update(interactive=True)
            return
        from engine.story_engine import Scene, Storyboard
        char_name = character_choice if character_choice != "None" else "Billy"
        profile = character_manager.get_profile(char_name)
        char_type = profile.animal_type if profile else "bunny"
        target_duration = CONFIG.get("motion", {}).get("final_duration_s", 60.0)
        num_scenes = len(sb_dict.get("scenes", []))
        scene_duration = target_duration / max(num_scenes, 1)
        # Apply edits from the Dataframe back onto the storyboard
        edited_df = storyboard_df if isinstance(storyboard_df, pd.DataFrame) else pd.DataFrame(storyboard_df)
        # ...existing code...
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        yield None, None, "**An error occurred. Please check your internet connection or try again.**", None, gr.update(interactive=True)
        return
    scenes = []
    for i, scene_data in enumerate(sb_dict.get("scenes", [])):
        narration = scene_data.get("narration", "")
        visual = scene_data.get("visual_description", "")
        emotion = scene_data.get("emotion_tone", "happy")
        setting = scene_data.get("setting", "meadow")

        # Overwrite from Dataframe if the user edited cells
        if i < len(edited_df):
            row = edited_df.iloc[i]
            df_narration = str(row.get("Narration", "")).strip()[:500]
            df_visual = str(row.get("Visual Description", "")).strip()[:500]
            df_emotion = str(row.get("Emotion", "")).strip()[:50]
            df_setting = str(row.get("Setting", "")).strip()[:50]
            if df_narration and df_narration != narration:
                _log(f"Scene {i+1} narration edited")
                narration = df_narration
            if df_visual and df_visual != visual:
                _log(f"Scene {i+1} visual edited")
                visual = df_visual
            if df_emotion:
                emotion = df_emotion
            if df_setting:
                setting = df_setting

        scenes.append(Scene(
            scene_id=i + 1,
            narration=narration,
            visual_description=visual,
            emotion_tone=emotion,
            setting=setting,
            duration_s=scene_duration,
        ))

    storyboard = Storyboard(
        title=sb_dict.get("title", "Untitled"),
        moral=sb_dict.get("moral", ""),
        scenes=scenes,
        character_name=char_name,
        character_type=char_type,
        theme=sb_dict.get("theme", ""),
    )

    # Sanitize user-edited content before rendering
    try:
        all_text = " ".join(s.narration + " " + s.visual_description for s in scenes)
        sanitize_and_check_prompt(all_text, safety_filter, max_length=10000)
    except ValueError as e:
        yield None, None, f"**Content rejected:** {e}", None, gr.update(interactive=True)
        return

    progress(0.1, desc="Generating animation scenes...")

    # Build per-request generation overrides (never mutate global animator)
    gen_kwargs = {}
    if quality_preset and quality_preset != "default":
        presets = CONFIG.get("quality_presets", {})
        if quality_preset in presets:
            qp = presets[quality_preset]
            gen_kwargs["num_inference_steps"] = qp.get("num_inference_steps")
            gen_kwargs["gen_height"] = qp.get("height")
            gen_kwargs["gen_width"] = qp.get("width")
            gen_kwargs["fps"] = qp.get("fps")
            gen_kwargs["guidance_scale"] = qp.get("guidance_scale")
    if style_choice and style_choice != "default":
        gen_kwargs["style"] = style_choice
    if camera_motion and camera_motion != "default":
        gen_kwargs["camera_motion"] = camera_motion
    if lora_strength is not None:
        gen_kwargs["lora_strength"] = lora_strength
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

    def anim_progress(idx, total, msg):
        progress(0.1 + 0.55 * idx / total, desc=msg)

    try:
        episode = animator.generate_episode(
            storyboard=storyboard,
            character_name=char_name if character_manager.get_profile(char_name) else None,
            progress_callback=anim_progress,
            **gen_kwargs,
        )
    except ConfigError as e:
        yield None, None, f"Configuration error: {e}", None, gr.update(interactive=True)
        return
    except Exception as e:
        yield None, None, f"Animation failed: {e}", None, gr.update(interactive=True)
        return
    finally:
        animator.cleanup_request_state()

    if not episode.get("video_path"):
        yield None, None, "All scenes failed safety.", None, gr.update(interactive=True)
        return

    progress(0.7, desc="Generating narration...")

    # Narration
    narration_texts = [s.narration for s in storyboard.scenes]
    voice_id = voice_choice if voice_choice else audio_engine.default_voice_id

    audio_warning = ""
    try:
        audio_dir = os.path.join(CONFIG["app"]["output_dir"], "_episode_audio")
        narration = audio_engine.generate_episode_narration(narration_texts, audio_dir, voice_id)

        bgm_path = None
        if bgm_choice and bgm_choice != "None":
            tracks = audio_engine.list_bgm_tracks()
            bgm_match = next((t for t in tracks if t["name"] == bgm_choice), None)
            if bgm_match:
                bgm_path = bgm_match["path"]

        final_video = audio_engine.apply_audio_to_video(
            video_path=episode["video_path"],
            narration_path=narration["full_narration_path"],
            bgm_path=bgm_path,
        )
    except ConfigError as e:
        logger.warning("Audio configuration failed: %s", e)
        final_video = episode["video_path"]
        audio_warning = f"\n\n⚠️ **Audio configuration failed:** {e}. Video exported without narration.\n"
    except Exception as e:
        logger.warning("Audio failed: %s", e)
        final_video = episode["video_path"]
        audio_warning = f"\n\n⚠️ **Audio failed:** {e}. Video exported without narration.\n"

    progress(0.85, desc="Exporting for YouTube...")

    safety_result = safety_filter.scan_text(" ".join(narration_texts))
    export_result = exporter.export(
        source_video=final_video,
        platform="youtube",
        title=storyboard.title,
        narration_texts=narration_texts,
        safety_result=safety_result,
        human_input_logged=len(run_inputs) > 0,
    )

    progress(1.0, desc="Done!")

    # Story info
    story_info = f"# {storyboard.title}\n\n"
    story_info += f"**Moral:** {storyboard.moral}\n\n"
    story_info += f"**Duration:** ~{storyboard.total_duration_s:.0f}s | **Scenes:** {len(storyboard.scenes)}\n\n"
    story_info += "---\n\n"
    for s in storyboard.scenes:
        story_info += f"### Scene {s.scene_id} — {s.emotion_tone.title()}\n"
        story_info += f"*{s.narration}*\n\n"
        story_info += f"Visual: {s.visual_description}\n\n"
    if audio_warning:
        story_info += audio_warning

    if export_result.success:
        comp = export_result.compliance
        comp_text = "**Compliance Checklist:**\n"
        for k, v in comp.items():
            if k != "_all_passed":
                icon = "✅" if v else "❌"
                comp_text += f"  {icon} {k.replace('_', ' ').title()}\n"
        story_info += f"\n---\n\n{comp_text}"

    thumbnail = export_result.thumbnail_path if export_result.success else None
    caption_file = export_result.caption_path if export_result.success else None

    yield final_video, thumbnail, story_info, caption_file, gr.update(interactive=True)


# ═════════════════════════════════════════════════════════
# TAB 3: CHARACTER LAB
# ═════════════════════════════════════════════════════════

def list_characters():
    """Get character list for dropdowns."""
    profiles = character_manager.list_profiles()
    names = [p.name for p in profiles]
    return names if names else ["None"]


def get_character_dropdown_choices():
    """Build dropdown choices including 'None'."""
    names = list_characters()
    return ["None"] + [n for n in names if n != "None"]


def save_character(name, animal_type, color, accessory, traits_text, ref_image):
    """Save a new character profile."""
    import traceback
    try:
        if not name.strip():
            return "Please enter a character name."
        log_human_input(f"Created character: {name}")
        traits = [t.strip() for t in traits_text.split(",") if t.strip()]
        # Save reference image if provided
        ref_path = None
        if ref_image is not None:
            from PIL import Image
            img = Image.open(ref_image).convert("RGB")
            ref_path = character_manager.save_reference_image(name, img)
        # Check for matching LoRA
        lora_path = None
        lora_trigger = ""
        lora_file = os.path.join(character_manager.loras_dir, f"{name}.safetensors")
        if os.path.exists(lora_file):
            lora_path = lora_file
            lora_trigger = name.lower()
        profile = CharacterProfile(
            name=name,
            animal_type=animal_type,
            color=color,
            accessory=accessory,
            traits=traits,
            reference_image=ref_path,
            lora_path=lora_path,
            lora_trigger_word=lora_trigger,
        )
        character_manager.create_profile(profile)
        desc = build_character_description(animal_type, name, color, accessory)
        return f"**{name}** saved!\n\nPrompt preview:\n> {desc}"
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return "**An error occurred. Please check your internet connection or try again.**"


def preview_character(name):
    """Show character info and prompt preview."""
    import traceback
    try:
        profile = character_manager.get_profile(name)
        if not profile:
            return "Character not found.", None
        desc = character_manager.get_character_prompt(name)
        info = f"**{profile.name}** the {profile.animal_type}\n\n"
        info += f"- Color: {profile.color}\n"
        info += f"- Accessory: {profile.accessory}\n"
        info += f"- Traits: {', '.join(profile.traits)}\n"
        info += f"- LoRA: {'✅ ' + profile.lora_path if profile.lora_path else '❌ Not loaded'}\n"
        info += f"- Reference: {'✅' if profile.reference_image else '❌ None'}\n\n"
        info += f"**Prompt:**\n> {desc}"
        ref_img = profile.reference_image if profile.reference_image and os.path.exists(profile.reference_image) else None
        return info, ref_img
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return "**An error occurred. Please check your internet connection or try again.**", None


def get_lora_list():
    """Get available LoRAs for display."""
    loras = character_manager.list_available_loras()
    if not loras:
        return "No LoRA files found in `./loras/`. See setup guide for training instructions."
    text = "**Available LoRAs:**\n\n"
    for l in loras:
        text += f"- `{l['name']}` ({l['size_mb']} MB) — Profile: {'✅' if l['has_profile'] else '❌'}\n"
    return text


# ── Usage tab helpers ────────────────────────────────────
from engine.usage_tracker import get_tracker


def _build_usage_summary() -> str:
    """Build a markdown summary of usage totals."""
    try:
        summary = get_tracker(CONFIG).get_summary()
    except Exception:
        return "*No usage data yet.*"

    if not summary["operations"]:
        return "*No usage data yet.*"

    total_cents = summary["total_cost_cents"]
    lines = [f"### Estimated Total: ${total_cents / 100:.2f}\n"]
    op_labels = {"llm_call": "LLM Calls", "tts_call": "TTS Calls", "video_generation": "Video Generation"}
    for op, data in summary["operations"].items():
        label = op_labels.get(op, op)
        lines.append(f"- **{label}:** {data['count']} calls — ${data['total_cost_cents'] / 100:.2f}")
        if data["total_tokens"]:
            lines.append(f"  ({data['total_tokens']:,} tokens)")
        if data["total_chars"]:
            lines.append(f"  ({data['total_chars']:,} characters)")
        if data["total_video_s"]:
            lines.append(f"  ({data['total_video_s']:.1f}s video)")
    return "\n".join(lines)


def _build_usage_table() -> list[list]:
    """Build a table of recent usage rows for Gradio Dataframe."""
    try:
        rows = get_tracker(CONFIG).get_recent(limit=50)
    except Exception:
        return []

    table = []
    for r in rows:
        ts = r.get("timestamp", "")[:19].replace("T", " ")
        table.append([
            ts,
            r.get("operation", ""),
            r.get("provider", "") or "",
            r.get("tokens_used") or "",
            r.get("characters_generated") or "",
            f"{r['video_duration_seconds']:.1f}" if r.get("video_duration_seconds") else "",
            f"{r['estimated_cost_cents']:.2f}" if r.get("estimated_cost_cents") else "",
        ])
    return table


# ═════════════════════════════════════════════════════════
# BUILD GRADIO UI
# ═════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    """Construct the full Gradio interface with pastel kid-friendly theme."""
    audio_engine = create_audio_engine()

    # Cupertino luxury theme
    theme = gr.themes.Monochrome(
        primary_hue=gr.themes.colors.blue,
        secondary_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("SF Pro Display"),
    ).set(
        body_background_fill="linear-gradient(135deg, #f8fafc 0%, #e0e7ef 100%)",
        block_background_fill="rgba(255,255,255,0.95)",
        block_border_width="0px",
        block_shadow="0 8px 32px rgba(0,0,0,0.12)",
        block_radius="24px",
        button_primary_background_fill="linear-gradient(90deg, #007aff 0%, #00c6fb 100%)",
        button_primary_text_color="#fff",
        input_radius="16px",
    )

    custom_css = """
        .gradio-container { max-width: 1400px !important; font-family: 'SF Pro Display', 'Nunito', sans-serif; }
        h1 { text-align: center; color: #007aff; font-weight: 800; letter-spacing: -1px; }
        .gr-button { border-radius: 16px !important; font-weight: 600; transition: background 0.2s, box-shadow 0.2s; }
        .gr-button-primary, .gr-button[variant="primary"] {
            background: linear-gradient(90deg, #D4AF37 0%, #FFD700 100%) !important;
            color: #fff !important;
            box-shadow: 0 2px 8px rgba(212,175,55,0.12) !important;
        }
        .gr-button-primary:active, .gr-button[variant="primary"]:active {
            background: #D4AF37 !important;
        }
        .lux-glass {
            background: rgba(255,255,255,0.85) !important;
            box-shadow: 0 8px 32px rgba(0,0,0,0.10) !important;
            border-radius: 24px !important;
            backdrop-filter: blur(12px) !important;
        }
        .smooth-reveal {
            opacity: 0;
            transform: translateY(24px);
            animation: smoothReveal 0.7s cubic-bezier(.4,0,.2,1) forwards;
        }
        @keyframes smoothReveal {
            to { opacity: 1; transform: none; }
        }
        .gold-progress-bar {
            background: linear-gradient(90deg, #D4AF37 0%, #FFD700 100%) !important;
            height: 8px !important;
            border-radius: 8px !important;
            margin-top: 8px;
        }
    """

    with gr.Blocks(title="🎬 AniMate Studio", theme=theme, css=custom_css) as app:
        gr.Markdown(
            """
            # 🎬 AniMate Studio
            ### 🏆 Cinematic Animation Engine — Manifest-Driven, 4K, Agentic, Luxury UI
            """
        )

        # Absolute path handling for output/cache/assets
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.abspath(os.path.join(base_dir, CONFIG["app"]["output_dir"]))
        assets_dir = os.path.abspath(os.path.join(base_dir, CONFIG["app"]["assets_dir"]))
        cache_dir = os.path.abspath(os.path.join(base_dir, CONFIG.get("performance", {}).get("cache_dir", "./output/.cache")))

        # Mission Control (left column)
        with gr.Row():
            with gr.Column(scale=5, min_width=420):
                with gr.Group(elem_classes=["lux-glass"]):
                    gr.Markdown("## Mission Control")
                    theme_box = gr.Textbox(label="Theme/Prompt", placeholder="e.g. A bunny finds a golden carrot", elem_id="quick_theme")
                    duration_slider = gr.Slider(15, 60, value=30, step=1, label="Duration (seconds)", elem_id="duration_slider")
                    character_dropdown = gr.Dropdown(["Billy", "Luna", "Max"], label="Character", value="Billy")
                    gr.Markdown("**Style**", elem_id="style_label")
                    style_pixar = gr.Button("Pixar Cute", elem_id="style_pixar", elem_classes=["style-tile"])
                    style_ghibli = gr.Button("Studio Ghibli", elem_id="style_ghibli", elem_classes=["style-tile"])
                    style_watercolor = gr.Button("Watercolor", elem_id="style_watercolor", elem_classes=["style-tile"])
                    gr.Markdown("**Quality Preset**", elem_id="quality_label")
                    quality_default = gr.Button("Default", elem_id="quality_default", elem_classes=["quality-tile", "selected"])
                    quality_high = gr.Button("High", elem_id="quality_high", elem_classes=["quality-tile"])
                    quality_ultra = gr.Button("Ultra", elem_id="quality_ultra", elem_classes=["quality-tile"])
                    gr.Markdown("**Beast Mode**", elem_id="beast_label")
                    beast_mode_toggle = gr.Checkbox(label="🔥 Beast Mode (Ultra Quality)", value=True, elem_id="beast_mode_toggle", elem_classes=["cupertino-toggle"])
                    motion_slider = gr.Slider(0.0, 1.0, value=0.5, step=0.01, label="Motion Energy", elem_id="motion_slider")
                    creativity_slider = gr.Slider(0.0, 1.0, value=0.7, step=0.01, label="Creativity Temperature", elem_id="creativity_slider")
                    gr.Markdown("**Aspect Ratio**", elem_id="aspect_label")
                    aspect_16_9 = gr.Button("16:9 Cinematic", elem_id="aspect_16_9", elem_classes=["aspect-toggle", "selected"])
                    aspect_9_16 = gr.Button("9:16 Vertical (TikTok)", elem_id="aspect_9_16", elem_classes=["aspect-toggle"])

            # Studio Monitor (right column)
            with gr.Column(scale=7, min_width=600):
                with gr.Group(elem_classes=["lux-glass"]):
                    gr.Markdown("## Studio Monitor")
                    video_preview = gr.Video(label="Preview Window", elem_id="quick_video_preview")
                    gr.Row(
                        gr.Markdown("<div class='status-micro-card'>Director: Planning...</div>", elem_id="status_director", show_label=False),
                        gr.Markdown("<div class='status-micro-card'>Renderer: Sampling...</div>", elem_id="status_renderer", show_label=False)
                    )
                    gr.Markdown("<b>Preview Mode:</b> Storyboard Sketch | Final 4K Render", elem_id="preview_mode_switch")
                with gr.Group(elem_classes=["lux-glass"]):
                    gr.Markdown("### Agentic Manifest & Telemetry")
                    output_info = gr.Markdown("Output info, logs, and manifest details will appear here.", elem_id="quick_output_info")
                with gr.Row():
                    quick_generate_btn = gr.Button("Generate", elem_id="quick_generate_btn", elem_classes=["gold-gradient-btn"])
                    btn_lut = gr.Button("Apply Cinematic LUT", elem_id="btn_lut")
                    btn_retry = gr.Button("Retry with New Seed", elem_id="btn_retry")
                    btn_lock_styles = gr.Button("Lock Styles", elem_id="btn_lock_styles")
                    btn_export_reels = gr.Button("Export for Reels", elem_id="btn_export_reels")

        # Event binding for Sample Project (populates theme textbox)
        def sample_project_callback():
            return "A bunny finds a golden carrot"
        sample_project_btn = gr.Button("Sample Project", elem_id="sample_project_btn")
        sample_project_btn.click(fn=sample_project_callback, outputs=theme_box)

        # Event binding for Generate button
        def quick_generate_callback(theme, duration, character, style, quality, beast_mode):
            try:
                # Call the backend quick_generate function
                return quick_generate(theme, duration, character, style, quality, beast_mode)
            except Exception as e:
                return None, f"Error: {e}", "", gr.update(interactive=True)
        quick_generate_btn.click(
            fn=quick_generate_callback,
            inputs=[theme_box, duration_slider, character_dropdown, style_pixar, quality_default, beast_mode_toggle],
            outputs=[video_preview, output_info, None, quick_generate_btn],
            queue=True,
        )

        # TODO: Bind other buttons (btn_lut, btn_retry, btn_lock_styles, btn_export_reels) to their respective backend functions

        # Footer
        gr.Markdown(
            """
            ---
            <center>
            <small>
            AniMate Studio v1.0 | AI-Assisted Animation Engine<br>
            Content is AI-generated with human creative direction.<br>
            All safety checks logged for platform compliance.
            </small>
            </center>
            """,
        )

    return app, theme, custom_css


# ═════════════════════════════════════════════════════════
# LAUNCH
# ═════════════════════════════════════════════════════════

def _kill_port(port: int = 7860):
    """Find and terminate any Python process holding the target port."""
    if not _HAS_PSUTIL:
        logger.warning(
            "psutil not installed — cannot auto-free port %d. "
            "Run: pip install psutil", port,
        )
        return
    for conn in psutil.net_connections(kind="inet"):
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            try:
                proc = psutil.Process(conn.pid)
                if proc.name().lower() not in (
                    "python.exe", "python3.exe", "pythonw.exe", "python",
                ):
                    logger.debug(
                        "Skipping non-Python process %s (PID %d) on port %d",
                        proc.name(), proc.pid, port,
                    )
                    continue
                logger.info(
                    "Terminating Python process %s (PID %d) on port %d",
                    proc.name(), proc.pid, port,
                )
                proc.terminate()
                proc.wait(timeout=5)
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.TimeoutExpired) as e:
                logger.warning("Could not kill PID %d: %s", conn.pid, e)


if __name__ == "__main__":
    app, theme, custom_css = build_ui()
    logger.info("=" * 60)
    logger.info("  AniMate Studio v1.0 — Starting...")
    logger.info("=" * 60)

    PORTS = [7860, 7861, 7862]
    _kill_port(PORTS[0])

    for port in PORTS:
        try:
            app.queue().launch(
                server_name="0.0.0.0",
                server_port=port,
                share=False,
                show_error=True,
                theme=theme,
                css=custom_css,
            )
            break
        except OSError:
            logger.warning("Port %d busy, trying next...", port)
    else:
        logger.error("All ports %s are occupied. Free a port and retry.", PORTS)
