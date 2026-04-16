"""
═══════════════════════════════════════════════════════════════
AniMate Studio — Main Application (Gradio UI)
═══════════════════════════════════════════════════════════════
Pastel-themed Gradio interface with 3 tabs:
  Tab 1: Quick Story (TikTok/Reels Hook)
  Tab 2: Full Episode (YouTube Kids)
  Tab 3: Character Lab

Launch: python main.py
═══════════════════════════════════════════════════════════════
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
import pandas as pd
import yaml

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# ── Setup logging ────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("animate_studio.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("animate_studio")

# ── Load config ──────────────────────────────────────────
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

# ── Ollama model check ───────────────────────────────────
def _check_ollama_model():
    """Verify the configured Ollama model is available locally."""
    story_cfg = CONFIG.get("models", {}).get("story", {})
    if story_cfg.get("provider") not in ("ollama", "local"):
        return
    model_name = story_cfg.get("model_name", "llama3.2")
    try:
        import subprocess
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and model_name in result.stdout:
            logger.info("Ollama model '%s' found.", model_name)
        else:
            logger.warning(
                "Ollama model '%s' not found locally. "
                "Run:  ollama pull %s",
                model_name, model_name,
            )
    except FileNotFoundError:
        logger.warning(
            "Ollama is not installed or not in PATH. "
            "Install from https://ollama.com then run:  ollama pull %s",
            model_name,
        )
    except Exception as e:
        logger.warning("Ollama check failed: %s", e)

_check_ollama_model()


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
from engine.story_engine import StoryEngine, Storyboard
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

# ── Initialize engines (lazy — heavy models load on first use) ─
story_engine = StoryEngine(CONFIG_PATH)
character_manager = CharacterManager(CONFIG_PATH)
animator = Animator(CONFIG_PATH)
audio_engine = AudioEngine(CONFIG_PATH)
safety_filter = SafetyFilter(CONFIG_PATH)
exporter = Exporter(CONFIG_PATH)

# ── Human input tracking for monetization compliance ─────
_human_inputs_log = []


def log_human_input(action: str):
    """Log a human creative decision for monetization compliance."""
    entry = {"action": action, "timestamp": datetime.now().isoformat()}
    _human_inputs_log.append(entry)
    if len(_human_inputs_log) > 1000:
        del _human_inputs_log[:-1000]
    logger.info("Human input logged: %s", action)


# ═════════════════════════════════════════════════════════
# TAB 1: QUICK STORY (TikTok / Reels Hook)
# ═════════════════════════════════════════════════════════

def quick_generate(theme, duration, character_choice, style_choice, quality_preset, progress=gr.Progress()):
    """Generate a quick 15-60s hook for TikTok/Reels."""
    log_human_input(f"Quick story theme: {theme}")
    log_human_input(f"Duration choice: {duration}s")
    log_human_input(f"Character: {character_choice}")
    log_human_input(f"Style: {style_choice}, Quality: {quality_preset}")

    progress(0.05, desc="Writing story...")

    # Map duration to scene count
    if duration <= 15:
        num_scenes = 2
    elif duration <= 30:
        num_scenes = 3
    else:
        num_scenes = 5

    scene_duration = duration / num_scenes

    # Determine character
    char_name = character_choice if character_choice != "None" else "Billy"
    profile = character_manager.get_profile(char_name)
    char_type = profile.animal_type if profile else "bunny"

    # Generate story
    try:
        storyboard = story_engine.generate_storyboard(
            theme=theme,
            character_name=char_name,
            character_type=char_type,
            num_scenes=num_scenes,
            scene_duration=scene_duration,
        )
    except Exception as e:
        return None, f"Story generation failed: {e}", ""

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
    except Exception as e:
        return None, f"Animation failed: {e}", ""

    if not episode.get("video_path"):
        return None, "No video generated — all scenes failed safety.", ""

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

    return final_video, story_info, json.dumps(story_engine.storyboard_to_dict(storyboard), indent=2)


def export_for_reels(video_path, story_json):
    """Quick export optimized for Facebook Reels."""
    if not video_path:
        return "No video to export."

    story_data = json.loads(story_json) if story_json else {}
    narration_texts = [s["narration"] for s in story_data.get("scenes", [])]
    title = story_data.get("title", "Quick_Hook")

    # Run safety
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


# ═════════════════════════════════════════════════════════
# TAB 2: FULL EPISODE (YouTube Kids) — Two-step workflow
# ═════════════════════════════════════════════════════════

def generate_storyboard_preview(
    story_concept, num_scenes, character_choice, progress=gr.Progress()
):
    """Step 1: Generate storyboard and return an editable Dataframe + JSON state."""
    log_human_input(f"Storyboard concept: {story_concept}")
    progress(0.1, desc="Crafting storyboard...")

    char_name = character_choice if character_choice != "None" else "Billy"
    profile = character_manager.get_profile(char_name)
    char_type = profile.animal_type if profile else "bunny"
    target_duration = CONFIG.get("motion", {}).get("final_duration_s", 60.0)
    scene_duration = target_duration / num_scenes

    try:
        storyboard = story_engine.generate_storyboard(
            theme=story_concept,
            character_name=char_name,
            character_type=char_type,
            num_scenes=int(num_scenes),
            scene_duration=scene_duration,
        )
    except Exception as e:
        empty_df = pd.DataFrame(columns=["Scene", "Narration", "Visual Description", "Emotion", "Setting"])
        return empty_df, "{}", f"**Story generation failed:** {e}"

    progress(1.0, desc="Storyboard ready!")

    # Build editable Dataframe rows
    rows = []
    for s in storyboard.scenes:
        rows.append({
            "Scene": s.scene_id,
            "Narration": s.narration,
            "Visual Description": s.visual_description,
            "Emotion": s.emotion_tone,
            "Setting": s.setting,
        })
    df = pd.DataFrame(rows)

    # Store full storyboard as JSON state for Step 2
    sb_state = story_engine.storyboard_to_dict(storyboard)
    sb_json = json.dumps(sb_state, indent=2)

    info = f"# {storyboard.title}\n**Moral:** {storyboard.moral}\n\n"
    info += f"**{len(storyboard.scenes)} scenes** — review & edit below, then click **Render Episode**."

    return df, sb_json, info


def render_episode_from_storyboard(
    storyboard_df, storyboard_json,
    voice_choice, bgm_choice, character_choice,
    style_choice, quality_preset, camera_motion,
    lora_strength, progress=gr.Progress()
):
    """Step 2: Render the (possibly edited) storyboard into a full episode."""
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
        return None, None, "**Error:** No storyboard to render. Generate one first.", None

    from engine.story_engine import Scene, Storyboard

    char_name = character_choice if character_choice != "None" else "Billy"
    profile = character_manager.get_profile(char_name)
    char_type = profile.animal_type if profile else "bunny"
    target_duration = CONFIG.get("motion", {}).get("final_duration_s", 60.0)
    num_scenes = len(sb_dict.get("scenes", []))
    scene_duration = target_duration / max(num_scenes, 1)

    # Apply edits from the Dataframe back onto the storyboard
    edited_df = storyboard_df if isinstance(storyboard_df, pd.DataFrame) else pd.DataFrame(storyboard_df)
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
    except Exception as e:
        return None, None, f"Animation failed: {e}", None

    if not episode.get("video_path"):
        return None, None, "All scenes failed safety.", None

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

    return final_video, thumbnail, story_info, caption_file


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


def preview_character(name):
    """Show character info and prompt preview."""
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


def get_lora_list():
    """Get available LoRAs for display."""
    loras = character_manager.list_available_loras()
    if not loras:
        return "No LoRA files found in `./loras/`. See setup guide for training instructions."
    text = "**Available LoRAs:**\n\n"
    for l in loras:
        text += f"- `{l['name']}` ({l['size_mb']} MB) — Profile: {'✅' if l['has_profile'] else '❌'}\n"
    return text


# ═════════════════════════════════════════════════════════
# BUILD GRADIO UI
# ═════════════════════════════════════════════════════════

def build_ui() -> gr.Blocks:
    """Construct the full Gradio interface with pastel kid-friendly theme."""

    # Custom pastel theme
    theme = gr.themes.Soft(
        primary_hue=gr.themes.colors.pink,
        secondary_hue=gr.themes.colors.purple,
        neutral_hue=gr.themes.colors.gray,
        font=gr.themes.GoogleFont("Nunito"),
    ).set(
        body_background_fill="linear-gradient(135deg, #fce4ec 0%, #e1f5fe 50%, #f3e5f5 100%)",
        block_background_fill="rgba(255, 255, 255, 0.85)",
        block_border_width="0px",
        block_shadow="0 4px 20px rgba(0,0,0,0.08)",
        block_radius="16px",
        button_primary_background_fill="linear-gradient(135deg, #f48fb1 0%, #ce93d8 100%)",
        button_primary_text_color="white",
        input_radius="12px",
    )

    custom_css = """
        .gradio-container { max-width: 1200px !important; }
        h1 { text-align: center; color: #ad1457; }
        .gr-button { border-radius: 12px !important; }
    """

    with gr.Blocks(title="🎬 AniMate Studio") as app:
        gr.Markdown(
            """
            # 🎬 AniMate Studio
            ### ✨ AI-Powered Kids Animation Engine — Create Monetizable Content
            """
        )

        with gr.Tabs():
            # ── Tab 1: Quick Story ───────────────────────
            with gr.Tab("🚀 Quick Story (Reels/TikTok)", id="quick"):
                gr.Markdown("*Generate a quick 15-60s animated hook for social platforms*")

                with gr.Row():
                    with gr.Column(scale=2):
                        quick_theme = gr.Textbox(
                            label="Story Theme",
                            placeholder="e.g., Billy Bunny learns to share his carrot",
                            lines=2,
                        )
                        quick_duration = gr.Slider(
                            minimum=15, maximum=60, value=30, step=15,
                            label="Duration (seconds)",
                        )
                        quick_char = gr.Dropdown(
                            choices=get_character_dropdown_choices(),
                            value="None",
                            label="Character",
                        )
                        with gr.Row():
                            quick_style = gr.Dropdown(
                                choices=["default"] + list(STYLE_PRESETS.keys()),
                                value="default",
                                label="Style",
                            )
                            quick_quality = gr.Dropdown(
                                choices=["default"] + list(CONFIG.get("quality_presets", {}).keys()),
                                value="default",
                                label="Quality Preset",
                            )
                        with gr.Row():
                            quick_btn = gr.Button("🎬 Generate Hook", variant="primary", size="lg")
                            quick_export_btn = gr.Button("📱 Export for Reels", variant="secondary")

                    with gr.Column(scale=3):
                        quick_video = gr.Video(label="Preview")
                        quick_info = gr.Markdown(label="Story Info")
                        quick_story_json = gr.Textbox(visible=False)

                quick_btn.click(
                    fn=quick_generate,
                    inputs=[quick_theme, quick_duration, quick_char, quick_style, quick_quality],
                    outputs=[quick_video, quick_info, quick_story_json],
                )
                quick_export_btn.click(
                    fn=export_for_reels,
                    inputs=[quick_video, quick_story_json],
                    outputs=[quick_info],
                )

            # ── Tab 2: Full Episode ──────────────────────
            with gr.Tab("🎥 Full Episode (YouTube Kids)", id="episode"):
                gr.Markdown("*Create a full 60-90s animated episode with narration & music*")

                with gr.Row():
                    # ── Left column: inputs ──
                    with gr.Column(scale=2):
                        ep_concept = gr.Textbox(
                            label="Story Concept",
                            placeholder="Write a detailed concept...\ne.g., Billy Bunny finds a lost duckling in the meadow and helps it find its way home.",
                            lines=4,
                        )
                        ep_scenes = gr.Slider(
                            minimum=3, maximum=8, value=5, step=1,
                            label="Number of Scenes",
                        )
                        ep_char = gr.Dropdown(
                            choices=get_character_dropdown_choices(),
                            value="None",
                            label="Main Character",
                        )
                        ep_storyboard_btn = gr.Button(
                            "📝 Generate Storyboard", variant="secondary", size="lg",
                        )

                        # ── Storyboard review area ──
                        ep_storyboard_info = gr.Markdown(
                            label="Storyboard Preview",
                            value="*Click 'Generate Storyboard' to create an editable scene plan.*",
                        )
                        ep_storyboard_df = gr.Dataframe(
                            headers=["Scene", "Narration", "Visual Description", "Emotion", "Setting"],
                            datatype=["number", "str", "str", "str", "str"],
                            interactive=True,
                            label="Editable Storyboard — tweak scenes before rendering",
                            col_count=(5, "fixed"),
                            wrap=True,
                        )
                        ep_storyboard_state = gr.Textbox(visible=False, value="{}")

                        gr.Markdown("---")
                        gr.Markdown("**Rendering Options**")
                        ep_voice = gr.Dropdown(
                            choices=[(v["name"], v["id"]) for v in audio_engine.list_voices()],
                            label="Narrator Voice",
                            value=audio_engine.default_voice_id,
                        )
                        ep_bgm = gr.Dropdown(
                            choices=["None"] + [t["name"] for t in audio_engine.list_bgm_tracks()],
                            value="None",
                            label="Background Music",
                        )
                        with gr.Row():
                            ep_style = gr.Dropdown(
                                choices=["default"] + list(STYLE_PRESETS.keys()),
                                value="default",
                                label="Style",
                            )
                            ep_quality = gr.Dropdown(
                                choices=["default"] + list(CONFIG.get("quality_presets", {}).keys()),
                                value="default",
                                label="Quality Preset",
                            )
                        with gr.Row():
                            ep_camera = gr.Dropdown(
                                choices=["default", "auto"] + list(CAMERA_MOTIONS.keys()),
                                value="default",
                                label="Camera Motion",
                            )
                            ep_lora_strength = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.8, step=0.05,
                                label="LoRA Strength",
                            )
                        ep_render_btn = gr.Button(
                            "🎬 Render Episode", variant="primary", size="lg",
                        )

                    # ── Right column: outputs ──
                    with gr.Column(scale=3):
                        ep_video = gr.Video(label="Episode Preview")
                        with gr.Row():
                            ep_thumbnail = gr.Image(label="Thumbnail", type="filepath")
                            ep_captions = gr.File(label="Caption File (.srt)")
                        ep_info = gr.Markdown(label="Episode Details & Compliance")

                # ── Step 1 wiring ──
                ep_storyboard_btn.click(
                    fn=generate_storyboard_preview,
                    inputs=[ep_concept, ep_scenes, ep_char],
                    outputs=[ep_storyboard_df, ep_storyboard_state, ep_storyboard_info],
                )
                # ── Step 2 wiring ──
                ep_render_btn.click(
                    fn=render_episode_from_storyboard,
                    inputs=[ep_storyboard_df, ep_storyboard_state,
                            ep_voice, ep_bgm, ep_char,
                            ep_style, ep_quality, ep_camera, ep_lora_strength],
                    outputs=[ep_video, ep_thumbnail, ep_info, ep_captions],
                )

            # ── Tab 3: Character Lab ─────────────────────
            with gr.Tab("🧸 Character Lab", id="characters"):
                gr.Markdown("*Design and manage consistent characters for your stories*")

                with gr.Row():
                    # Create Character
                    with gr.Column():
                        gr.Markdown("### Create Character")
                        char_name = gr.Textbox(label="Character Name", placeholder="Billy")
                        char_type = gr.Dropdown(
                            choices=list(CHARACTER_TEMPLATES.keys()),
                            value="bunny",
                            label="Animal Type",
                        )
                        char_color = gr.Textbox(label="Primary Color", placeholder="soft blue", value="soft blue")
                        char_accessory = gr.Dropdown(
                            choices=ACCESSORIES,
                            value="red bowtie",
                            label="Accessory",
                        )
                        char_traits = gr.Textbox(
                            label="Traits (comma-separated)",
                            value="friendly, curious, kind",
                        )
                        char_ref_img = gr.Image(
                            label="Reference Image (optional)",
                            type="filepath",
                        )
                        char_save_btn = gr.Button("💾 Save Character", variant="primary")
                        char_save_result = gr.Markdown()

                    # Preview Character
                    with gr.Column():
                        gr.Markdown("### Preview Character")
                        preview_name = gr.Dropdown(
                            choices=get_character_dropdown_choices(),
                            label="Select Character",
                        )
                        preview_btn = gr.Button("🔍 Preview")
                        preview_info = gr.Markdown()
                        preview_img = gr.Image(label="Reference Image", type="filepath")

                        gr.Markdown("---")
                        lora_info = gr.Markdown(value=get_lora_list())
                        refresh_btn = gr.Button("🔄 Refresh")

                char_save_btn.click(
                    fn=save_character,
                    inputs=[char_name, char_type, char_color, char_accessory, char_traits, char_ref_img],
                    outputs=[char_save_result],
                )
                preview_btn.click(
                    fn=preview_character,
                    inputs=[preview_name],
                    outputs=[preview_info, preview_img],
                )
                refresh_btn.click(
                    fn=lambda: (get_character_dropdown_choices(), get_lora_list()),
                    outputs=[preview_name, lora_info],
                )

        # ── Footer ───────────────────────────────────────
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
            app.launch(
                server_name="127.0.0.1",
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
