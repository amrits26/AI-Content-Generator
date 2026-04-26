"""
═══════════════════════════════════════════════════════════════
AniMate Studio — Batch Video Generator
═══════════════════════════════════════════════════════════════
Reads themes from themes.csv, generates one video per theme,
saves progress to batch_state.json so it can resume after
interruption.  Designed for unattended overnight runs.

Usage:
    python batch_generate.py
    python batch_generate.py --csv my_themes.csv
    python batch_generate.py --cooldown 30
    python batch_generate.py --resume          # pick up from last state
═══════════════════════════════════════════════════════════════
"""

import argparse
import csv
import gc
import json
import logging
import os
import shutil
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

from engine.config import load_config

# ── Paths ────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")
DEFAULT_CSV = os.path.join(SCRIPT_DIR, "themes.csv")
STATE_FILE = os.path.join(SCRIPT_DIR, "batch_state.json")
BATCH_LOG = os.path.join(SCRIPT_DIR, "batch_log.txt")

# ── Logging ──────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(name)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(SCRIPT_DIR, "animate_studio.log"), encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("animate_studio.batch")


def _write_batch_log(message: str):
    """Append a timestamped line to batch_log.txt."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(BATCH_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


# ── Tokenizer guard (same as main.py) ───────────────────
def _setup_tokenizer():
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        logger.error("tiktoken not installed — run:  pip install tiktoken>=0.7.0")
        sys.exit(1)


# ── State management ────────────────────────────────────
def load_state() -> dict:
    """Load batch_state.json or return empty state."""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_index": -1}


def save_state(state: dict):
    """Persist progress to disk."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ── Theme loading ────────────────────────────────────────
def load_themes(csv_path: str) -> list[dict]:
    """
    Read themes.csv.  Supports two formats:

    Simple (one column):
        Billy Bunny learns to share
        Kitten finds a rainbow

    Full (columns: theme, character, duration):
        theme,character,duration
        Billy Bunny learns to share,Billy Bunny,30
        Kitten finds a rainbow,None,15
    """
    themes = []
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(1024)
        f.seek(0)
        has_header = "theme" in sample.lower().split("\n")[0]

        if "," in sample.split("\n")[0] and has_header:
            reader = csv.DictReader(f)
            for row in reader:
                themes.append({
                    "theme": row.get("theme", "").strip(),
                    "character": row.get("character", "None").strip() or "None",
                    "duration": int(row.get("duration", 30)),
                })
        else:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    themes.append({
                        "theme": line,
                        "character": "None",
                        "duration": 30,
                    })
    return themes


# ── Single video generation ──────────────────────────────
def generate_one_video(
    theme_entry: dict,
    config: dict,
    story_engine,
    animator,
    audio_engine,
    safety_filter,
    exporter,
    character_manager,
) -> dict:
    """
    Generate one video from a theme dict.

    Returns dict with keys: success, video_path, title, error, duration_s
    """
    theme = theme_entry["theme"]
    char_name = theme_entry["character"]
    duration = theme_entry["duration"]

    if char_name == "None":
        char_name = "Billy"
    profile = character_manager.get_profile(char_name)
    char_type = profile.animal_type if profile else "bunny"

    # Map duration to scene count
    if duration <= 15:
        num_scenes = 1
        scene_duration = duration
    elif duration <= 30:
        num_scenes = 2
        scene_duration = duration / num_scenes
    elif duration <= 60:
        num_scenes = 3
        scene_duration = duration / num_scenes
    else:
        num_scenes = 5
        scene_duration = duration / num_scenes

    # 1. Story generation
    logger.info("Generating storyboard for: %s", theme)
    manifest, style_lock = story_engine.generate_manifest(
        theme=theme,
        character_name=char_name,
        character_type=char_type,
        num_scenes=num_scenes,
        scene_duration=scene_duration,
    )

    # 2. Animation
    logger.info("Animating %d scenes...", len(storyboard.scenes))
    try:
        episode = animator.generate_episode(
            storyboard=storyboard,
            character_name=char_name if character_manager.get_profile(char_name) else None,
        )
    finally:
        animator.cleanup_request_state()

    if not episode.get("video_path"):
        raise RuntimeError("All scenes failed safety checks — no video produced.")

    # 3. Narration
    narration_texts = [s.narration for s in storyboard.scenes]
    audio_dir = os.path.join(config["app"]["output_dir"], "_batch_audio")
    try:
        narration = audio_engine.generate_episode_narration(narration_texts, audio_dir)
        final_video = audio_engine.apply_audio_to_video(
            video_path=episode["video_path"],
            narration_path=narration["full_narration_path"],
        )
    except Exception as e:
        logger.warning("Audio failed, using video without narration: %s", e)
        final_video = episode["video_path"]

    # 4. Safety check on narration text
    safety_result = safety_filter.scan_text(" ".join(narration_texts))

    # 5. Export for TikTok (9:16, fastest encode)
    export_result = exporter.export(
        source_video=final_video,
        platform="tiktok",
        title=storyboard.title,
        narration_texts=narration_texts,
        safety_result=safety_result,
        human_input_logged=True,
    )

    if not export_result.success:
        raise RuntimeError(f"Export failed: {export_result.error}")

    return {
        "success": True,
        "video_path": export_result.video_path,
        "title": storyboard.title,
        "moral": storyboard.moral,
        "error": "",
        "duration_s": episode.get("duration_s", duration),
    }


# ── Main batch loop ──────────────────────────────────────
def run_batch(csv_path: str, cooldown_s: int, resume: bool):
    """Execute the full batch generation loop."""
    _setup_tokenizer()

    # Load config
    config = load_config(CONFIG_PATH)

    # Load themes
    if not os.path.exists(csv_path):
        logger.error("themes CSV not found: %s", csv_path)
        sys.exit(1)

    themes = load_themes(csv_path)
    if not themes:
        logger.error("No themes found in %s", csv_path)
        sys.exit(1)

    logger.info("Loaded %d themes from %s", len(themes), csv_path)

    # Load state
    state = load_state() if resume else {"completed": [], "failed": [], "last_index": -1}
    start_index = state["last_index"] + 1 if resume else 0

    if start_index >= len(themes):
        logger.info("All %d themes already processed. Delete batch_state.json to restart.", len(themes))
        return

    logger.info(
        "Starting batch from index %d/%d (completed=%d, failed=%d)",
        start_index, len(themes), len(state["completed"]), len(state["failed"]),
    )
    _write_batch_log(f"=== Batch started: {len(themes)} themes, resuming from #{start_index} ===")

    # Initialize engines (lazy — models load on first use)
    from engine.story_engine import StoryEngine
    from engine.character_manager import CharacterManager
    from engine.animator import Animator
    from engine.audio_engine import AudioEngine
    from engine.safety_filter import SafetyFilter
    from engine.exporter import Exporter

    story_engine = StoryEngine(CONFIG_PATH, config=config)
    character_manager = CharacterManager(CONFIG_PATH, config=config)
    animator = Animator(CONFIG_PATH, fast_mode=True, config=config)
    audio_engine = AudioEngine(CONFIG_PATH, config=config)
    safety_filter = SafetyFilter(CONFIG_PATH, config=config)
    exporter = Exporter(CONFIG_PATH, config=config)

    total = len(themes)
    successes = len(state["completed"])
    failures = len(state["failed"])


    # Beast Mode: Parallel rendering
    beast_mode_cfg = config.get("beast_mode", {})
    parallel_cfg = beast_mode_cfg.get("parallel_rendering", {})
    parallel_enabled = parallel_cfg.get("enabled", False)
    max_workers = parallel_cfg.get("max_workers", 2)

    def process_entry(idx, entry):
        theme = entry["theme"]
        logger.info(
            "━━━ [%d/%d] Theme: %s ━━━", idx + 1, total, theme,
        )
        _write_batch_log(f"[{idx + 1}/{total}] START: {theme}")
        t0 = time.time()
        try:
            result = generate_one_video(
                theme_entry=entry,
                config=config,
                story_engine=story_engine,
                animator=animator,
                audio_engine=audio_engine,
                safety_filter=safety_filter,
                exporter=exporter,
                character_manager=character_manager,
            )
            elapsed = time.time() - t0
            return (idx, True, result, elapsed, None)
        except Exception as e:
            elapsed = time.time() - t0
            tb = traceback.format_exc()
            return (idx, False, str(e), elapsed, tb)

    if parallel_enabled:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        logger.info(f"Beast Mode parallel rendering enabled (max_workers={max_workers})")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_entry, idx, themes[idx]): idx for idx in range(start_index, total)}
            for future in as_completed(futures):
                idx = futures[future]
                entry = themes[idx]
                theme = entry["theme"]
                try:
                    idx, ok, result, elapsed, tb = future.result()
                    if ok:
                        successes += 1
                        state["completed"].append({
                            "index": idx,
                            "theme": theme,
                            "title": result["title"],
                            "video_path": result["video_path"],
                            "elapsed_s": round(elapsed, 1),
                            "timestamp": datetime.now().isoformat(),
                        })
                        msg = (
                            f"[{idx + 1}/{total}] OK: \"{result['title']}\" -> "
                            f"{result['video_path']} ({elapsed:.0f}s)"
                        )
                        logger.info(msg)
                        _write_batch_log(msg)
                    else:
                        failures += 1
                        state["failed"].append({
                            "index": idx,
                            "theme": theme,
                            "error": result,
                            "elapsed_s": round(elapsed, 1),
                            "timestamp": datetime.now().isoformat(),
                        })
                        msg = f"[{idx + 1}/{total}] FAIL: {theme} — {result}"
                        logger.error(msg)
                        logger.debug(tb)
                        _write_batch_log(msg)
                    state["last_index"] = idx
                    save_state(state)
                    # Progress summary
                    remaining = total - idx - 1
                    logger.info(
                        "Progress: %d OK / %d FAIL / %d remaining",
                        successes, failures, remaining,
                    )
                except Exception as e:
                    logger.error(f"Exception in parallel batch: {e}")
    else:
        for idx in range(start_index, total):
            entry = themes[idx]
            theme = entry["theme"]
            logger.info(
                "━━━ [%d/%d] Theme: %s ━━━", idx + 1, total, theme,
            )
            _write_batch_log(f"[{idx + 1}/{total}] START: {theme}")
            t0 = time.time()
            try:
                result = generate_one_video(
                    theme_entry=entry,
                    config=config,
                    story_engine=story_engine,
                    animator=animator,
                    audio_engine=audio_engine,
                    safety_filter=safety_filter,
                    exporter=exporter,
                    character_manager=character_manager,
                )
                elapsed = time.time() - t0
                successes += 1
                state["completed"].append({
                    "index": idx,
                    "theme": theme,
                    "title": result["title"],
                    "video_path": result["video_path"],
                    "elapsed_s": round(elapsed, 1),
                    "timestamp": datetime.now().isoformat(),
                })
                state["last_index"] = idx
                save_state(state)
                msg = (
                    f"[{idx + 1}/{total}] OK: \"{result['title']}\" -> "
                    f"{result['video_path']} ({elapsed:.0f}s)"
                )
                logger.info(msg)
                _write_batch_log(msg)
            except Exception as e:
                elapsed = time.time() - t0
                failures += 1
                error_msg = str(e)
                tb = traceback.format_exc()
                state["failed"].append({
                    "index": idx,
                    "theme": theme,
                    "error": error_msg,
                    "elapsed_s": round(elapsed, 1),
                    "timestamp": datetime.now().isoformat(),
                })
                state["last_index"] = idx
                save_state(state)
                msg = f"[{idx + 1}/{total}] FAIL: {theme} — {error_msg}"
                logger.error(msg)
                logger.debug(tb)
                _write_batch_log(msg)
            # Progress summary
            remaining = total - idx - 1
            logger.info(
                "Progress: %d OK / %d FAIL / %d remaining",
                successes, failures, remaining,
            )
            # Free VRAM between videos
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            # Cooldown between videos (let GPU cool)
            if remaining > 0:
                logger.info("Cooling down for %ds before next video...", cooldown_s)
                time.sleep(cooldown_s)

    # Final summary
    summary = (
        f"=== Batch complete: {successes} OK / {failures} FAIL / "
        f"{total} total ==="
    )
    logger.info(summary)
    _write_batch_log(summary)
    print(f"\n{'=' * 60}")
    print(f"  BATCH COMPLETE")
    print(f"  Successes: {successes}")
    print(f"  Failures:  {failures}")
    print(f"  Total:     {total}")
    print(f"  Log:       {BATCH_LOG}")
    print(f"  State:     {STATE_FILE}")
    print(f"{'=' * 60}\n")


# ── CLI ──────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="AniMate Studio — Batch Video Generator",
    )
    parser.add_argument(
        "--csv", default=DEFAULT_CSV,
        help="Path to themes CSV file (default: themes.csv)",
    )
    parser.add_argument(
        "--cooldown", type=int, default=60,
        help="Seconds to wait between videos for GPU cooling (default: 60)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last saved state in batch_state.json",
    )
    args = parser.parse_args()

    run_batch(
        csv_path=args.csv,
        cooldown_s=args.cooldown,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
