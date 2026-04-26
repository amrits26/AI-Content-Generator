"""
AniMate Studio — End-to-End Video Generation Test
Bypasses Gradio UI to generate a complete video directly.
"""
import os
import sys
import json
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")

def main():
    t_total = time.time()

    # ── Step 1: Story ─────────────────────────────────
    print("\n[1/4] Generating story via Ollama...")
    t0 = time.time()
    from engine.story_engine import StoryEngine, StoryParseError
    se = StoryEngine(CONFIG_PATH)
    fallback_manifest_path = os.path.join(SCRIPT_DIR, "output", "_smoke_test", "fallback_storyboard.json")
    try:
        manifest, style_lock = se.generate_manifest(
            theme="Billy Bunny learns to share his carrots with friends",
            character_name="Billy",
            character_type="bunny",
            num_scenes=3,
            scene_duration=5.0,
        )
        print(f"  Title: {manifest['title']}")
        print(f"  Scenes: {len(manifest['scenes'])}")
        for s in manifest['scenes']:
            print(f"    Scene {s['scene_id']}: {str(s.get('action', ''))[:60]}...")
        print(f"  Done ({time.time()-t0:.1f}s)")
        # Save fallback for future runs
        os.makedirs(os.path.dirname(fallback_manifest_path), exist_ok=True)
        with open(fallback_manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
    except (StoryParseError, Exception) as e:
        print(f"  LLM offline or failed: {e}\n  Using fallback storyboard...")
        if os.path.exists(fallback_manifest_path):
            with open(fallback_manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            style_lock = manifest.get("character", {}).get("style_lock", {})
            print(f"  Fallback Title: {manifest['title']}")
            print(f"  Scenes: {len(manifest['scenes'])}")
        else:
            print("  No fallback storyboard found. Exiting.")
            return

    # ── Step 2: Animation ─────────────────────────────
    print("\n[2/4] Loading video pipeline & generating scenes...")
    t0 = time.time()
    from engine.animator import Animator
    anim = Animator(CONFIG_PATH)

    def progress_cb(idx, total, msg):
        print(f"  [{idx}/{total}] {msg}")

    episode = anim.generate_episode(
        storyboard=storyboard,
        character_name="kids_style",  # Use trained LoRA for style
        progress_callback=progress_cb,
    )

    video_path = episode.get("video_path", "")
    print(f"  Video: {video_path}")
    print(f"  Done ({time.time()-t0:.1f}s)")

    if not video_path or not os.path.exists(video_path):
        print("\n  FAILED: No video generated!")
        if episode.get("error"):
            print(f"  Error: {episode['error']}")
        return

    # ── Step 3: Audio narration ───────────────────────
    print("\n[3/4] Generating narration audio (Edge TTS)...")
    t0 = time.time()
    from engine.audio_engine import AudioEngine
    ae = AudioEngine(CONFIG_PATH)

    narration_texts = [s.narration for s in storyboard.scenes]
    audio_dir = os.path.join(SCRIPT_DIR, "output", "_e2e_audio")
    os.makedirs(audio_dir, exist_ok=True)

    try:
        narration = ae.generate_episode_narration(narration_texts, audio_dir)
        narration_path = narration.get("full_narration_path", "")
        print(f"  Narration: {narration_path}")

        if narration_path and os.path.exists(narration_path):
            final_video = ae.apply_audio_to_video(
                video_path=video_path,
                narration_path=narration_path,
            )
            print(f"  Final video with audio: {final_video}")
            video_path = final_video
        else:
            print("  No narration audio generated, using video without audio")
    except Exception as e:
        print(f"  Audio failed (using video without audio): {e}")
        traceback.print_exc()

    print(f"  Done ({time.time()-t0:.1f}s)")

    # ── Step 4: Result ────────────────────────────────
    total_time = time.time() - t_total
    file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0

    print(f"\n{'='*60}")
    print(f"  VIDEO GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  File: {os.path.abspath(video_path)}")
    print(f"  Size: {file_size / 1024:.1f} KB")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'='*60}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
