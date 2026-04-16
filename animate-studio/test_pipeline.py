"""
AniMate Studio — Pipeline Integration Test
Tests each step: story → animation → audio → export
"""
import os
import sys
import json
import time
import traceback

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")

def test_step(name, fn):
    print(f"\n{'='*60}")
    print(f"  STEP: {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        result = fn()
        elapsed = time.time() - t0
        print(f"  PASS ({elapsed:.1f}s)")
        return result
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL ({elapsed:.1f}s): {e}")
        traceback.print_exc()
        return None

# ── Step 1: Story Engine ──────────────────────────────────
def test_story():
    from engine.story_engine import StoryEngine
    se = StoryEngine(CONFIG_PATH)
    sb = se.generate_storyboard(
        theme="Billy Bunny learns to share his carrots",
        character_name="Billy",
        character_type="bunny",
        num_scenes=3,
        scene_duration=4.0,
    )
    print(f"  Title: {sb.title}")
    print(f"  Moral: {sb.moral}")
    print(f"  Scenes: {len(sb.scenes)}")
    for s in sb.scenes:
        narr = s.narration[:60] if isinstance(s.narration, str) else str(s.narration)[:60]
        vis = s.visual_description
        if isinstance(vis, dict):
            vis = vis.get("description", str(vis))
        vis = str(vis)[:60]
        print(f"    Scene {s.scene_id}: narr={narr}...")
        print(f"                   vis={vis}...")
    # Ensure visual_description is a string for downstream
    for s in sb.scenes:
        if isinstance(s.visual_description, dict):
            s.visual_description = s.visual_description.get("description", json.dumps(s.visual_description))
        if isinstance(s.narration, dict):
            s.narration = s.narration.get("text", json.dumps(s.narration))
    return sb

# ── Step 2: Animator (CogVideoX pipeline load) ────────────
def test_pipeline_load():
    from engine.animator import Animator
    anim = Animator(CONFIG_PATH)
    anim.load_pipeline()
    print(f"  Pipeline type: {type(anim._pipeline).__name__}")
    print(f"  Device: {anim.device}")
    return anim

# ── Step 3: Generate one scene ────────────────────────────
def test_generate_scene(anim, storyboard):
    if not anim or not storyboard:
        raise RuntimeError("Prerequisite step failed")
    scene = storyboard.scenes[0]
    # Ensure visual_description is string
    if isinstance(scene.visual_description, dict):
        scene.visual_description = scene.visual_description.get("description", str(scene.visual_description))
    result = anim.generate_scene(scene=scene, seed=42)
    print(f"  Video path: {result.get('video_path', 'NONE')}")
    print(f"  Safety passed: {result.get('safety_passed', 'N/A')}")
    print(f"  Frame count: {len(result.get('frames', []))}")
    return result

# ── Step 4: Audio (Edge TTS) ─────────────────────────────
def test_audio():
    from engine.audio_engine import AudioEngine
    ae = AudioEngine(CONFIG_PATH)
    test_dir = os.path.join(SCRIPT_DIR, "output", "_test_audio")
    os.makedirs(test_dir, exist_ok=True)
    out_path = os.path.join(test_dir, "test_narration.wav")
    ae.generate_narration("Billy Bunny learned to share his carrots with his friends.", out_path)
    exists = os.path.exists(out_path)
    size = os.path.getsize(out_path) if exists else 0
    print(f"  Audio file: {out_path}")
    print(f"  Exists: {exists}, Size: {size} bytes")
    return out_path if exists else None

# ── Step 5: FFmpeg availability ───────────────────────────
def test_ffmpeg():
    from utils.ffmpeg_utils import get_media_duration
    import subprocess
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    version_line = result.stdout.split("\n")[0] if result.returncode == 0 else "NOT FOUND"
    print(f"  FFmpeg: {version_line}")
    return result.returncode == 0

# ── Run all tests ─────────────────────────────────────────
if __name__ == "__main__":
    print("\nAniMate Studio — Pipeline Integration Test")
    print("=" * 60)

    ffmpeg_ok = test_step("FFmpeg availability", test_ffmpeg)
    storyboard = test_step("Story Engine (Ollama)", test_story)
    audio_path = test_step("Audio Engine (Edge TTS)", test_audio)
    animator = test_step("CogVideoX Pipeline Load", test_pipeline_load)
    
    if animator and storyboard:
        scene_result = test_step("Generate One Scene", lambda: test_generate_scene(animator, storyboard))
    else:
        print("\n  SKIP: Scene generation (prerequisite failed)")
        scene_result = None

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  FFmpeg:     {'PASS' if ffmpeg_ok else 'FAIL'}")
    print(f"  Story:      {'PASS' if storyboard else 'FAIL'}")
    print(f"  Audio:      {'PASS' if audio_path else 'FAIL'}")
    print(f"  Pipeline:   {'PASS' if animator else 'FAIL'}")
    print(f"  Scene Gen:  {'PASS' if scene_result and scene_result.get('video_path') else 'FAIL'}")
    
    if scene_result and scene_result.get("video_path"):
        print(f"\n  Generated video: {scene_result['video_path']}")
    print()
