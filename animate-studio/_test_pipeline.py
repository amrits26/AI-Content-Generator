"""Quick end-to-end pipeline test for AniMate Studio."""
import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── STEP 1: Story Generation ──────────────────────────────
print("=== STEP 1: Story Generation ===")
from engine.story_engine import StoryEngine

se = StoryEngine("config.yaml")
try:
    sb = se.generate_storyboard(
        theme="Billy Bunny learns to share his carrots",
        character_name="Billy Bunny",
        character_type="bunny",
        num_scenes=2,
        scene_duration=4.0,
    )
    print(f"  Title: {sb.title}")
    print(f"  Moral: {sb.moral}")
    print(f"  Scenes: {len(sb.scenes)}")
    for s in sb.scenes:
        print(f"    Scene {s.scene_id}: {str(s.narration)[:80]}")
        print(f"      Visual: {str(s.visual_description)[:80]}")
        print(f"      Types: narration={type(s.narration).__name__}, visual={type(s.visual_description).__name__}")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()
    sys.exit(1)


# ── STEP 2: Edge TTS Audio ───────────────────────────────
print("\n=== STEP 2: Edge TTS Audio ===")
from engine.audio_engine import AudioEngine

audio = AudioEngine("config.yaml")
os.makedirs("output", exist_ok=True)
test_audio = os.path.join("output", "_test_narration.wav")
try:
    audio.generate_narration("Billy Bunny loves to share.", test_audio)
    from utils.ffmpeg_utils import get_media_duration
    dur = get_media_duration(test_audio)
    print(f"  Audio: {test_audio} ({dur:.1f}s)")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()


# ── STEP 3: Animation (single scene) ─────────────────────
print("\n=== STEP 3: Animation (1 scene — CogVideoX) ===")
from engine.animator import Animator

anim = Animator("config.yaml")
try:
    scene = sb.scenes[0]
    result = anim.generate_scene(
        scene=scene,
        character_prompt="fluffy blue bunny with big round sparkling eyes",
        seed=42,
    )
    vp = result.get("video_path", "NONE")
    sp = result.get("safety_passed", "?")
    nf = len(result.get("frames", []))
    print(f"  Video path: {vp}")
    print(f"  Safety passed: {sp}")
    print(f"  Frames: {nf}")
    if vp and os.path.exists(vp):
        print(f"  File size: {os.path.getsize(vp) / 1024:.1f} KB")
except Exception as e:
    print(f"  FAILED: {e}")
    traceback.print_exc()

print("\n=== ALL TESTS COMPLETE ===")
