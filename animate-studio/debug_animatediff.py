"""Debug: trace single scene through AnimateDiff path."""
import logging, os, sys, time

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from engine.story_engine import Scene
from engine.animator import Animator

scene = Scene(
    scene_id=99,
    narration="Test scene",
    visual_description="cute cartoon bunny in meadow, bright colors, butterflies",
    emotion_tone="happy",
    setting="meadow",
    duration_s=4.0,
)

a = Animator("config.yaml")
print(f"AnimateDiff enabled: {a.use_animatediff}")
print(f"AnimateDiff engine: {a._animatediff}")

t0 = time.time()
result = a.generate_scene(scene=scene, character_name="kids_style", character_prompt="", seed=12345)
elapsed = time.time() - t0
print(f"Time: {elapsed:.1f}s")
print(f"Video: {result.get('video_path', 'NONE')}")
print(f"Frames: {len(result.get('frames', []))}")
print(f"Keys: {list(result.keys())}")
