"""Quick smoke test: generate one 16-frame AnimateDiff clip and save as mp4."""
import os, sys, time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

from engine.animate_diff_engine import AnimateDiffEngine
from utils.ffmpeg_utils import frames_to_video

print("Loading AnimateDiff engine...")
t0 = time.time()
eng = AnimateDiffEngine("config.yaml")
ok = eng.load()
print(f"Load result: {ok}  ({time.time()-t0:.1f}s)")
if not ok:
    print(f"Error: {eng._load_error}")
    sys.exit(1)

prompt = (
    "cute cartoon children playing in a sunny meadow, colorful, vibrant, "
    "kids animation style, gentle breeze, butterflies flying"
)
negative = "blurry, dark, scary, realistic, photographic, nsfw"

print("Generating 16-frame clip...")
t1 = time.time()
frames = eng.generate_clip(prompt=prompt, negative_prompt=negative, seed=42)
print(f"Got {len(frames)} frames in {time.time()-t1:.1f}s")

if frames:
    out_dir = os.path.join("output", "_smoke_test")
    os.makedirs(out_dir, exist_ok=True)
    for i, f in enumerate(frames):
        f.save(os.path.join(out_dir, f"frame_{i:04d}.png"))
    vid = frames_to_video(frames, os.path.join(out_dir, "smoke_test.mp4"), fps=8)
    print(f"Video saved: {vid}")
    print("SMOKE TEST PASSED")
else:
    print("SMOKE TEST FAILED — no frames generated")
    sys.exit(1)
