"""
AniMate Studio — 60s Production E2E Run
Creates a handcrafted, richer storyboard and renders with kids_style LoRA.
Ensures final output duration is ~60s by matching video loop length to narration.
"""

import os
import time
import traceback

from engine.story_engine import Scene, Storyboard

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
CONFIG_PATH = os.path.join(SCRIPT_DIR, "config.yaml")


def build_storyboard_60s() -> Storyboard:
    title = "Billy Bunny and the Great Carrot Share"
    moral = "Sharing turns little worries into big happiness."

    # Six detailed scenes with warm emotional arc and cinematic visuals.
    # visual_descriptions kept compact to stay under 77 CLIP tokens.
    scenes = [
        Scene(
            scene_id=1,
            narration=(
                "Morning sunlight spills across Clover Meadow as Billy Bunny hops out with a basket of bright orange carrots. "
                "He beams with pride, dreaming of the perfect picnic all to himself."
            ),
            visual_description=(
                "fluffy blue bunny with red bowtie holding carrot basket, "
                "sunny meadow, golden morning light, butterflies, soft depth of field"
            ),
            emotion_tone="happy",
            setting="meadow",
            duration_s=10.0,
        ),
        Scene(
            scene_id=2,
            narration=(
                "Soon, his friends gather nearby: a tiny fox, a gentle lamb, and a curious duckling. "
                "They wave and smile, but Billy hugs his basket tightly, unsure if he should share."
            ),
            visual_description=(
                "group of cartoon animals, fox lamb duckling around blue bunny, "
                "bunny clutching basket, expressive eyes, flower meadow"
            ),
            emotion_tone="curious",
            setting="meadow",
            duration_s=10.0,
        ),
        Scene(
            scene_id=3,
            narration=(
                "A playful gust twirls through the field and sends carrots rolling down a little hill. "
                "Billy gasps and scrambles after them, while his friends dash in to help without hesitation."
            ),
            visual_description=(
                "carrots rolling downhill, bunny chasing, friends running to help, "
                "windy meadow, floating petals, dynamic action"
            ),
            emotion_tone="surprised",
            setting="meadow",
            duration_s=10.0,
        ),
        Scene(
            scene_id=4,
            narration=(
                "Together they gather every last carrot, laughing as they tumble into a soft patch of clover. "
                "Billy realizes his happiest moment came from everyone working together."
            ),
            visual_description=(
                "cartoon animals laughing together in clover patch, "
                "warm rim light, cozy pastel colors, happy faces"
            ),
            emotion_tone="happy",
            setting="meadow",
            duration_s=10.0,
        ),
        Scene(
            scene_id=5,
            narration=(
                "At picnic time, Billy takes a deep breath, then proudly offers carrots to each friend. "
                "Their eyes light up, and the meadow fills with cheerful chatter and tiny happy dances."
            ),
            visual_description=(
                "picnic blanket scene, bunny sharing carrots with friends, "
                "joyful character poses, bright colorful food, sunny meadow"
            ),
            emotion_tone="happy",
            setting="meadow",
            duration_s=10.0,
        ),
        Scene(
            scene_id=6,
            narration=(
                "As the sky turns peach and lavender, they sit in a circle sharing the last sweet carrots. "
                "Billy smiles at his friends and whispers, sharing made everything taste better."
            ),
            visual_description=(
                "sunset scene, cartoon animals sitting in circle, "
                "peach and lavender sky, warm glow, storybook ending"
            ),
            emotion_tone="peaceful",
            setting="meadow",
            duration_s=10.0,
        ),
    ]

    return Storyboard(
        title=title,
        moral=moral,
        scenes=scenes,
        character_name="Billy",
        character_type="bunny",
        theme="Billy Bunny learns to share his carrots with friends",
    )


def main():
    t_total = time.time()

    print("\n[1/4] Building handcrafted 60s storyboard...")
    storyboard = build_storyboard_60s()
    print(f"  Title: {storyboard.title}")
    print(f"  Scenes: {len(storyboard.scenes)}")
    print(f"  Planned narration duration: {sum(s.duration_s for s in storyboard.scenes):.1f}s")

    print("\n[2/5] Rendering animation with ToonYou + Storybook LoRA...")
    t0 = time.time()
    from engine.animator import Animator

    anim = Animator(CONFIG_PATH)

    def progress_cb(idx, total, msg):
        print(f"  [{idx}/{total}] {msg}")

    episode = anim.generate_episode(
        storyboard=storyboard,
        character_name="storybook",
        progress_callback=progress_cb,
    )

    video_path = episode.get("video_path", "")
    print(f"  Video: {video_path}")
    print(f"  Done ({time.time()-t0:.1f}s)")

    if not video_path or not os.path.exists(video_path):
        print("\nFAILED: No video generated.")
        if episode.get("error"):
            print(f"  Error: {episode['error']}")
        return

    print("\n[3/5] Generating narration + applying audio...")
    t0 = time.time()
    from engine.audio_engine import AudioEngine

    ae = AudioEngine(CONFIG_PATH)
    narration_texts = [s.narration for s in storyboard.scenes]

    audio_dir = os.path.join(SCRIPT_DIR, "output", "_e2e_audio")
    os.makedirs(audio_dir, exist_ok=True)

    narration = ae.generate_episode_narration(
        narration_texts,
        audio_dir,
        pause_between_s=0.2,
    )
    narration_path = narration.get("full_narration_path", "")
    print(f"  Narration: {narration_path}")

    final_video = ae.apply_audio_to_video(
        video_path=video_path,
        narration_path=narration_path,
    )
    print(f"  Final video with audio: {final_video}")
    print(f"  Done ({time.time()-t0:.1f}s)")

    print("\n[4/5] Vertical remaster (1080x1920)...")
    t0 = time.time()
    from utils.ffmpeg_utils import vertical_remaster

    remaster_path = final_video.replace("_with_audio.mp4", "_vertical_1080x1920.mp4")
    vertical_remaster(final_video, remaster_path, width=1080, height=1920)
    print(f"  Remastered: {remaster_path}")
    print(f"  Done ({time.time()-t0:.1f}s)")

    print("\n[5/5] Final report...")
    total_time = time.time() - t_total
    size_kb = os.path.getsize(remaster_path) / 1024.0
    print("=" * 64)
    print("  60s VERTICAL VIDEO GENERATION COMPLETE")
    print("=" * 64)
    print(f"  File: {os.path.abspath(remaster_path)}")
    print(f"  Size: {size_kb:.1f} KB")
    print(f"  Total time: {total_time:.1f}s")
    print("=" * 64)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        raise
