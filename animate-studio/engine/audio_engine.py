"""
═══════════════════════════════════════════════════════════════
AniMate Studio — Audio Engine
═══════════════════════════════════════════════════════════════
TTS narration via ElevenLabs (primary) or Edge TTS (free fallback),
plus BGM mixing with audio ducking via FFmpeg.
═══════════════════════════════════════════════════════════════
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import yaml

from utils.ffmpeg_utils import (
    add_narration_only,
    get_media_duration,
    mix_audio,
    run_ffmpeg,
)

logger = logging.getLogger("animate_studio.audio")


class AudioEngine:
    """
    Handles narration (TTS) + background music mixing for episodes.

    TTS Providers:
      - ElevenLabs API: High-quality, child-friendly voices (paid)
      - Edge TTS: Free Microsoft neural TTS fallback

    Audio Pipeline:
      1. Generate narration audio per scene → concatenate
      2. Select BGM track from assets/audio/
      3. Mix narration + BGM with ducking via FFmpeg
      4. Apply to video, adjusting speed if needed
    """

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        audio_cfg = self.config["audio"]
        self.tts_provider = audio_cfg["tts"]["provider"]
        self.elevenlabs_key = os.environ.get("ELEVENLABS_API_KEY") or audio_cfg["tts"].get("elevenlabs_api_key", "")
        self.voices = audio_cfg["tts"]["voices"]
        self.default_voice_id = audio_cfg["tts"]["default_voice_id"]
        self.stability = audio_cfg["tts"]["stability"]
        self.similarity_boost = audio_cfg["tts"]["similarity_boost"]

        self.bgm_volume = audio_cfg["bgm"]["default_volume"]
        self.duck_volume = audio_cfg["bgm"]["duck_volume"]
        self.fade_ms = audio_cfg["bgm"]["fade_duration_ms"]

        self.assets_dir = self.config["app"]["assets_dir"]
        self.audio_dir = os.path.join(self.assets_dir, "audio")
        self.output_dir = self.config["app"]["output_dir"]

        os.makedirs(self.audio_dir, exist_ok=True)

    # ── Voice Management ─────────────────────────────────
    def list_voices(self) -> list[dict]:
        """Return configured voice options."""
        return self.voices

    def get_voice_name(self, voice_id: str) -> str:
        """Get display name for a voice ID."""
        for v in self.voices:
            if v["id"] == voice_id:
                return v["name"]
        return voice_id

    # ── TTS Generation ───────────────────────────────────
    def generate_narration(
        self,
        text: str,
        output_path: str,
        voice_id: Optional[str] = None,
    ) -> str:
        """
        Generate narration audio from text.

        Args:
            text: Narration text to speak
            output_path: Where to save the audio file
            voice_id: ElevenLabs voice ID (uses default if None)
        """
        voice_id = voice_id or self.default_voice_id

        if self.tts_provider == "elevenlabs" and self.elevenlabs_key:
            return self._generate_elevenlabs(text, output_path, voice_id)
        else:
            return self._generate_edge_tts(text, output_path)

    def _generate_elevenlabs(
        self, text: str, output_path: str, voice_id: str
    ) -> str:
        """Generate audio using ElevenLabs API."""
        import requests

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"
        headers = {
            "Accept": "audio/mpeg",
            "Content-Type": "application/json",
            "xi-api-key": self.elevenlabs_key,
        }
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": {
                "stability": self.stability,
                "similarity_boost": self.similarity_boost,
            },
        }

        response = requests.post(url, json=payload, headers=headers, timeout=30)
        response.raise_for_status()

        # Save as mp3 then convert to wav for consistency
        mp3_path = output_path + ".mp3"
        with open(mp3_path, "wb") as f:
            f.write(response.content)

        # Convert to WAV
        run_ffmpeg(
            ["-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "44100", output_path],
            description="elevenlabs_mp3_to_wav",
        )
        os.remove(mp3_path)

        logger.info("ElevenLabs narration: %s (%.1fs)", output_path, get_media_duration(output_path))
        return output_path

    def _generate_edge_tts(self, text: str, output_path: str) -> str:
        """Generate audio using Edge TTS (free fallback)."""
        import asyncio
        import concurrent.futures
        import edge_tts

        # Use a warm, expressive storyteller voice
        voice = "en-US-AnaNeural"  # Friendly female child voice

        async def _edge_tts_task():
            communicate = edge_tts.Communicate(
                text, voice,
                rate="-5%",     # Slightly slower for clarity
                pitch="+5Hz",   # Slightly higher for warmth
            )
            mp3_path = output_path + ".mp3"
            await communicate.save(mp3_path)

            # Convert to WAV
            run_ffmpeg(
                ["-i", mp3_path, "-acodec", "pcm_s16le", "-ar", "44100", output_path],
                description="edge_tts_to_wav",
            )
            os.remove(mp3_path)

        # Safe async execution: use a thread if an event loop is already running (Gradio)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as pool:
                pool.submit(asyncio.run, _edge_tts_task()).result(timeout=120)
        else:
            asyncio.run(_edge_tts_task())

        logger.info("Edge TTS narration: %s (%.1fs)", output_path, get_media_duration(output_path))
        return output_path

    # ── Full Episode Audio ───────────────────────────────
    def generate_episode_narration(
        self,
        narration_texts: list[str],
        output_dir: str,
        voice_id: Optional[str] = None,
        pause_between_s: float = 0.8,
    ) -> dict:
        """
        Generate narration for all scenes and concatenate.

        Args:
            narration_texts: List of narration strings (one per scene)
            output_dir: Directory to save audio files
            voice_id: Voice to use
            pause_between_s: Silence gap between scene narrations

        Returns:
            dict with "full_narration_path", "scene_audio_paths", "total_duration_s"
        """
        os.makedirs(output_dir, exist_ok=True)
        scene_paths = []

        for i, text in enumerate(narration_texts):
            scene_audio = os.path.join(output_dir, f"narration_scene_{i+1:03d}.wav")
            self.generate_narration(text, scene_audio, voice_id)
            scene_paths.append(scene_audio)

        # Generate silence between scenes
        silence_path = os.path.join(output_dir, "_silence.wav")
        run_ffmpeg(
            [
                "-f", "lavfi",
                "-i", f"anullsrc=r=44100:cl=stereo",
                "-t", str(pause_between_s),
                "-acodec", "pcm_s16le",
                silence_path,
            ],
            description="generate_silence",
        )

        # Build concat list: narration, silence, narration, silence, ...
        concat_list = os.path.join(output_dir, "_narration_concat.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for j, sp in enumerate(scene_paths):
                sp_safe = os.path.abspath(sp).replace("\\", "/")
                f.write(f"file '{sp_safe}'\n")
                if j < len(scene_paths) - 1:
                    sil_safe = os.path.abspath(silence_path).replace("\\", "/")
                    f.write(f"file '{sil_safe}'\n")

        full_narration = os.path.join(output_dir, "full_narration.wav")
        run_ffmpeg(
            ["-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", full_narration],
            description="concat_narration",
        )

        total_dur = get_media_duration(full_narration)
        logger.info("Full narration: %.1fs across %d scenes", total_dur, len(scene_paths))

        return {
            "full_narration_path": full_narration,
            "scene_audio_paths": scene_paths,
            "total_duration_s": total_dur,
        }

    # ── BGM Management ───────────────────────────────────
    def list_bgm_tracks(self) -> list[dict]:
        """List available background music tracks."""
        tracks = []
        if not os.path.exists(self.audio_dir):
            return tracks

        for f in os.listdir(self.audio_dir):
            if f.lower().endswith((".mp3", ".wav", ".ogg", ".flac")):
                path = os.path.join(self.audio_dir, f)
                try:
                    dur = get_media_duration(path)
                except Exception:
                    dur = 0
                tracks.append({
                    "name": os.path.splitext(f)[0],
                    "path": path,
                    "duration_s": round(dur, 1),
                })
        return tracks

    def prepare_bgm(
        self,
        bgm_path: str,
        target_duration: float,
        output_path: str,
    ) -> str:
        """
        Prepare BGM track: loop to target duration, fade in/out.

        Args:
            bgm_path: Path to music file
            target_duration: How long the track should be
            output_path: Where to save prepared track
        """
        fade_s = self.fade_ms / 1000.0

        run_ffmpeg(
            [
                "-stream_loop", "-1",
                "-i", bgm_path,
                "-t", str(target_duration),
                "-af", (
                    f"afade=t=in:d={fade_s},"
                    f"afade=t=out:st={target_duration - fade_s}:d={fade_s}"
                ),
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                output_path,
            ],
            description="prepare_bgm",
        )
        logger.info("BGM prepared: %s (%.1fs)", output_path, target_duration)
        return output_path

    # ── Mix Everything Together ───────────────────────────
    def apply_audio_to_video(
        self,
        video_path: str,
        narration_path: str,
        bgm_path: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> str:
        """
        Apply narration + optional BGM to a video.

        If BGM is provided, uses audio ducking (BGM lowers during narration).
        """
        if output_path is None:
            base = os.path.splitext(video_path)[0]
            output_path = f"{base}_with_audio.mp4"

        narration_dur = get_media_duration(narration_path)

        if bgm_path and os.path.exists(bgm_path):
            # Prepare BGM to match narration length + some extra
            prepared_bgm = os.path.join(os.path.dirname(output_path), "_prepared_bgm.wav")
            self.prepare_bgm(bgm_path, narration_dur + 2.0, prepared_bgm)

            mix_audio(
                video_path=video_path,
                narration_path=narration_path,
                bgm_path=prepared_bgm,
                output_path=output_path,
                bgm_volume=self.bgm_volume,
                duck_volume=self.duck_volume,
                fade_ms=self.fade_ms,
            )

            # Clean up
            if os.path.exists(prepared_bgm):
                os.remove(prepared_bgm)
        else:
            add_narration_only(video_path, narration_path, output_path)

        logger.info("Audio applied: %s", output_path)
        return output_path
