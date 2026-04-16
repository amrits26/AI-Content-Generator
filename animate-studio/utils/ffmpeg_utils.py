"""
═══════════════════════════════════════════════════════════════
AniMate Studio — FFmpeg Utilities
═══════════════════════════════════════════════════════════════
All video processing goes through FFmpeg subprocess calls for
maximum stability and reliability on long renders.
═══════════════════════════════════════════════════════════════
"""

import json
import logging
import os
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger("animate_studio.ffmpeg")


def get_ffmpeg_path() -> str:
    """Return ffmpeg executable path from PATH."""
    return "ffmpeg"


def get_ffprobe_path() -> str:
    """Return ffprobe executable path from PATH."""
    return "ffprobe"


def run_ffmpeg(args: list[str], description: str = "") -> subprocess.CompletedProcess:
    """
    Run an FFmpeg command with full logging.
    Raises subprocess.CalledProcessError on failure.
    """
    cmd = [get_ffmpeg_path(), "-y", "-hide_banner", "-loglevel", "warning"] + args
    logger.info("FFmpeg [%s]: %s", description, " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    if result.stderr:
        logger.debug("FFmpeg stderr: %s", result.stderr[:500])
    return result


def get_media_duration(filepath: str) -> float:
    """Get duration of a media file in seconds using ffprobe."""
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    info = json.loads(result.stdout)
    return float(info["format"]["duration"])


def get_video_info(filepath: str) -> dict:
    """Get video stream info (width, height, fps, duration)."""
    cmd = [
        get_ffprobe_path(),
        "-v", "quiet",
        "-print_format", "json",
        "-show_streams",
        "-show_format",
        filepath,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s["codec_type"] == "video"), {}
    )
    # Parse frame rate safely (no eval)
    rate_str = video_stream.get("r_frame_rate", "0/1")
    if "/" in rate_str:
        try:
            num, den = rate_str.split("/", 1)
            fps = float(num) / float(den) if float(den) != 0 else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0
    else:
        try:
            fps = float(rate_str)
        except ValueError:
            fps = 0.0

    return {
        "width": int(video_stream.get("width", 0)),
        "height": int(video_stream.get("height", 0)),
        "fps": fps,
        "duration": float(data.get("format", {}).get("duration", 0)),
    }


def frames_to_video(
    frames,
    output_path: str,
    fps: int = 8,
    pattern: str = "frame_%04d.png",
) -> str:
    """
    Convert frames to a video file.
    Accepts either a directory path (str) or a list of PIL Images.
    """
    if isinstance(frames, str):
        # Legacy: directory of numbered PNGs
        run_ffmpeg(
            [
                "-framerate", str(fps),
                "-i", os.path.join(frames, pattern),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                output_path,
            ],
            description="frames_to_video",
        )
        return output_path

    # New: list of PIL Image objects
    with tempfile.TemporaryDirectory(prefix="animdiff_frames_") as tmp:
        for i, img in enumerate(frames):
            img.save(os.path.join(tmp, f"frame_{i:04d}.png"))
        run_ffmpeg(
            [
                "-framerate", str(fps),
                "-i", os.path.join(tmp, "frame_%04d.png"),
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                output_path,
            ],
            description="frames_to_video_pil",
        )
    return output_path


def concat_videos(video_paths: list[str], output_path: str) -> str:
    """
    Concatenate multiple video files using FFmpeg concat demuxer.
    All videos must have the same resolution/fps/codec.
    """
    # Write concat list file
    list_path = output_path + ".concat.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for vp in video_paths:
            # Use forward slashes for cross-platform FFmpeg compatibility
            safe_path = os.path.abspath(vp).replace("\\", "/")
            f.write(f"file '{safe_path}'\n")

    run_ffmpeg(
        [
            "-f", "concat",
            "-safe", "0",
            "-i", list_path,
            "-c", "copy",
            output_path,
        ],
        description="concat_videos",
    )
    os.remove(list_path)
    return output_path


def add_crossfade(
    video_a: str,
    video_b: str,
    output_path: str,
    fade_duration: float = 0.5,
) -> str:
    """Add a crossfade transition between two video clips."""
    dur_a = get_media_duration(video_a)
    offset = dur_a - fade_duration

    run_ffmpeg(
        [
            "-i", video_a,
            "-i", video_b,
            "-filter_complex",
            f"[0:v][1:v]xfade=transition=fade:duration={fade_duration}:offset={offset}[v]",
            "-map", "[v]",
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ],
        description="crossfade",
    )
    return output_path


def crop_aspect_ratio(
    input_path: str,
    output_path: str,
    target_width: int,
    target_height: int,
) -> str:
    """
    Scale and crop video to target aspect ratio.
    Strategy: scale to fill target, then center-crop.
    """
    run_ffmpeg(
        [
            "-i", input_path,
            "-vf", (
                f"scale={target_width}:{target_height}:"
                f"force_original_aspect_ratio=increase,"
                f"crop={target_width}:{target_height}"
            ),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ],
        description="crop_aspect",
    )
    return output_path


def mix_audio(
    video_path: str,
    narration_path: str,
    bgm_path: str,
    output_path: str,
    bgm_volume: float = 0.15,
    duck_volume: float = 0.06,
    fade_ms: int = 500,
) -> str:
    """
    Mix narration + background music onto video.
    Uses sidechaincompress-style ducking: BGM lowers when narration is active.
    """
    # Audio ducking filter: detect narration presence via volume, duck BGM
    filter_complex = (
        f"[1:a]aformat=fltp:44100:stereo,volume=1.0[narr];"
        f"[2:a]aformat=fltp:44100:stereo,volume={bgm_volume}[bgm_raw];"
        f"[bgm_raw][narr]sidechaincompress="
        f"threshold=0.02:ratio=8:attack={fade_ms}:release={fade_ms}[bgm_ducked];"
        f"[narr][bgm_ducked]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    )

    # If narration is longer than video, loop video to match narration duration.
    temp_loop_video = None
    video_for_mix = video_path
    try:
        video_dur = get_media_duration(video_path)
        narr_dur = get_media_duration(narration_path)
    except Exception:
        video_dur = 0.0
        narr_dur = 0.0

    if narr_dur > video_dur + 0.1:
        temp_loop_video = output_path + ".loop.mp4"
        run_ffmpeg(
            [
                "-stream_loop", "-1",
                "-i", video_path,
                "-t", f"{narr_dur:.3f}",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                temp_loop_video,
            ],
            description="loop_video_for_audio_mix",
        )
        video_for_mix = temp_loop_video

    run_ffmpeg(
        [
            "-i", video_for_mix,
            "-i", narration_path,
            "-i", bgm_path,
            "-filter_complex", filter_complex,
            "-map", "0:v",
            "-map", "[aout]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ],
        description="mix_audio",
    )

    if temp_loop_video and os.path.exists(temp_loop_video):
        os.remove(temp_loop_video)
    return output_path


def add_narration_only(
    video_path: str,
    narration_path: str,
    output_path: str,
) -> str:
    """Add narration audio to a video (no BGM)."""
    # If narration is longer than video, loop video to match narration duration.
    temp_loop_video = None
    video_for_mix = video_path
    try:
        video_dur = get_media_duration(video_path)
        narr_dur = get_media_duration(narration_path)
    except Exception:
        video_dur = 0.0
        narr_dur = 0.0

    if narr_dur > video_dur + 0.1:
        temp_loop_video = output_path + ".loop.mp4"
        run_ffmpeg(
            [
                "-stream_loop", "-1",
                "-i", video_path,
                "-t", f"{narr_dur:.3f}",
                "-an",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-crf", "18",
                temp_loop_video,
            ],
            description="loop_video_for_narration",
        )
        video_for_mix = temp_loop_video

    run_ffmpeg(
        [
            "-i", video_for_mix,
            "-i", narration_path,
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            output_path,
        ],
        description="add_narration",
    )

    if temp_loop_video and os.path.exists(temp_loop_video):
        os.remove(temp_loop_video)
    return output_path


def extract_frame(video_path: str, output_path: str, timestamp: float = 0.0) -> str:
    """Extract a single frame from video at given timestamp."""
    run_ffmpeg(
        [
            "-ss", str(timestamp),
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "2",
            output_path,
        ],
        description="extract_frame",
    )
    return output_path


def extract_best_thumbnail(
    video_path: str,
    output_path: str,
    count: int = 5,
) -> str:
    """
    Extract the most visually interesting frame as thumbnail.
    Uses scene detection to find high-activity frames, picks the brightest.
    """
    # Extract several candidate frames evenly spaced
    duration = get_media_duration(video_path)
    candidates = []
    temp_dir = os.path.dirname(output_path)

    for i in range(count):
        ts = (duration / (count + 1)) * (i + 1)
        candidate_path = os.path.join(temp_dir, f"_thumb_candidate_{i}.png")
        extract_frame(video_path, candidate_path, ts)
        candidates.append(candidate_path)

    # Pick the brightest / most colorful candidate
    best_path = candidates[0]
    best_score = 0
    for cpath in candidates:
        if os.path.exists(cpath):
            from PIL import Image
            import numpy as np
            img = np.array(Image.open(cpath))
            # Score = brightness * color variance (higher = more interesting)
            score = float(img.mean()) * float(img.std())
            if score > best_score:
                best_score = score
                best_path = cpath

    # Copy best to output, clean up
    from shutil import copy2
    copy2(best_path, output_path)
    for cpath in candidates:
        if os.path.exists(cpath):
            os.remove(cpath)

    return output_path


def adjust_video_speed(
    input_path: str,
    output_path: str,
    target_duration: float,
) -> str:
    """
    Stretch or shrink video duration to match target_duration.
    Uses setpts for video and atempo for audio.
    """
    current_dur = get_media_duration(input_path)
    if current_dur <= 0:
        raise ValueError("Input video has zero duration")

    speed_factor = current_dur / target_duration
    # atempo only accepts 0.5-2.0 range, chain if needed
    video_filter = f"setpts={1/speed_factor}*PTS"

    atempo_filters = []
    remaining = speed_factor
    while remaining > 2.0:
        atempo_filters.append("atempo=2.0")
        remaining /= 2.0
    while remaining < 0.5:
        atempo_filters.append("atempo=0.5")
        remaining /= 0.5
    atempo_filters.append(f"atempo={remaining:.4f}")
    audio_filter = ",".join(atempo_filters)

    run_ffmpeg(
        [
            "-i", input_path,
            "-vf", video_filter,
            "-af", audio_filter,
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            output_path,
        ],
        description="adjust_speed",
    )
    return output_path


def interpolate_frames(
    input_path: str,
    output_path: str,
    target_fps: int = 24,
) -> str:
    """
    Double the frame rate using FFmpeg minterpolate for smoother motion.
    Uses motion-compensated interpolation (MCFI) for natural results.
    """
    run_ffmpeg(
        [
            "-i", input_path,
            "-vf", (
                f"minterpolate='mi_mode=mci:mc_mode=aobmc:me_mode=bidir"
                f":vsbmc=1:fps={target_fps}'"
            ),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "17",
            output_path,
        ],
        description="frame_interpolation",
    )
    return output_path


def enhance_video(
    input_path: str,
    output_path: str,
    sharpen: bool = True,
    denoise: bool = True,
    boost_saturation: float = 1.15,
) -> str:
    """
    Post-processing enhancement: sharpen, denoise, and boost color.
    Designed for AI-generated animation that tends to be soft/noisy.
    """
    filters = []
    if denoise:
        # Gentle denoise — removes AI generation artifacts
        filters.append("hqdn3d=2:2:3:3")
    if sharpen:
        # Unsharp mask: luma_x:luma_y:luma_amount:chroma_x:chroma_y:chroma_amount
        filters.append("unsharp=5:5:0.8:3:3:0.4")
    if boost_saturation != 1.0:
        filters.append(f"eq=saturation={boost_saturation}")

    if not filters:
        from shutil import copy2
        copy2(input_path, output_path)
        return output_path

    run_ffmpeg(
        [
            "-i", input_path,
            "-vf", ",".join(filters),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "17",
            output_path,
        ],
        description="enhance_video",
    )
    return output_path


def apply_final_encode(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    video_codec: str = "libx264",
    video_bitrate: str = "8M",
    audio_codec: str = "aac",
    audio_bitrate: str = "192k",
    crf: int = 18,
    preset: str = "slow",
    pixel_format: str = "yuv420p",
) -> str:
    """Final encoding pass with platform-specific parameters."""
    run_ffmpeg(
        [
            "-i", input_path,
            "-vf", (
                f"scale={width}:{height}:"
                f"force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=white"
            ),
            "-c:v", video_codec,
            "-b:v", video_bitrate,
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", pixel_format,
            "-c:a", audio_codec,
            "-b:a", audio_bitrate,
            "-movflags", "+faststart",
            output_path,
        ],
        description="final_encode",
    )
    return output_path


def vertical_remaster(
    input_path: str,
    output_path: str,
    width: int = 1080,
    height: int = 1920,
    target_duration: float = 0.0,
) -> str:
    """
    Remaster a square (512x512) video to vertical 9:16 (1080x1920).
    Strategy: scale to fill width, pad top/bottom with blurred background.
    Optionally adjust speed to hit target_duration.
    """
    # Blurred-background + sharp center overlay for professional vertical look
    vf = (
        f"[0:v]split=2[bg][fg];"
        f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase,"
        f"crop={width}:{height},boxblur=20:5[bg_blurred];"
        f"[fg]scale={width}:-2:force_original_aspect_ratio=decrease[fg_scaled];"
        f"[bg_blurred][fg_scaled]overlay=(W-w)/2:(H-h)/2[out]"
    )

    args = [
        "-i", input_path,
        "-filter_complex", vf,
        "-map", "[out]",
    ]

    # Carry audio if present
    try:
        probe_cmd = [
            get_ffprobe_path(), "-v", "quiet",
            "-select_streams", "a",
            "-show_entries", "stream=index",
            "-of", "csv=p=0",
            input_path,
        ]
        probe = subprocess.run(probe_cmd, capture_output=True, text=True)
        has_audio = bool(probe.stdout.strip())
    except Exception:
        has_audio = False

    if has_audio:
        args.extend(["-map", "0:a", "-c:a", "aac", "-b:a", "192k"])

    args.extend([
        "-c:v", "libx264",
        "-crf", "18",
        "-preset", "slow",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        output_path,
    ])

    run_ffmpeg(args, description="vertical_remaster")

    # Optionally adjust to target duration
    if target_duration > 0:
        current_dur = get_media_duration(output_path)
        if abs(current_dur - target_duration) > 0.5:
            speed_adjusted = output_path.replace(".mp4", "_speed.mp4")
            adjust_video_speed(output_path, speed_adjusted, target_duration)
            os.replace(speed_adjusted, output_path)

    return output_path
