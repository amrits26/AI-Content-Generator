"""
Download ModelScope text-to-video-ms-1.7b (fp16 variant) using direct HTTP requests
with progress reporting and resume support.
"""
import os
import sys
import time
import requests

REPO = "damo-vilab/text-to-video-ms-1.7b"
BASE_URL = f"https://huggingface.co/{REPO}/resolve/main"

# Files to download (fp16 safetensors + configs)
FILES = [
    ("text_encoder/model.fp16.safetensors", 649.3),
    ("unet/diffusion_pytorch_model.fp16.safetensors", 2691.9),
    ("vae/diffusion_pytorch_model.fp16.safetensors", 159.6),
]

# Determine HuggingFace cache snapshot directory
CACHE_BASE = os.path.join(
    os.path.expanduser("~"), ".cache", "huggingface", "hub",
    "models--damo-vilab--text-to-video-ms-1.7b", "snapshots"
)

def find_snapshot_dir():
    """Find the existing snapshot directory."""
    if os.path.isdir(CACHE_BASE):
        dirs = [d for d in os.listdir(CACHE_BASE) if os.path.isdir(os.path.join(CACHE_BASE, d))]
        if dirs:
            return os.path.join(CACHE_BASE, dirs[0])
    return None


def download_file(url, dest_path, expected_mb):
    """Download a file with resume support and progress reporting."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    # Check if already fully downloaded
    if os.path.exists(dest_path):
        existing_mb = os.path.getsize(dest_path) / (1024 * 1024)
        if existing_mb >= expected_mb * 0.99:
            print(f"  Already downloaded ({existing_mb:.1f} MB)")
            return True

    # Resume support
    start_byte = 0
    if os.path.exists(dest_path):
        start_byte = os.path.getsize(dest_path)
        print(f"  Resuming from {start_byte / 1024 / 1024:.1f} MB")

    headers = {}
    if start_byte > 0:
        headers["Range"] = f"bytes={start_byte}-"

    try:
        r = requests.get(url, stream=True, timeout=30, headers=headers)
        if r.status_code == 416:  # Range not satisfiable = file complete
            print(f"  Already complete")
            return True
        r.raise_for_status()

        total = int(r.headers.get("content-length", 0)) + start_byte
        total_mb = total / (1024 * 1024)

        mode = "ab" if start_byte > 0 else "wb"
        downloaded = start_byte
        t0 = time.time()
        last_print = t0

        with open(dest_path, mode) as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
                f.write(chunk)
                downloaded += len(chunk)

                now = time.time()
                if now - last_print >= 5:  # Print every 5 seconds
                    elapsed = now - t0
                    speed = (downloaded - start_byte) / elapsed / (1024 * 1024)
                    pct = downloaded / total * 100 if total else 0
                    remaining = (total - downloaded) / (speed * 1024 * 1024) if speed > 0 else 0
                    print(f"  {downloaded / 1024 / 1024:.0f}/{total_mb:.0f} MB ({pct:.0f}%) | {speed:.1f} MB/s | ETA {remaining:.0f}s")
                    sys.stdout.flush()
                    last_print = now

        final_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  Complete: {final_mb:.1f} MB")
        return True

    except Exception as e:
        print(f"  ERROR: {e}")
        return False


def main():
    snap_dir = find_snapshot_dir()
    if not snap_dir:
        print("ERROR: No snapshot directory found. Run the config download first:")
        print("  python -c \"from huggingface_hub import snapshot_download; snapshot_download('damo-vilab/text-to-video-ms-1.7b', allow_patterns=['*.json','*.txt','*.model'])\"")
        sys.exit(1)

    print(f"Snapshot dir: {snap_dir}")
    print(f"Files to download: {len(FILES)}")
    total_mb = sum(f[1] for f in FILES)
    print(f"Total: {total_mb:.0f} MB ({total_mb/1024:.1f} GB)")
    print()

    for filename, expected_mb in FILES:
        url = f"{BASE_URL}/{filename}"
        dest = os.path.join(snap_dir, filename)
        print(f"[{filename}] ({expected_mb:.0f} MB)")
        success = download_file(url, dest, expected_mb)
        if not success:
            print(f"  FAILED - you may need to retry")
        print()

    print("Download complete! You can now run the pipeline.")


if __name__ == "__main__":
    main()
