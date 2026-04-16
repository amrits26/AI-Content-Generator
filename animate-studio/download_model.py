"""Download CogVideoX-2b model files"""
from huggingface_hub import snapshot_download
import os

print("Downloading CogVideoX-2b model (this may take a while)...")
path = snapshot_download(
    "THUDM/CogVideoX-2b",
    local_dir_use_symlinks=False,
)
print(f"Model downloaded to: {path}")

# Verify key files
for name in ["transformer", "text_encoder", "vae", "tokenizer"]:
    subdir = os.path.join(path, name)
    if os.path.isdir(subdir):
        files = os.listdir(subdir)
        total_mb = sum(os.path.getsize(os.path.join(subdir, f)) for f in files) / 1e6
        print(f"  {name}: {len(files)} files, {total_mb:.0f} MB")
    else:
        print(f"  {name}: MISSING")
print("Done!")
