import json
import os
import re
from datetime import datetime

def generate_marketing_kit(manifest_path, output_dir=None):
    """
    Analyze manifest.json and generate:
      - 3 SEO YouTube titles
      - 150-word description with hashtags
      - TikTok Hook caption
    Save as marketing_kit.json in output_dir (or manifest dir).
    """
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    title = manifest.get("title", "AniMate Studio Animation")
    theme = manifest.get("theme", "")
    scenes = manifest.get("scenes", [])
    keywords = ["AI Animation", "Kids Story", "Vintage", "4K", "Cinematic", "Storybook"]
    # 1. SEO Titles
    base = re.sub(r'[^\w\s]', '', title)
    titles = [
        f"{base} | 4K Cinematic AI Animation",
        f"{base} — Vintage Storybook in 4K",
        f"{base} | {theme[:40]}... [AI Kids Animation]"
    ]
    # 2. Description
    hashtags = ["#AIAnimation", "#VintageStorybook", "#4K", "#KidsCartoon", "#AniMateStudio"]
    desc = f"Experience a cinematic, agentic animation: '{title}'. This story brings {theme} to life with beautiful 4K visuals, emotional arcs, and a vintage storybook feel.\n\n"
    for i, s in enumerate(scenes):
        desc += f"Scene {i+1}: {s.get('action','')}\n"
    desc += "\nCreated with AniMate Studio. " + " ".join(hashtags)
    # 3. TikTok Hook
    hook = f"{title}: {scenes[0]['action'] if scenes else theme} #AIAnimation #Shorts"
    kit = {
        "youtube_titles": titles,
        "description": desc,
        "tiktok_hook": hook,
        "generated_at": datetime.now().isoformat(),
    }
    out_dir = output_dir or os.path.dirname(manifest_path)
    out_path = os.path.join(out_dir, "marketing_kit.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(kit, f, indent=2)
    return out_path
