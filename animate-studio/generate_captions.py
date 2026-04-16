"""Generate metadata.jsonl captions for LoRA training dataset."""
import json
import os

dataset_dir = os.path.join(os.path.dirname(__file__), "datasets", "kids_style_v1")
files = sorted(f for f in os.listdir(dataset_dir) if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")))

caption_map = {
    "unicorn": "cute 3D animated unicorn character for children, bright pastel colors, big sparkling eyes, soft lighting, Pixar quality cartoon style",
    "sheep": "adorable cartoon sheep character illustration for kids, fluffy wool, cute expression, soft pastel colors, children storybook style",
    "fox": "cute cartoon fox character for children, bright orange fur, friendly expression, children storybook illustration style",
    "hamster": "cute adorable hamster cartoon, happy funny baby animal, fluffy soft fur, big round eyes, cheerful expression, kids cartoon style",
    "zoo": "cute 3D animated zoo animals cartoon, colorful bright characters, friendly faces, children animation style, cheerful atmosphere",
    "children": "cute animated children group, happy cartoon kids, bright colors, friendly expressions, 3D Pixar animation style",
    "reading": "happy cartoon children reading books together, bright classroom, colorful illustration, kids education animation style",
    "baby-animal": "cute baby animals cartoon characters, adorable fluffy creatures, big innocent eyes, pastel colors, children storybook illustration",
    "cartoon": "cute cartoon animal characters, bright colors, adorable expressions, soft rounded shapes, children animation style, high quality",
}

default_caption = (
    "cute adorable animated characters, bright colors, soft warm lighting, "
    "big expressive eyes, children cartoon style, Pixar quality 3D animation, "
    "fluffy soft textures"
)

entries = []
for f in files:
    fl = f.lower()
    caption = default_caption
    for keyword, cap in caption_map.items():
        if keyword in fl:
            caption = cap
            break
    entries.append({"file_name": f, "text": caption})

metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
with open(metadata_path, "w", encoding="utf-8") as fp:
    for e in entries:
        fp.write(json.dumps(e) + "\n")

print(f"Generated {len(entries)} captions in metadata.jsonl")
for e in entries[:5]:
    fname = e["file_name"][:45]
    cap = e["text"][:60]
    print(f"  {fname} => {cap}...")
