"""
═══════════════════════════════════════════════════════════════
AniMate Studio — Kid-Safe Animation Prompt Templates
═══════════════════════════════════════════════════════════════
Standardized prompt engineering for consistent Pixar-esque
3D/2.5D animation style across all generated content.
Supports multiple style presets and camera motion descriptors.
═══════════════════════════════════════════════════════════════
"""

# ── Style Presets ────────────────────────────────────────
STYLE_PRESETS = {
    "pixar_cute": {
        "prefix": (
            "3D animated Pixar render, soft volumetric lighting, "
            "vibrant saturated colors, smooth animation, high detail, "
            "big expressive eyes, fluffy soft textures, rounded shapes, "
            "warm golden color palette, studio quality render, "
            "cute adorable character design, clean sharp focus, "
            "professional CGI animation, beautiful composition"
        ),
        "negative": (
            "ugly, deformed, disfigured, extra limbs, bad anatomy, "
            "blurry, out of focus, watermark, text, logo, signature, "
            "realistic human, photorealistic, creepy, sharp teeth, "
            "dark lighting, horror, nightmare, shadow, monochrome, "
            "nsfw, nudity, adult content, violence, blood, "
            "low quality, pixelated, grainy, jpeg artifacts, noise, "
            "bad proportions, cropped, worst quality, normal quality, "
            "poorly drawn, bad hands, mutation, extra fingers"
        ),
    },
    # Compact preset for AnimateDiff — stays well under 77 CLIP tokens
    "animatediff_cartoon": {
        "prefix": (
            "cartoon style, pastel colors, warm soft lighting"
        ),
        "negative": (
            "ugly, deformed, blurry, watermark, text, logo, "
            "realistic, photorealistic, horror, dark, monochrome, "
            "nsfw, nudity, violence, low quality, worst quality"
        ),
    },
    "hyper_realistic": {
        "prefix": (
            "Photorealistic, masterpiece, best quality, 8k UHD, "
            "cinematic lighting, shallow depth of field, "
            "natural skin tones, professional photography, "
            "detailed textures, lifelike, golden hour glow, "
            "sharp focus, high resolution, beautiful composition"
        ),
        "negative": (
            "cartoon, 3d render, illustration, painting, watermark, "
            "anime, cel-shaded, flat colors, vector art, "
            "nsfw, nudity, adult content, text, logo, "
            "low quality, blurry, pixelated, grainy, deformed, "
            "bad anatomy, worst quality, jpeg artifacts"
        ),
    },
}

# ── Default style (backward-compatible) ──────────────────
STYLE_PREFIX = STYLE_PRESETS["pixar_cute"]["prefix"]
NEGATIVE_PROMPT = STYLE_PRESETS["pixar_cute"]["negative"]


def get_style_preset(style_name: str = "pixar_cute") -> dict:
    """Get a style preset dict with 'prefix' and 'negative' keys."""
    return STYLE_PRESETS.get(style_name, STYLE_PRESETS["pixar_cute"])


# ── Camera Motion Presets ────────────────────────────────
CAMERA_MOTIONS = {
    "static": "static medium shot, steady camera, clean composition",
    "slow_zoom_in": "slow smooth cinematic zoom in, gradually revealing detail",
    "gentle_pan_right": "gentle smooth pan right, panoramic reveal of scene",
    "dolly_out": "slow dolly out, widening reveal of the full environment",
    "tilt_up": "gentle tilt up, ground to sky reveal, environment showcase",
    "tracking": "smooth tracking shot following character movement",
    "close_up": "close-up shot, detailed face expression, shallow depth of field",
    "bird_eye": "bird eye view, looking down, miniature world perspective",
}


def get_camera_motion(motion_key: str = "static") -> str:
    """Get camera motion descriptor string."""
    return CAMERA_MOTIONS.get(motion_key, "")


# Scene-appropriate automatic camera picks
_AUTO_CAMERA_MAP = {
    "happy": "slow_zoom_in",
    "sad": "static",
    "excited": "tracking",
    "curious": "gentle_pan_right",
    "brave": "dolly_out",
    "shy": "static",
    "surprised": "slow_zoom_in",
    "loving": "slow_zoom_in",
    "sleepy": "static",
    "proud": "tilt_up",
}

# ── Emotion Tone Modifiers ───────────────────────────────────
EMOTION_TONES = {
    "happy": "joyful expression, bright warm smile, sparkling eyes, warm golden lighting, cheerful atmosphere, vibrant colors",
    "sad": "gentle tears, droopy ears, soft blue lighting, empathetic mood, muted cool tones, rain drops on window",
    "excited": "wide sparkling eyes, bouncing dynamic pose, vibrant warm colors, motion blur effect, energetic atmosphere",
    "curious": "tilted head, wide wondering eyes, soft spotlight, whimsical atmosphere, floating sparkles, mystery glow",
    "brave": "determined expression, confident heroic stance, warm dramatic sunrise lighting, wind in fur, epic atmosphere",
    "shy": "half-hidden behind object, peeking eyes, soft pink blush, gentle pastel lighting, cozy safe atmosphere",
    "surprised": "wide open mouth, raised eyebrows, bright pop of color, sparkle effect, dramatic reveal lighting",
    "loving": "heart-shaped eyes, warm embrace pose, golden hour lighting, cozy warm atmosphere, soft bokeh hearts",
    "sleepy": "half-closed drowsy eyes, gentle yawn, moonlit scene, soft purple-blue tones, twinkling stars, peaceful",
    "proud": "chest puffed out, chin up, sparkling achievement glow, warm spotlight, confetti, celebration atmosphere",
}

# ── Scene Setting Templates (vivid background descriptors) ──
SCENE_SETTINGS = {
    "meadow": "sunlit meadow with wildflowers and gentle rolling hills, golden hour light, butterflies dancing, soft ambient glow",
    "forest": "magical enchanted forest, sunbeams streaming through canopy, mossy path with glowing mushrooms, friendly fireflies, dappled light",
    "bedroom": "cozy pastel bedroom with soft plush toys, warm nightlight casting gentle shadows, fluffy bed with star-patterned blanket",
    "kitchen": "warm farmhouse kitchen with copper pots, soft window light streaming in, checkered floor, fresh bread on counter, cozy atmosphere",
    "playground": "colorful playground with soft rubber ground, bright slides and swings, sunny day with fluffy clouds, happy atmosphere",
    "beach": "gentle turquoise ocean waves, soft golden sand with pastel shells, warm sunset painting the sky orange and pink",
    "classroom": "friendly classroom with colorful decorations, tiny wooden desks, alphabet on walls, big sunny windows",
    "garden": "beautiful blooming flower garden, butterflies and bumblebees, stone path leading to a white picket fence, rainbow overhead",
    "space": "friendly outer space with smiling stars, colorful nebulae, cute planets with rings, cozy rocket ship with portholes",
    "underwater": "magical underwater coral reef, friendly tropical fish, sunbeams filtering through crystal clear water, sea turtles gliding",
    "nursery": "cozy pastel nursery with plush toys, soft crib with mobile, starry night light projecting constellations on ceiling, warm lamp glow",
    "park": "sunny city park with ancient oak trees, children's fountain, park bench with feeding birds, green grass with dandelions",
    "treehouse": "magical wooden treehouse high in an old oak, rope ladder, twinkling fairy lights, cozy blanket nest inside",
    "bakery": "warm vintage bakery with golden pastries in glass cases, flour-dusted counter, brick oven glowing, sweet aroma atmosphere",
}

# ── Character Description Templates ──────────────────────────
CHARACTER_TEMPLATES = {
    "bunny": (
        "A fluffy {color} bunny named {name} with big round sparkling eyes, "
        "long floppy ears, tiny pink nose, small cotton tail, "
        "wearing a tiny {accessory}"
    ),
    "duckling": (
        "A tiny {color} duckling named {name} with bright curious eyes, "
        "small orange beak, fluffy downy feathers, little webbed feet, "
        "wearing a tiny {accessory}"
    ),
    "kitten": (
        "An adorable {color} kitten named {name} with large expressive eyes, "
        "tiny whiskers, soft fur, small pink toe beans, "
        "wearing a tiny {accessory}"
    ),
    "puppy": (
        "A playful {color} puppy named {name} with big floppy ears, "
        "shiny wet nose, wagging fluffy tail, "
        "wearing a tiny {accessory}"
    ),
    "bear_cub": (
        "A cuddly {color} bear cub named {name} with round ears, "
        "button nose, soft plush fur, chubby cheeks, "
        "wearing a tiny {accessory}"
    ),
    "owl": (
        "A wise little {color} owl named {name} with huge round eyes, "
        "fluffy feather tufts, tiny beak, soft downy belly, "
        "wearing a tiny {accessory}"
    ),
}

# ── Common Accessories ───────────────────────────────────────
ACCESSORIES = [
    "red bowtie", "blue scarf", "yellow hat", "pink ribbon",
    "green backpack", "purple cape", "orange bandana", "star necklace",
]


def build_scene_prompt(
    character_desc: str,
    action: str,
    emotion: str = "happy",
    setting: str = "meadow",
    extra_details: str = "",
    style: str = "pixar_cute",
    camera_motion: str = "auto",
) -> str:
    """
    Build a complete scene generation prompt with style prefix.

    Args:
        character_desc: Full character description string
        action: What the character is doing (e.g., "learning to share a carrot")
        emotion: Key from EMOTION_TONES dict
        setting: Key from SCENE_SETTINGS dict or custom string
        extra_details: Additional prompt details
        style: Style preset key ("pixar_cute" or "hyper_realistic")
        camera_motion: Camera motion key or "auto" for emotion-based selection
    """
    preset = get_style_preset(style)
    style_prefix = preset["prefix"]
    emotion_mod = EMOTION_TONES.get(emotion, EMOTION_TONES["happy"])
    scene_bg = SCENE_SETTINGS.get(setting, setting)

    # Camera motion
    if camera_motion == "auto":
        cam_key = _AUTO_CAMERA_MAP.get(emotion, "static")
    else:
        cam_key = camera_motion
    cam_desc = get_camera_motion(cam_key)

    prompt = f"{style_prefix}, {cam_desc}, " if cam_desc else f"{style_prefix}, "
    prompt += (
        f"{character_desc} is {action}. "
        f"{emotion_mod}. "
        f"{scene_bg}."
    )
    if extra_details:
        prompt += f" {extra_details}."

    return prompt


def get_negative_prompt(style: str = "pixar_cute") -> str:
    """Get the negative prompt for a given style."""
    preset = get_style_preset(style)
    return preset["negative"]


def build_character_description(
    animal_type: str = "bunny",
    name: str = "Billy",
    color: str = "soft blue",
    accessory: str = "red bowtie",
) -> str:
    """
    Build a character description from template.

    Args:
        animal_type: Key from CHARACTER_TEMPLATES
        name: Character's name
        color: Primary color descriptor
        accessory: What the character wears
    """
    template = CHARACTER_TEMPLATES.get(animal_type, CHARACTER_TEMPLATES["bunny"])
    return template.format(name=name, color=color, accessory=accessory)


def build_transition_prompt(prev_scene_desc: str, next_scene_desc: str) -> str:
    """Build a smooth transition prompt between two scenes."""
    return (
        f"{STYLE_PREFIX}, "
        f"smooth cinematic transition, "
        f"from {prev_scene_desc} transitioning to {next_scene_desc}, "
        f"gentle camera movement, soft focus shift, "
        f"maintaining consistent pastel art style"
    )


def get_story_system_prompt() -> str:
    """System prompt for the LLM story generator."""
    return """You are an award-winning children's story writer and animation director specializing in short animated episodes for kids ages 2-8. You think cinematically — every scene is a visual masterpiece.

RULES:
1. Stories must be positive, gentle, and educational
2. Main character learns a simple moral lesson (sharing, kindness, bravery, etc.)
3. NO scary elements, villains, conflict involving danger, or sad endings
4. Use simple vocabulary suitable for young children
5. Each scene should be visually distinct with different settings, colors, and camera angles
6. Stories must have a clear structure: setup → gentle challenge → resolution → happy ending
7. Keep narration to 1-2 short sentences per scene (conversational, warm tone)
8. Characters should be cute baby animals with expressive faces

VISUAL DESCRIPTION GUIDELINES (critical for animation quality):
- Start each visual_description with the character's FULL appearance (color, species, clothing)
- Include specific LIGHTING details (golden hour, soft morning light, warm sunset glow)
- Describe the BACKGROUND and environment richly (specific plants, objects, weather)
- Include a specific CHARACTER ACTION/POSE (reaching forward, jumping with arms up)
- Mention FACIAL EXPRESSION details (wide sparkling eyes, big toothy grin, raised eyebrows)
- Include at least one SPECIAL EFFECT (sparkles, light rays, floating particles, soft bokeh)
- Each scene's visual_description should be 2-3 detailed sentences

OUTPUT FORMAT (strict JSON):
{
  "title": "Episode Title",
  "moral": "One-sentence moral lesson",
  "scenes": [
    {
      "scene_id": 1,
      "narration": "Short narration text read aloud",
      "visual_description": "Detailed visual description for image generation — include character appearance, action, expression, lighting, background, and special effects",
      "emotion_tone": "happy|sad|excited|curious|brave|shy|surprised|loving|sleepy|proud",
      "setting": "meadow|forest|bedroom|kitchen|playground|beach|classroom|garden|space|underwater"
    }
  ]
}"""


def get_story_user_prompt(
    theme: str,
    character_name: str,
    character_type: str,
    num_scenes: int = 5,
) -> str:
    """Build the user prompt for story generation."""
    return (
        f"Write a {num_scenes}-scene children's animated story about: {theme}\n"
        f"Main character: {character_name} the {character_type}\n"
        f"Each scene should be 10-12 seconds of animation.\n"
        f"IMPORTANT: Make each visual_description extremely detailed — describe the character's exact pose, "
        f"facial expression, the lighting (golden hour, soft morning light, etc.), background details "
        f"(specific flowers, clouds, objects), and at least one magical visual effect (sparkles, light rays, etc.).\n"
        f"Use DIFFERENT settings and camera angles for each scene to keep it visually engaging.\n"
        f"Remember: keep it gentle, positive, and educational for ages 2-8."
    )
