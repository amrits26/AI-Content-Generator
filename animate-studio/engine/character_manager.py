"""
═══════════════════════════════════════════════════════════════
AniMate Studio — Character Consistency Manager
═══════════════════════════════════════════════════════════════
Manages character LoRA loading, IP-Adapter face locking,
and character profile persistence for consistent designs.
═══════════════════════════════════════════════════════════════
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image

from engine.config import load_config

logger = logging.getLogger("animate_studio.character")


@dataclass
class CharacterProfile:
    """Persistent character profile."""
    name: str
    animal_type: str = "bunny"
    color: str = "soft blue"
    accessory: str = "red bowtie"
    description: str = ""                   # Custom prompt override
    traits: List[str] = field(default_factory=lambda: ["friendly", "curious"])
    reference_image: Optional[str] = None   # Path to reference image
    lora_path: Optional[str] = None         # Path to .safetensors LoRA
    lora_trigger_word: str = ""             # Trigger word for LoRA
    lora_strength: float = 0.8

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "animal_type": self.animal_type,
            "color": self.color,
            "accessory": self.accessory,
            "description": self.description,
            "traits": self.traits,
            "reference_image": self.reference_image,
            "lora_path": self.lora_path,
            "lora_trigger_word": self.lora_trigger_word,
            "lora_strength": self.lora_strength,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CharacterProfile":
        return cls(**{k: v for k, v in data.items()
                   if k in cls.__dataclass_fields__})


class CharacterManager:
    """
    Manages character profiles, LoRA loading, and IP-Adapter
    for maintaining consistent character appearance across scenes.
    """

    def __init__(self, config_path: str = "config.yaml", config: Optional[dict] = None):
        self.config = load_config(config_path=config_path, config=config)

        self.loras_dir = self.config["app"]["loras_dir"]
        self.references_dir = os.path.join(self.config["app"]["assets_dir"], "references")
        self.profiles_path = os.path.join(self.loras_dir, "character_profiles.json")

        os.makedirs(self.loras_dir, exist_ok=True)
        os.makedirs(self.references_dir, exist_ok=True)

        self._profiles: Dict[str, CharacterProfile] = {}
        self._load_profiles()

        if not self._profiles:
            self._create_default_characters()

        # Cached models (loaded on demand)
        self._current_lora_name: Optional[str] = None
        self._ip_adapter_loaded = False

    # ── Default Characters ───────────────────────────────
    def _create_default_characters(self):
        """Seed the library with two starter characters when empty."""
        defaults = [
            CharacterProfile(
                name="Billy Bunny",
                animal_type="bunny",
                color="soft blue",
                accessory="red bowtie",
                description="fluffy blue bunny with big round eyes and a red bowtie, Pixar style",
                traits=["friendly", "curious", "kind"],
            ),
            CharacterProfile(
                name="Baby & Pup",
                animal_type="puppy",
                color="golden",
                accessory="star necklace",
                description="chubby baby in a diaper with a fluffy golden retriever puppy, soft nursery lighting",
                traits=["playful", "giggly", "cuddly"],
            ),
        ]
        for p in defaults:
            self._profiles[p.name] = p
        self._save_profiles()
        logger.info("Created %d default character profiles.", len(defaults))

    # ── Profile Management ───────────────────────────────
    def _load_profiles(self):
        """Load saved character profiles from disk."""
        if os.path.exists(self.profiles_path):
            with open(self.profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for name, pd in data.items():
                self._profiles[name] = CharacterProfile.from_dict(pd)
            logger.info("Loaded %d character profiles.", len(self._profiles))

    def _save_profiles(self):
        """Persist profiles to disk."""
        data = {name: p.to_dict() for name, p in self._profiles.items()}
        with open(self.profiles_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def create_profile(self, profile: CharacterProfile) -> CharacterProfile:
        """Create or update a character profile."""
        self._profiles[profile.name] = profile
        self._save_profiles()
        logger.info("Character '%s' saved.", profile.name)
        return profile

    def get_profile(self, name: str) -> Optional[CharacterProfile]:
        """Get a character by name."""
        return self._profiles.get(name)

    def list_profiles(self) -> List[CharacterProfile]:
        """List all saved character profiles."""
        return list(self._profiles.values())

    def delete_profile(self, name: str) -> bool:
        """Delete a character profile."""
        if name in self._profiles:
            del self._profiles[name]
            self._save_profiles()
            logger.info("Character '%s' deleted.", name)
            return True
        return False

    # ── LoRA Discovery ───────────────────────────────────
    def list_available_loras(self) -> List[Dict[str, Any]]:
        """
        Scan loras/ directory for .safetensors files.
        Auto-creates profiles for orphan LoRA files.
        """
        loras = []
        if not os.path.exists(self.loras_dir):
            return loras

        for f in os.listdir(self.loras_dir):
            if f.endswith(".safetensors"):
                name = f.replace(".safetensors", "")
                full_path = os.path.join(self.loras_dir, f)
                # Check if there's a matching profile
                profile = self._profiles.get(name)
                # Auto-register orphan LoRA files as profiles
                if profile is None:
                    logger.info("Auto-registering orphan LoRA: %s", name)
                    profile = CharacterProfile(
                        name=name,
                        lora_path=full_path,
                        lora_trigger_word=name.lower().replace("_", " "),
                        description=f"{name}, consistent character, high quality",
                    )
                    self.create_profile(profile)
                elif profile.lora_path != full_path:
                    # Update path if file moved
                    profile.lora_path = full_path
                    self._save_profiles()

                loras.append({
                    "name": name,
                    "path": full_path,
                    "has_profile": True,
                    "size_mb": round(os.path.getsize(full_path) / (1024 * 1024), 1),
                })
        return loras

    # ── LoRA Loading (for CogVideoX / ModelScope pipeline) ─
    def load_lora_into_pipeline(self, pipeline, character_name: str, strength_override: float = None) -> bool:
        """
        Load a character's LoRA weights into a diffusion pipeline.
        Uses PEFT directly for UNet3D models (TextToVideoSDPipeline).

        Args:
            pipeline: A diffusers pipeline with load_lora_weights support
            character_name: Name of the character whose LoRA to load
            strength_override: Override the profile's lora_strength (0.0-1.0)
        """
        profile = self.get_profile(character_name)
        if not profile or not profile.lora_path:
            logger.warning("No LoRA found for '%s'.", character_name)
            return False

        if not os.path.exists(profile.lora_path):
            logger.warning("LoRA file not found: %s", profile.lora_path)
            return False

        strength = strength_override if strength_override is not None else profile.lora_strength

        try:
            # Unload previous LoRA if different
            if self._current_lora_name and self._current_lora_name != character_name:
                try:
                    pipeline.unload_lora_weights()
                except Exception:
                    pass

            # Determine LoRA directory (file path → parent dir)
            lora_path = profile.lora_path
            if os.path.isfile(lora_path):
                lora_dir = os.path.dirname(lora_path)
            else:
                lora_dir = lora_path

            # Prefer PEFT loading for UNet3D/TextToVideo pipelines.
            # It supports both PEFT-format and diffusers-format checkpoint keys.
            self._load_lora_via_peft(pipeline, lora_dir, character_name, strength)

            self._current_lora_name = character_name
            logger.info(
                "LoRA loaded: '%s' (strength=%.2f)",
                character_name, strength,
            )
            return True

        except Exception as e:
            logger.error("Failed to load LoRA for '%s': %s", character_name, e)
            return False

    def _load_lora_via_peft(self, pipeline, lora_dir: str, name: str, strength: float):
        """Load LoRA weights via PEFT for UNet3D models."""
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        import json as _json

        # Read adapter config if present, otherwise use training defaults.
        config_path = os.path.join(lora_dir, "adapter_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = _json.load(f)
        else:
            cfg = {
                "r": 8,
                "lora_alpha": 8,
                "target_modules": ["to_q", "to_k", "to_v", "to_out.0"],
                "lora_dropout": 0.05,
            }

        unet = pipeline.unet

        # Add adapter directly when supported; otherwise wrap with PEFT.
        if not hasattr(unet, "peft_config"):
            lora_config = LoraConfig(
                r=cfg["r"],
                lora_alpha=cfg["lora_alpha"],
                target_modules=cfg["target_modules"],
                lora_dropout=cfg.get("lora_dropout", 0.0),
            )
            if hasattr(unet, "add_adapter"):
                unet.add_adapter(lora_config)
            else:
                # Older UNet3D variants require wrapping with PEFT.
                # Keep a no-op hook so accelerate offload cleanup won't crash.
                class _NoOpHook:
                    def detach_hook(self, _module):
                        return None

                pipeline.unet = get_peft_model(unet, lora_config)
                if not hasattr(pipeline.unet, "_hf_hook"):
                    pipeline.unet._hf_hook = _NoOpHook()
                unet = pipeline.unet

        # Load saved weights
        weights_path = os.path.join(lora_dir, "adapter_model.safetensors")
        if not os.path.exists(weights_path):
            # Fallback to pytorch_lora_weights.safetensors
            weights_path = os.path.join(lora_dir, "pytorch_lora_weights.safetensors")

        state_dict = load_file(weights_path)

        # Convert diffusers-format keys to PEFT-format keys when needed.
        # diffusers: unet.<layer>.lora_A.weight
        # peft:      base_model.model.<layer>.lora_A.default.weight
        if any(k.startswith("unet.") for k in state_dict.keys()):
            converted = {}
            for key, value in state_dict.items():
                new_key = key.replace("unet.", "base_model.model.")
                new_key = new_key.replace(".lora_A.weight", ".lora_A.default.weight")
                new_key = new_key.replace(".lora_B.weight", ".lora_B.default.weight")
                converted[new_key] = value
            state_dict = converted

        unet.load_state_dict(state_dict, strict=False)

        # Scale LoRA weights by strength
        for param_name, param in unet.named_parameters():
            if "lora_" in param_name:
                param.data *= strength

        logger.info("LoRA loaded via PEFT: %s (%d tensors)", name, len(state_dict))

    def unload_lora(self, pipeline) -> None:
        """Unload current LoRA weights from pipeline."""
        if self._current_lora_name:
            try:
                pipeline.unload_lora_weights()
            except Exception:
                pass
            self._current_lora_name = None

    # ── Face Embedding for IP-Adapter Consistency ─────────
    def get_face_embedding(self, character_name: str, reference_image_path: str):
        """
        Compute or load cached face embedding for IP-Adapter consistency.
        Falls back to dummy embedding if face libraries are not available.
        """
        import cv2

        cache_dir = Path("loras/character_embeddings")
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{character_name}.npy"

        # Return cached embedding if it exists
        if cache_path.exists():
            return np.load(cache_path)

        embedding = None

        try:
            import insightface
            model = insightface.app.FaceAnalysis()
            model.prepare(ctx_id=0, det_size=(640, 640))
            img = cv2.imread(reference_image_path)
            if img is None:
                raise ValueError(f"Cannot read image: {reference_image_path}")
            faces = model.get(img)
            if faces:
                embedding = faces[0].embedding
        except ImportError:
            try:
                import face_recognition
                img = face_recognition.load_image_file(reference_image_path)
                encodings = face_recognition.face_encodings(img)
                if encodings:
                    embedding = encodings[0]
            except ImportError:
                # Fallback to dummy embedding
                embedding = np.random.randn(512).astype(np.float32)

        if embedding is None:
            embedding = np.random.randn(512).astype(np.float32)

        # Save to cache
        np.save(cache_path, embedding)
        return embedding

    # ── IP-Adapter (Reference Image Fallback) ────────────
    def setup_ip_adapter(self, pipeline, character_name: str) -> bool:
        """
        Set up IP-Adapter for face/style locking using a reference image.
        Used as fallback when no trained LoRA is available.

        Args:
            pipeline: A diffusers pipeline with IP-Adapter support
            character_name: Name of character to use reference image for
        """
        profile = self.get_profile(character_name)
        if not profile or not profile.reference_image:
            logger.warning("No reference image for '%s'.", character_name)
            return False

        if not os.path.exists(profile.reference_image):
            logger.warning("Reference image not found: %s", profile.reference_image)
            return False

        try:
            pipeline.load_ip_adapter(
                "h94/IP-Adapter",
                subfolder="models",
                weight_name="ip-adapter_sd15.bin",
            )
            self._ip_adapter_loaded = True
            logger.info("IP-Adapter loaded for '%s'.", character_name)
            return True

        except Exception as e:
            logger.error("Failed to load IP-Adapter: %s", e)
            return False

    def get_reference_image(self, character_name: str) -> Optional[Image.Image]:
        """Load the reference image for a character."""
        profile = self.get_profile(character_name)
        if not profile or not profile.reference_image:
            return None
        if not os.path.exists(profile.reference_image):
            return None
        return Image.open(profile.reference_image).convert("RGB")

    # ── Character Prompt Builder ─────────────────────────
    def get_character_prompt(self, character_name: str) -> str:
        """
        Build a prompt-ready character description string.
        Includes LoRA trigger word if available.
        """
        from utils.prompt_templates import build_character_description

        profile = self.get_profile(character_name)
        if not profile:
            return f"a cute {character_name}"

        if profile.description:
            desc = profile.description
        else:
            desc = build_character_description(
                animal_type=profile.animal_type,
                name=profile.name,
                color=profile.color,
                accessory=profile.accessory,
            )

        # Prepend LoRA trigger word
        if profile.lora_trigger_word:
            desc = f"{profile.lora_trigger_word}, {desc}"

        return desc

    def save_reference_image(self, character_name: str, image: Image.Image) -> str:
        """Save a reference image for a character and update their profile."""
        filename = f"{character_name}_reference.png"
        path = os.path.join(self.references_dir, filename)
        image.save(path)

        profile = self.get_profile(character_name)
        if profile:
            profile.reference_image = path
            self._save_profiles()

        logger.info("Reference image saved: %s", path)
        return path
