import torch
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import numpy as np

proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
model.eval()

# Test with full forward pass (proper way)
texts = ["gore", "horror", "scary"]
dummy = Image.fromarray(np.zeros((256, 256, 3), dtype=np.uint8))

text_inputs = proc(text=texts, return_tensors="pt", padding=True)
img_inputs = proc(images=dummy, return_tensors="pt")

with torch.no_grad():
    # Full forward: gives projected embeddings
    full_out = model(**text_inputs, **img_inputs)
    print(f"Full output type: {type(full_out)}")
    if hasattr(full_out, 'text_embeds'):
        print(f"text_embeds shape: {full_out.text_embeds.shape}")
    if hasattr(full_out, 'image_embeds'):
        print(f"image_embeds shape: {full_out.image_embeds.shape}")
    
    # Also try text_projection directly
    text_out = model.text_model(**text_inputs)
    pooled = text_out.pooler_output
    print(f"\ntext_model pooler_output shape: {pooled.shape}")
    if hasattr(model, 'text_projection'):
        projected = model.text_projection(pooled)
        print(f"text_projection shape: {projected.shape}")
    
    img_out = model.vision_model(**img_inputs)
    img_pooled = img_out.pooler_output
    print(f"\nvision_model pooler_output shape: {img_pooled.shape}")
    if hasattr(model, 'visual_projection'):
        img_projected = model.visual_projection(img_pooled)
        print(f"visual_projection shape: {img_projected.shape}")
