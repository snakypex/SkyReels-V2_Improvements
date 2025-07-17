# Basic usage example
from diffusers import DiffusionPipeline
import torch

# Load the model (with float16 precision for GPU)
pipe = DiffusionPipeline.from_pretrained(
    "Heartsync/NSFW-Uncensored",
    torch_dtype=torch.float16
)
pipe.to("cuda")  # Move to GPU

# Generate an image with a simple prompt
prompt = "Portrait stylisé d'une femme nu, rayonnante de bonheur, tenant des liasses de billets de banque. Elle est dans une chambre luxueuse aux couleurs chaudes, avec un éclairage doux et tamisé. Ambiance de luxe décontracté et d'opulence discrète"
negative_prompt = "low quality, blurry, deformed"

# Create the image
image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    num_inference_steps=30,
    guidance_scale=7.5
).images[0]

# Save the image
image.save("generated_image.png")

# Advanced example - fixed seed and additional parameters
import numpy as np

# Set seed for reproducible results
seed = 42
generator = torch.Generator("cuda").manual_seed(seed)

# Advanced parameter settings
prompt = "Portrait stylisé d'une femme nu, rayonnante de bonheur, tenant des liasses de billets de banque. Elle est dans une chambre luxueuse aux couleurs chaudes, avec un éclairage doux et tamisé. Ambiance de luxe décontracté et d'opulence discrète"
image = pipe(
    prompt=prompt,
    negative_prompt="ugly, deformed, disfigured, poor quality, low resolution",
    num_inference_steps=50,  # More steps for higher quality
    guidance_scale=8.0,     # Increase prompt fidelity
    width=768,              # Adjust image width
    height=768,             # Adjust image height
    generator=generator     # Fixed seed
).images[0]

image.save("high_quality_image.png")
