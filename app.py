import os
import gdown
import gradio as gr
import torch
from diffusers import StableDiffusionPipeline

# Download model if not already present
if not os.path.exists("model.safetensors"):
    gdown.download("https://drive.google.com/uc?id=1ErCyGDdmZl8056BiBsfWbDj02zA_sgC-", "model.safetensors", quiet=False)

# Load the model
pipe = StableDiffusionPipeline.from_single_file("model.safetensors", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")

# Gradio interface
def generate(prompt):
    result = pipe(prompt).images[0]
    return result

gr.Interface(fn=generate, inputs="text", outputs="image", title="My AI Model").launch()
