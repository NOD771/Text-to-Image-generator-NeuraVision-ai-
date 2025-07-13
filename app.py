import torch
from diffusers import StableDiffusionXLPipeline, LCMScheduler
import gradio as gr

base_model_id = "stabilityai/stable-diffusion-xl-base-1.0"
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    variant="fp16" if torch.cuda.is_available() else None,
    use_safetensors=True
)

# Your LoRA HF repo
pipe.load_lora_weights("Wiuhh/NeuraVisionlorar1")
pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
pipe.to("cuda" if torch.cuda.is_available() else "cpu")

def generate(prompt):
    image = pipe(prompt=prompt, num_inference_steps=25, guidance_scale=7.5).images[0]
    return image

gr.Interface(
    fn=generate,
    inputs=gr.Textbox(label="Prompt"),
    outputs=gr.Image(label="Result"),
    title="LoRA SDXL Generator"
).launch()
