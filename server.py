import glob, io, os, threading, torch
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
LORA = os.environ.get("LORA_PATH") or glob.glob(os.path.join(HERE, "*.safetensors"))[0]
NEG = "lowres, blurry, bad anatomy, bad hands, extra fingers, deformed, jpeg artifacts, text, watermark"
SIZES = {"1:1": (1024, 1024), "3:4": (896, 1152), "4:3": (1152, 896), "9:16": (768, 1344), "16:9": (1344, 768)}

vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", vae=vae, torch_dtype=torch.float16, variant="fp16"
).to("cuda")  # needs an NVIDIA GPU
pipe.load_lora_weights(os.path.dirname(LORA), weight_name=os.path.basename(LORA))
gpu = threading.Lock()  # one image at a time; other requests wait their turn

