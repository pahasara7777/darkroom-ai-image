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

class Req(BaseModel):
    prompt: str = Field(min_length=1, max_length=1000)
    negative: str = Field("", max_length=500)
    shape: Literal["1:1", "3:4", "4:3", "9:16", "16:9"] = "1:1"
    steps: int = Field(28, ge=10, le=50)
    style: float = Field(1.0, ge=0, le=1.5)
    seed: int = Field(0, ge=0, lt=2**32)

app = FastAPI()

