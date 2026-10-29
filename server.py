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

@app.get("/api/health")
def health():
    return {"ok": True}

@app.post("/api/generate")
def generate(r: Req):
    w, h = SIZES[r.shape]
    try:
        with gpu:
            img = pipe(prompt=r.prompt, negative_prompt=r.negative.strip() or NEG,
                       num_inference_steps=r.steps, guidance_scale=6.5, width=w, height=h,
                       cross_attention_kwargs={"scale": r.style},
                       generator=torch.Generator("cuda").manual_seed(r.seed)).images[0]
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        raise HTTPException(503, "The GPU ran out of memory. Try Draft quality or a smaller shape.")
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")

app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
