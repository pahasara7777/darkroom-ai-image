import glob, io, os, threading, torch
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from diffusers import AutoencoderKL, StableDiffusionXLPipeline

HERE = os.path.dirname(os.path.abspath(__file__))
LORA = os.environ.get("LORA_PATH") or glob.glob(os.path.join(HERE, "*.safetensors"))[0]
