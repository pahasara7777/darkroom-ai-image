# darkroom-ai-image

FastAPI + static UI for generating images with SDXL and a custom LoRA, run on a Colab T4 GPU.

1. Upload `server.py` and `static/` to `MyDrive/darkroom_app/`.
2. Open `run_darkroom_colab.ipynb` in Colab, set the runtime to T4 GPU.
3. Run cell 2 (install), then cell 3 (starts the server and prints a trycloudflare.com link).

The LoRA is expected at `MyDrive/lora_general/output/pytorch_lora_weights.safetensors`.
