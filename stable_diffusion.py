"""Loading pretrained Stable Diffusion v1.5 (established component; nothing is trained here)."""
from __future__ import annotations
import torch
from diffusers import StableDiffusionPipeline
from utils import resolve_device_dtype


def load_stable_diffusion(cfg: dict) -> tuple[StableDiffusionPipeline, torch.device, torch.dtype]:
    """Load tokenizer, text encoder, UNet, VAE and scheduler through one pipeline object.

    Components exposed on the returned pipe: pipe.tokenizer, pipe.text_encoder,
    pipe.unet, pipe.vae (encoder+decoder), pipe.scheduler.
    """
    d = cfg["diffusion"]
    res = int(d["resolution"])
    if res % 8 != 0:
        raise ValueError(f"resolution must be divisible by 8 (VAE downsampling factor); got {res}")
    device, dtype = resolve_device_dtype(d.get("dtype", "auto"))
    kwargs = dict(torch_dtype=dtype, low_cpu_mem_usage=True)
    if d.get("disable_safety_checker", True):
        kwargs.update(safety_checker=None, requires_safety_checker=False)
    try:
        pipe = StableDiffusionPipeline.from_pretrained(d["model_id"], **kwargs)
    except Exception as e:  # network / auth / missing weights
        raise RuntimeError(
            f"Could not load '{d['model_id']}'. Check internet access, the model id, and Hugging Face availability. "
            f"Original error: {e}") from e
    pipe = pipe.to(device)
    if d.get("attention_slicing", True): pipe.enable_attention_slicing()
    if d.get("vae_slicing", True): pipe.enable_vae_slicing()
    for m in (pipe.vae, pipe.text_encoder, pipe.unet):
        m.eval(); m.requires_grad_(False)
    return pipe, device, dtype
