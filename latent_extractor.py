"""Prompt semantic representation (phi), latent generation (Z), decoding and VAE round-trip."""
from __future__ import annotations
import torch


@torch.inference_mode()
def encode_prompt(pipe, prompts: list[str]) -> dict[str, torch.Tensor]:
    """phi = TextEncoder(prompt) using the SD v1.5 CLIP text encoder.

    Returns float32 CPU tensors:
      phi_tokens : [B, 77, 768]  per-token hidden states (what the UNet cross-attends to)
      phi_pooled : [B, 768]      EOS-token summary vector (compact phi for LLSM-ZP)
    """
    tok = pipe.tokenizer(prompts, padding="max_length", max_length=pipe.tokenizer.model_max_length,
                         truncation=True, return_tensors="pt")
    out = pipe.text_encoder(tok.input_ids.to(pipe.device))
    return {"phi_tokens": out.last_hidden_state.float().cpu(),
            "phi_pooled": out.pooler_output.float().cpu(),
            "input_ids": tok.input_ids.cpu()}


@torch.inference_mode()
def generate_latents(pipe, prompts: list[str], cfg: dict, seed: int) -> torch.Tensor:
    """Run the full SD denoising loop and return the final *scaled* latent Z [B,4,H/8,W/8].

    'Scaled' means multiplied by vae.config.scaling_factor (0.18215), the convention used by the UNet.
    """
    d = cfg["diffusion"]
    gen = torch.Generator(device=pipe.device).manual_seed(seed)
    out = pipe(prompt=prompts, height=d["resolution"], width=d["resolution"],
               num_inference_steps=d["num_inference_steps"], guidance_scale=d["guidance_scale"],
               generator=gen, output_type="latent")
    return out.images  # latent tensor on pipe.device, pipe dtype


@torch.inference_mode()
def decode_latents(pipe, z: torch.Tensor) -> torch.Tensor:
    """VAE decoder: scaled latent -> image tensor in [-1, 1], shape [B,3,H,W]."""
    x = pipe.vae.decode(z.to(pipe.vae.dtype) / pipe.vae.config.scaling_factor).sample
    return x.clamp(-1, 1)


@torch.inference_mode()
def encode_image(pipe, x: torch.Tensor) -> torch.Tensor:
    """VAE encoder (posterior mean): image in [-1,1] -> scaled latent. Needed later for secret extraction."""
    z = pipe.vae.encode(x.to(pipe.vae.dtype)).latent_dist.mean
    return z * pipe.vae.config.scaling_factor


def tensor_to_pil(pipe, x: torch.Tensor):
    return pipe.image_processor.postprocess(x, output_type="pil")
