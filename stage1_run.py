"""Stage 1: SD v1.5 -> image + latent Z + prompt representation phi, with compatibility checks.

Usage (Colab):  !python stage1_run.py --config configs/config.yaml
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import torch
from utils import load_config, set_seed, save_json
from diffusion.stable_diffusion import load_stable_diffusion
from diffusion.latent_extractor import (encode_prompt, generate_latents, decode_latents,
                                        encode_image, tensor_to_pil)


def validate(prompts: list[str]) -> None:
    if not prompts or any((not isinstance(p, str)) or not p.strip() for p in prompts):
        raise ValueError("Prompts must be a non-empty list of non-empty strings.")


def check_compat(z: torch.Tensor, phi: dict, cfg: dict) -> dict:
    """Verify tensors satisfy the interface expected by Stage 2 (regions) and LLSM (Stage 5-6)."""
    res, ps, C = cfg["diffusion"]["resolution"], cfg["latent"]["patch_size"], cfg["latent"]["channels"]
    B, c, h, w = z.shape
    checks = {
        "latent_channels_ok": c == C,
        "latent_spatial_ok": h == res // 8 and w == res // 8,
        "latent_divisible_by_patch": h % ps == 0 and w % ps == 0,
        "latent_finite": bool(torch.isfinite(z).all()),
        "phi_tokens_shape_ok": tuple(phi["phi_tokens"].shape[1:]) == (77, 768),
        "phi_pooled_shape_ok": tuple(phi["phi_pooled"].shape[1:]) == (768,),
        "phi_finite": bool(torch.isfinite(phi["phi_tokens"]).all() and torch.isfinite(phi["phi_pooled"]).all()),
        "batch_matches": phi["phi_pooled"].shape[0] == B,
        "n_regions_per_image": (h // ps) * (w // ps),
        "region_feature_dim": c * ps * ps,
    }
    bad = [k for k, v in checks.items() if isinstance(v, bool) and not v]
    if bad:
        raise AssertionError(f"Stage 1 compatibility checks failed: {bad}")
    return checks


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default="configs/config.yaml")
    cfg = load_config(ap.parse_args().config)
    seed = cfg["experiment"]["seed"]; set_seed(seed)
    out = Path(cfg["experiment"]["output_dir"]); out.mkdir(parents=True, exist_ok=True)
    prompts = cfg["prompts"]; validate(prompts)

    pipe, device, dtype = load_stable_diffusion(cfg)
    print(f"Device={device}, dtype={dtype}")
    bs = cfg["diffusion"]["batch_size"]; log = []
    for start in range(0, len(prompts), bs):
        batch = prompts[start:start + bs]
        try:
            phi = encode_prompt(pipe, batch)
            z = generate_latents(pipe, batch, cfg, seed + start)
            x = decode_latents(pipe, z)
            rt = None
            if cfg["latent"].get("roundtrip_check", True):
                z_rt = encode_image(pipe, x)  # encode(decode(Z)) vs Z: relevant for later extraction
                rt = float(torch.mean((z_rt.float() - z.float()) ** 2).cpu())
        except torch.cuda.OutOfMemoryError:
            raise RuntimeError("CUDA out of memory. Lower diffusion.resolution or batch_size, "
                               "keep attention/vae slicing on, or restart the runtime.")
        z_cpu = z.float().cpu()
        checks = check_compat(z_cpu, phi, cfg)
        for i, (p, img) in enumerate(zip(batch, tensor_to_pil(pipe, x))):
            idx = start + i
            img.save(out / f"image_{idx:03d}.png")
            torch.save(z_cpu[i], out / f"latent_{idx:03d}.pt")               # [4,64,64]
            torch.save({"phi_tokens": phi["phi_tokens"][i], "phi_pooled": phi["phi_pooled"][i]},
                       out / f"prompt_emb_{idx:03d}.pt")                       # [77,768], [768]
            print(f"[{idx}] '{p}'\n  image {img.size} | Z {tuple(z_cpu[i].shape)} "
                  f"(mean={z_cpu[i].mean():.3f}, std={z_cpu[i].std():.3f}) | "
                  f"phi_tokens {tuple(phi['phi_tokens'][i].shape)} | phi_pooled {tuple(phi['phi_pooled'][i].shape)} "
                  f"| VAE round-trip latent MSE={rt}")
            log.append({"index": idx, "prompt": p, "seed": seed + start, "roundtrip_latent_mse": rt})
    save_json({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "config": cfg,
               "compat_checks": checks, "items": log}, out / "stage1_log.json")
    print("\nAll Stage 1 compatibility checks passed:", checks)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {type(e).__name__}: {e}", file=sys.stderr); sys.exit(1)
