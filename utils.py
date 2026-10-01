"""Shared helpers: config loading, seeding, device selection."""
from __future__ import annotations
import json, os, random
from pathlib import Path
import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike) -> dict:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")
    with open(p, "r") as f:
        return yaml.safe_load(f)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed python/numpy/torch. Full determinism on GPU is only best-effort."""
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def resolve_device_dtype(dtype_cfg: str = "auto") -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("[WARN] CUDA not available -> running on CPU (very slow). In Colab: Runtime > Change runtime type > GPU.")
    if dtype_cfg == "auto":
        dtype = torch.float16 if device.type == "cuda" else torch.float32
    else:
        dtype = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}[dtype_cfg]
    if device.type == "cpu" and dtype == torch.float16:
        print("[WARN] float16 is unsupported for most CPU ops; using float32."); dtype = torch.float32
    return device, dtype


def save_json(obj: dict, path: str | os.PathLike) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
