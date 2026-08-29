'''
NOTE: torch.load() default argument is weights_only=True, which is restricted load. It only loads model weights with limited datatypes i.e.,  [int, float, str, bool, bytes, list, tuple, dict, set, tensor]. Therefore, in _pck_rng() we convert numpy ndarray to list, and also in save() we convert obs_rms.mean & var to torch tensor.
If not, you'll get raise pickle.UnpicklingError.
'''

import json
import os
import random
from pathlib import Path
from dataclasses import asdict

import numpy as np
import torch


# torch, numpy, and python random states
def _pack_rng() -> dict:
    k, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [k, keys.tolist(), pos, has_gauss, cached],
        "python": random.getstate(),
    }

def _unpack_rng(rng: dict):
    torch.set_rng_state(rng["torch"])
    if rng["torch_cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["torch_cuda"])
    k, keys, pos, has_gauss, cached = rng["numpy"]
    np.random.set_state((k, np.array(keys, dtype=np.uint32), pos, has_gauss, cached))
    random.setstate(rng["python"])

def save(path, *, agent, optimizer, cfg, iteration: int, global_step: int, obs_rms=None, extra: dict | None = None):
    path = Path(path)
    blob = {
        "agent" : agent.state_dict(),
        "optimizer" : optimizer.state_dict(),
        "cfg_json": json.dumps(asdict(cfg)),
        "iteration": iteration,
        "global_step": global_step,
        "rng": _pack_rng(),
        "extra": extra or {},
    }
    if obs_rms is not None:
        blob["obs_rms"] = {
            "mean": torch.as_tensor(obs_rms.mean, dtype=torch.float64),
            "var": torch.as_tensor(obs_rms.var, dtype=torch.float64),
            "count": float(obs_rms.count),
        }

    # Atomic save: write to temp file, rename temp file if model is successfully saved.
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(blob, tmp)
    os.replace(tmp, path)
    return path

def load(path, map_location="cpu") -> dict:
    return torch.load(path, map_location=map_location)

def restore(ckpt: dict, agent, optimizer=None, obs_rms=None, restore_rng: bool = True) -> tuple[int, int]:
    agent.load_state_dict(ckpt["agent"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if obs_rms is not None and "obs_rms" in ckpt:
        obs_rms.mean = ckpt["obs_rms"]["mean"].numpy()
        obs_rms.var = ckpt["obs_rms"]["var"].numpy()
        obs_rms.count = ckpt["obs_rms"]["count"]
    if restore_rng:
        _unpack_rng(ckpt["rng"])
    return ckpt["iteration"], ckpt["global_step"]

def config_of(ckpt: dict) -> dict:
    return json.loads(ckpt["cfg_json"])
