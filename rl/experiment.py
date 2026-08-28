'''
Every experiment tracks couple of things:
config.json - the config used to run the experiment
commit.txt - the hash of the git commit used to run the experiment
dirty.patch - Uncommitted edits to the repo in patch format.
meta.json - the meta information about the run.
'''
from __future__ import annotations
import json
import os
import sys
import platform
import socket
import subprocess
import datetime
from dataclasses import asdict
from pathlib import Path

import random
import torch
import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_MAX_UNTRACKED_BYTES = 1_000_000  # Bigger than this are named in the patch but not embedded.

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def _git(*args: str, ok: tuple[int, ...] = (0,)) -> str:
    '''Run a git command in the repo. '''
    try:
        r = subprocess.run(["git", *args], cwd=_REPO, capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.SubprocessError):
        return ""
    return r.stdout if r.returncode in ok else ""

def _untracked() -> list[str]:
    return _git("ls-files", "--others", "--exclude-standard").split()

def _dirty_patch() -> str:
    '''Track modifications + contents of untracked files.
    `git diff HEAD` shows modifications, not untracked files.
    `git diff --no-index /dev/null <file> shows untracked files.'''
    parts = [_git("diff", "HEAD")]

    for file in _untracked():
        path = Path(_REPO) / file
        try:
            if path.stat().st_size > _MAX_UNTRACKED_BYTES:
                parts.append(f"# untracked, too large to embed: {file} \n")
                continue
        except OSError:
            continue
        parts.append(_git("diff", "--no-index", "/dev/null", file, ok=(0, 1)))
    return "\n".join(parts)

def create_run_dir(cfg, root="runs") -> Path:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    parts = [stamp] + ([cfg.tag] if cfg.tag else []) + [f"s{cfg.seed}"]
    base = Path(root) / cfg.env_id / "_".join(parts)
    rundir, n = base, 1
    while rundir.exists():
        rundir = base.with_name(f"{base.name}_{n}")
        n += 1

    rundir.mkdir(parents=True, exist_ok=False)
    (rundir / "checkpoints").mkdir()
    (rundir / "videos").mkdir()
    (rundir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")

    commit = _git("rev-parse", "HEAD").strip() or "unknown"
    (rundir / "commit.txt").write_text(commit + "\n")
    (rundir / "dirty.patch").write_text(_dirty_patch())

    meta = {
        "started": datetime.datetime.now().isoformat(timespec="seconds"),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "git_commit": commit,
        "git_untracked": _untracked(),
    }

    (rundir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return rundir
