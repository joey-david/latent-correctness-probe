from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional


def _run_text(args: Iterable[str], cwd: Optional[Path] = None) -> Optional[str]:
    try:
        completed = subprocess.run(
            list(args),
            cwd=str(cwd) if cwd else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except Exception:
        return None
    return completed.stdout.strip()


def git_state(repo_root: Path) -> Dict[str, Any]:
    commit = _run_text(["git", "rev-parse", "HEAD"], cwd=repo_root)
    status = _run_text(["git", "status", "--short"], cwd=repo_root) or ""
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_short": status.splitlines(),
    }


def gpu_state() -> Dict[str, Any]:
    try:
        import torch

        if torch.cuda.is_available():
            index = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(index)
            return {
                "available": True,
                "source": "torch",
                "index": int(index),
                "name": props.name,
                "total_memory_mb": int(props.total_memory // (1024 * 1024)),
            }
    except Exception:
        pass

    query = _run_text(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    if query:
        first = query.splitlines()[0]
        parts = [part.strip() for part in first.split(",")]
        if len(parts) >= 2:
            return {
                "available": True,
                "source": "nvidia-smi",
                "name": parts[0],
                "total_memory_mb": int(float(parts[1])),
            }
    return {"available": False, "source": "none", "name": None, "total_memory_mb": None}


def build_manifest(
    *,
    command: Iterable[str],
    config_path: Optional[str],
    config: Mapping[str, Any],
    artifact_paths: Mapping[str, Any],
    repo_root: Path,
    split_manifest: Optional[str] = None,
    dataset_version: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "pid": os.getpid(),
        "command_line": list(command),
        "config_path": config_path,
        "git": git_state(repo_root),
        "gpu": gpu_state(),
        "model_id": config.get("model", {}).get("id"),
        "dataset": config.get("dataset", {}).get("name"),
        "dataset_version": dataset_version,
        "split_manifest": split_manifest,
        "artifact_paths": dict(artifact_paths),
        "run": dict(config.get("run", {})),
    }


def write_manifest(payload: Mapping[str, Any], path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out
