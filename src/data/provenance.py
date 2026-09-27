"""Deterministic provenance helpers (Phase 2).

All hashes are computed from actual file bytes / content, never invented.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA256 hex digest of a file's bytes."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Provenance file not found: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_schema_hash(feature_names: list[str], feature_version: str) -> str:
    """Hash of ordered feature names + version (order matters)."""
    payload = feature_version + "\n" + "\n".join(feature_names)
    return sha256_text(payload)


def git_commit(repo: str | Path = ".") -> str:
    """Short git commit hash, or 'unknown' if git is unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(repo),
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def python_version() -> str:
    return sys.version.replace("\n", " ")


def package_versions(packages: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in packages:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            out[name] = "missing"
    return out


def cuda_info() -> dict[str, str]:
    try:
        import torch

        return {
            "torch": getattr(torch, "__version__", "unknown"),
            "cuda_available": str(torch.cuda.is_available()),
            "cuda_version": str(getattr(torch.version, "cuda", "unknown")),
            "device_count": str(
                torch.cuda.device_count() if torch.cuda.is_available() else 0
            ),
        }
    except Exception:
        return {"torch": "missing"}
