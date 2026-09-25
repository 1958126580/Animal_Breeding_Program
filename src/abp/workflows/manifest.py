"""Run manifest: the traceable record of one execution (module M25).

Field names follow ``contracts/run_manifest.schema.json`` (derived from the
project's run-record template).  Values that could not be determined are
``null`` together with a reason; nothing is invented.
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import os
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import scipy

from .. import __version__

MANIFEST_SCHEMA_VERSION = "1.0"


def new_run_id() -> str:
    """Timestamped, collision-resistant run identifier."""
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


def code_provenance() -> dict[str, Any]:
    """Git commit and working-tree patch hash of the ABP source, if available."""
    src = Path(__file__).resolve().parents[3]
    out: dict[str, Any] = {"abp_version": __version__, "commit": None,
                           "working_tree_patch_sha256": None, "reason": None}
    try:
        commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                                text=True, timeout=10)
        if commit.returncode != 0:
            out["reason"] = "not a git checkout (installed package)"
            return out
        out["commit"] = commit.stdout.strip()
        diff = subprocess.run(["git", "-C", str(src), "diff", "HEAD", "--", "src"],
                              capture_output=True, timeout=30)
        out["working_tree_patch_sha256"] = hashlib.sha256(diff.stdout).hexdigest()
        out["working_tree_clean"] = diff.stdout == b""
    except (OSError, subprocess.SubprocessError) as exc:
        out["reason"] = f"git unavailable: {exc.__class__.__name__}"
    return out


def environment() -> dict[str, Any]:
    """Software and hardware description (no host names or user names)."""
    blas = None
    try:
        cfg = np.show_config(mode="dicts")  # numpy >= 1.25
        b = cfg.get("Build Dependencies", {}).get("blas", {})
        blas = {"name": b.get("name"), "version": b.get("version")}
    except Exception:  # pragma: no cover - older numpy
        blas = None
    from ..core.pedigree import native_kernel_available
    return {
        "os": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "blas": blas,
        "cpu_count": os.cpu_count(),
        "gpu": None,
        "native_kernel": native_kernel_available(),
        "dtype": "float64",
    }


def peak_rss_bytes() -> int | None:
    """Peak resident memory of this process, or None where not measurable."""
    try:
        import resource
    except ImportError:  # Windows without psutil
        return None
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def sha256_array(*arrays: np.ndarray | list) -> str:
    """Hash of one or more arrays / string lists (for sample-mapping hashes)."""
    h = hashlib.sha256()
    for a in arrays:
        if isinstance(a, np.ndarray):
            h.update(np.ascontiguousarray(a).tobytes())
        else:
            h.update("\x1f".join(map(str, a)).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
