"""Atomic, all-or-nothing output publication.

A run writes into a *staging* folder next to the requested output folder:

    <out>.partial-<run_id>/      while running
    <out>/                       after success (atomic rename)
    <out>.failed-<run_id>/       after an error (log + manifest kept for diagnosis)
    <out>.cancelled-<run_id>/    after cancellation

So an output folder named exactly as requested only ever contains a
complete, successful run.  Individual files are written to a temporary name
and moved into place with :func:`os.replace` (atomic on POSIX and Windows).
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any

from ..errors import ABPError


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def check_free_disk(folder: Path, min_free_mb: float) -> int:
    """Raise ``RESOURCE_DISK`` if ``folder``'s file system has too little space."""
    folder = folder if folder.exists() else folder.parent
    free = shutil.disk_usage(folder).free
    if free < min_free_mb * 2**20:
        raise ABPError("RESOURCE_DISK",
                       f"only {free / 2**20:.0f} MiB free in {folder}, need {min_free_mb:.0f} MiB",
                       folder=str(folder), free_bytes=free, required_mb=min_free_mb)
    return free


class OutputStage:
    """Staging folder that becomes the output folder only on success."""

    def __init__(self, out_dir: Path, run_id: str, force: bool, min_free_mb: float):
        self.final = Path(out_dir).resolve()
        if self.final.exists() and any(self.final.iterdir()) and not force:
            raise ABPError("OUTPUT_EXISTS", f"output folder {self.final} is not empty",
                           folder=str(self.final))
        self.final.parent.mkdir(parents=True, exist_ok=True)
        check_free_disk(self.final.parent, min_free_mb)
        self.run_id = run_id
        self.force = force
        self.min_free_mb = min_free_mb
        self.dir = self.final.with_name(f"{self.final.name}.partial-{run_id}")
        self.dir.mkdir(parents=False, exist_ok=False)

    def path(self, name: str) -> Path:
        return self.dir / name

    def publish(self) -> Path:
        """Atomically move the staging folder to the final location."""
        check_free_disk(self.final.parent, self.min_free_mb)
        backup = None
        if self.final.exists():
            backup = self.final.with_name(f"{self.final.name}.replaced-{self.run_id}")
            os.replace(self.final, backup)
        os.replace(self.dir, self.final)
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
        self.dir = self.final
        return self.final

    def abandon(self, status: str) -> Path:
        """Rename the staging folder to ``<out>.<status>-<run_id>`` and return it."""
        target = self.final.with_name(f"{self.final.name}.{status}-{self.run_id}")
        try:
            os.replace(self.dir, target)
            self.dir = target
        except OSError:
            pass
        return self.dir
