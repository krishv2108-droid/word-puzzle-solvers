"""Crash-safe cache files.

Caches in this project (pattern matrix, partition table, embeddings, LLM replies) are built by
long-running jobs. Writing straight to the final path means a killed run leaves a truncated file
that then fails to load on every later run. Here a file is written to a temporary name in the same
directory and renamed into place, which is atomic: readers see the old file or the complete new one.
Loading treats an unreadable cache as a miss so it is simply rebuilt.
"""
from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import numpy as np


@contextlib.contextmanager
def atomic_open(path: Path, mode: str = "wb") -> Iterator:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, mode) as f:
            yield f
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def save_npy(path: Path, array: np.ndarray) -> bool:
    """Atomically save `array`; returns False (and does not raise) if the location is not writable."""
    try:
        with atomic_open(path) as f:
            np.save(f, array)
        return True
    except OSError:
        return False


def load_npy(path: Path, shape: tuple[int, ...] | None = None) -> np.ndarray | None:
    """Load an .npy cache, or None if it is missing, unreadable, truncated or the wrong shape."""
    try:
        array = np.load(path)
    except (OSError, ValueError, EOFError):
        return None
    if shape is not None and array.shape != shape:
        return None
    return array
