"""Small terminal progress indicators without an extra runtime dependency."""
from __future__ import annotations

import sys


def update_progress(label: str, completed: int, total: int, detail: str = "") -> None:
    """Render one in-place progress update when stderr is an interactive terminal."""
    if not sys.stderr.isatty():
        return
    total = max(1, total)
    completed = min(max(0, completed), total)
    width = 24
    filled = int(width * completed / total)
    bar = "=" * filled + ">" * (filled < width) + " " * max(0, width - filled - 1)
    suffix = f" | {detail}" if detail else ""
    sys.stderr.write(f"\r{label} [{bar}] {completed}/{total}{suffix}")
    sys.stderr.flush()


def finish_progress() -> None:
    """End the current in-place progress line."""
    if sys.stderr.isatty():
        sys.stderr.write("\n")
        sys.stderr.flush()
