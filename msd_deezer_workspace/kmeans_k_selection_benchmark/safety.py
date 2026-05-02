from __future__ import annotations

from pathlib import Path


WORKSPACE_DIR = Path(__file__).resolve().parent


def assert_inside_workspace(path: str | Path, label: str = "path") -> Path:
    """Resolve a path and refuse writes outside this benchmark folder."""
    resolved = Path(path).resolve()
    root = WORKSPACE_DIR.resolve()
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"Refusing to write {label} outside the K-selection benchmark root.\n"
            f"  attempted: {resolved}\n"
            f"  benchmark root: {root}"
        ) from exc
    return resolved
