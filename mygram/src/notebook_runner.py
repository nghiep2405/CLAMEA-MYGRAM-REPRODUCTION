"""Runtime support for the code-only notebook workflows."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any


def run_script(
    path: str | Path,
    namespace: MutableMapping[str, Any],
) -> None:
    """Execute an extracted notebook step in the notebook's shared namespace."""

    script_path = Path(path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"MyGram notebook step does not exist: {script_path}")
    source = script_path.read_text(encoding="utf-8")
    exec(compile(source, str(script_path), "exec"), namespace, namespace)

