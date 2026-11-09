"""Execute an extracted notebook stage in a shared namespace."""

from __future__ import annotations

from collections.abc import MutableMapping
from pathlib import Path
from typing import Any


def run_script(
    path: str | Path,
    namespace: MutableMapping[str, Any],
) -> None:
    """Execute *path* in *namespace*, matching sequential notebook-cell behavior.

    CLAMEA's original cells intentionally share variables such as ``WORK``,
    ``fs_drive``, and ``source_dir``.  Passing the notebook's ``globals()`` keeps
    that contract while allowing each stage to live in a normal Python file.
    """

    script_path = Path(path).resolve()
    if not script_path.is_file():
        raise FileNotFoundError(f"CLAMEA stage does not exist: {script_path}")

    previous_file = namespace.get("__file__")
    namespace["__file__"] = str(script_path)
    try:
        source = script_path.read_text(encoding="utf-8")
        exec(compile(source, str(script_path), "exec"), namespace, namespace)
    finally:
        if previous_file is None:
            namespace.pop("__file__", None)
        else:
            namespace["__file__"] = previous_file

