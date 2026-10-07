"""Locate optional local dependencies and CUDA libraries without system changes."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
_dll_handles: list[object] = []


def configure_dependencies() -> None:
    """Support the local Windows bundle while preferring an active virtualenv."""
    dependencies = ROOT / "deps"
    in_virtualenv = sys.prefix != sys.base_prefix
    if dependencies.is_dir() and not in_virtualenv:
        sys.path.insert(0, str(dependencies))


def configure_cuda_libraries() -> None:
    """Keep Windows DLL directories registered for the lifetime of this process."""
    if os.name != "nt":
        return

    search_roots: list[Path] = []
    for package in ("ctranslate2", "nvidia"):
        specification = importlib.util.find_spec(package)
        if specification and specification.submodule_search_locations:
            search_roots.extend(Path(path) for path in specification.submodule_search_locations)

    directories = sorted({dll.parent for root in search_roots for dll in root.rglob("*.dll")})
    if directories:
        os.environ["PATH"] = (
            os.pathsep.join(map(str, directories)) + os.pathsep + os.environ.get("PATH", "")
        )
    for directory in directories:
        _dll_handles.append(os.add_dll_directory(str(directory)))


def resolve_model(model_name: str) -> str:
    """Use an explicit model directory or a cached snapshot before downloading."""
    explicit_path = Path(model_name).expanduser()
    if explicit_path.is_dir():
        return str(explicit_path.resolve())

    for cache_directory in (ROOT / "models", ROOT / "modelos"):
        repository = cache_directory / f"models--Systran--faster-whisper-{model_name}"
        reference = repository / "refs" / "main"
        if reference.is_file():
            snapshot = repository / "snapshots" / reference.read_text(encoding="utf-8").strip()
            if (snapshot / "model.bin").is_file():
                return str(snapshot)
    return model_name
