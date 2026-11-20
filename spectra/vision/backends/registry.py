"""Reconstruction back-ends: names, run-folder tag, lazy loading.

Adding a back-end = one entry in ``BACKENDS`` + a module exposing
``run(cfg, inputs) -> RawReconstruction`` (see ``types.py`` for the contract)
+ its own section ``vision.<name>`` in the config (``vision/config.py``) + its
uv extra in ``pyproject.toml``. The pipeline picks it with
``vision.backend: <name>`` and never knows which one ran.

Environments: each back-end runs in its own virtual environment (``venv``,
built by ``Setup_env.sh <name>``) with ``processing`` + its ``uv_extras``, so
models with clashing dependencies never share one and jobs of different
models never re-sync each other's environment.

Pure Python on purpose (no numpy / torch): ``acquisition/remote.py`` loads this
file by path on the robot PC to name the run folders on Mufasa.
"""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Tuple


@dataclass(frozen=True)
class BackendSpec:
    name: str
    #: module with ``run(cfg, inputs) -> RawReconstruction``
    module: str
    #: key(s) of ``vision.<name>`` that select a variant of the method (they go
    #: in the run-folder tag: two variants of one model never share a folder;
    #: a value "none" is left out of the tag)
    variant_key: str | Tuple[str, ...] | None = None
    #: the value(s) when ``variant_key`` is not in the YAML (= the config defaults)
    default_variant: str | Tuple[str, ...] | None = None
    #: importable modules checked before a local run
    requires: Tuple[str, ...] = ()
    #: how to install what is missing
    install_hint: str = ""
    #: uv extras (pyproject.toml) installed on top of ``processing``
    uv_extras: Tuple[str, ...] = ()
    #: its virtual environment, relative to the repo (UV_PROJECT_ENVIRONMENT)
    venv: str = ".venv"


DEFAULT_BACKEND = "mast3r"

BACKENDS: dict[str, BackendSpec] = {
    "mast3r": BackendSpec(
        name="mast3r",
        module="spectra.vision.backends.mast3r_backend",
        variant_key="pipeline_variant",
        default_variant="sfm",
        requires=("torch", "PIL", "roma"),
        install_hint="bash Setup_env.sh mast3r (README §1.3)",
        uv_extras=("mast3r",),
        venv=".venv",
    ),
    "vggt_omega": BackendSpec(
        name="vggt_omega",
        module="spectra.vision.backends.vggt_omega_backend",
        variant_key="variant",
        default_variant="512",
        requires=("torch", "PIL", "vggt_omega", "huggingface_hub"),
        install_hint="bash Setup_env.sh vggt_omega (README §1.3, weights §1.6)",
        uv_extras=("vggt-omega",),
        venv=".venv-vggt_omega",
    ),
    "pi3": BackendSpec(
        name="pi3",
        module="spectra.vision.backends.pi3_backend",
        variant_key=("variant", "conditioning"),
        default_variant=("x", "none"),
        requires=("torch", "PIL", "pi3", "huggingface_hub"),
        install_hint="bash Setup_env.sh pi3 (README §1.3)",
        uv_extras=("pi3",),
        venv=".venv-pi3",
    ),
    "da3": BackendSpec(
        name="da3",
        module="spectra.vision.backends.da3_backend",
        variant_key=("variant", "conditioning"),
        default_variant=("nested-giant-large-1.1", "none"),
        requires=("torch", "PIL", "depth_anything_3"),
        install_hint="bash Setup_env.sh da3 (README §1.3)",
        uv_extras=("da3",),
        venv=".venv-da3",
    ),
    "mapanything": BackendSpec(
        name="mapanything",
        module="spectra.vision.backends.mapanything_backend",
        variant_key=("variant", "conditioning"),
        default_variant=("nc", "none"),
        requires=("torch", "PIL", "mapanything"),
        install_hint="bash Setup_env.sh mapanything (README §1.3)",
        uv_extras=("mapanything",),
        venv=".venv-mapanything",
    ),
}


def backend_spec(name: str) -> BackendSpec:
    key = str(name or DEFAULT_BACKEND).strip().lower()
    if key not in BACKENDS:
        raise ValueError(f"vision.backend {name!r} unknown; available: {sorted(BACKENDS)}")
    return BACKENDS[key]


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def model_tag(vision: Any) -> str:
    """Name of the model that made a run: ``<backend>[-<variant>]``, e.g.
    ``mast3r-sfm``. Results go to ``RESULTS/<date>/<sample>/<tag>/``.

    ``vision`` is the ``vision:`` section, as a dict (raw YAML) or as the
    validated config object."""
    spec = backend_spec(_get(vision, "backend") or DEFAULT_BACKEND)
    tag = spec.name
    if spec.variant_key:
        keys = (spec.variant_key,) if isinstance(spec.variant_key, str) else tuple(spec.variant_key)
        defaults = spec.default_variant
        defaults = (defaults,) if not isinstance(defaults, tuple) else defaults
        section = _get(vision, spec.name)
        for key, default in zip(keys, defaults):
            value = _get(section, key) or default
            if value and str(value).lower() != "none":
                tag += f"-{value}"
    return str(tag).strip().lower().replace(" ", "_").replace("/", "_")


def load_backend(name: str) -> Callable[..., Any]:
    """``run(cfg, inputs)`` of back-end ``name`` (imported only now: the other
    back-ends' packages need not be installed)."""
    spec = backend_spec(name)
    try:
        module = importlib.import_module(spec.module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Back-end {spec.name!r} cannot be imported ({exc}). Install it: {spec.install_hint}"
        ) from exc
    return module.run


def _mast3r_present() -> bool:
    repo_root = Path(__file__).resolve().parents[3]
    return (repo_root / "mast3r").is_dir() or importlib.util.find_spec("mast3r") is not None


#: extra checks beyond ``requires`` (e.g. a repo cloned next to the package)
_EXTRA_CHECKS: dict[str, Tuple[Tuple[str, Callable[[], bool]], ...]] = {
    "mast3r": (("mast3r (clone it in the repo, README §1.3)", _mast3r_present),),
}


def missing_requirements(name: str) -> list[str]:
    """What back-end ``name`` needs and this environment lacks (empty = ok)."""
    spec = backend_spec(name)
    missing = [m for m in spec.requires if importlib.util.find_spec(m) is None]
    missing += [label for label, ok in _EXTRA_CHECKS.get(spec.name, ()) if not ok()]
    return missing


__all__ = [
    "BACKENDS",
    "DEFAULT_BACKEND",
    "BackendSpec",
    "backend_spec",
    "load_backend",
    "missing_requirements",
    "model_tag",
]
