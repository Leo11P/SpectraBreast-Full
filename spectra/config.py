"""Unified configuration schema for the merged Spectra pipeline.

This module exposes three top-level dataclasses:

- :class:`VisionConfig`        — MASt3R-SfM + ArUco 3D reconstruction (legacy
                                 ``ReconstructionConfig``, kept as an alias for
                                 backward compatibility).
- :class:`RegistrationConfig`  — HSI→mesh registration + spectral point cloud.
- :class:`UnifiedConfig`       — wraps both, plus ``sample_name`` / ``data_root``
                                 / ``results_root`` / ``results_date`` /
                                 ``stages``.

The unified config drives all three CLI sub-commands::

    spectra vision        # uses cfg.vision
    spectra registration  # uses cfg.registration
    spectra full          # uses cfg.vision + cfg.registration

For ``spectra full`` (and for ``spectra registration`` after a successful
``spectra vision`` run), missing per-sample paths are auto-derived from
``sample_name`` + ``data_root`` + ``results_root`` — see
:func:`spectra.orchestrator.resolve_unified_paths`.

All loaders accept the same dotted-path override syntax as the legacy config
(e.g. ``--set vision.aruco.marker_edge_length_mm=30``,
``--set registration.render.resolution_mm_per_px=1.0``).

Marker geometry lives in ONE place: ``vision.aruco.marker_edge_length_mm`` and
``vision.aruco.dictionary``. Registration inherits both (see
:func:`spectra.orchestrator.resolve_unified_paths`); the mirrored fields under
``registration.detection`` exist only as explicit overrides and are ``None`` by
default. Every marker length in the config is in **millimetres**.
"""

from __future__ import annotations

import copy
import json
from datetime import datetime
from pathlib import Path
from typing import Any, List, Literal, Mapping, MutableMapping, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .vision.aruco import ARUCO_DICTIONARIES as _ARUCO_VISION_DICTS


# =============================================================================
# Re-export Vision's existing pydantic classes UNCHANGED.
# We deliberately import them by their original names to avoid touching the
# Vision codebase, then wrap the top-level into VisionConfig below.
# =============================================================================
from .vision.config import (
    ArucoConfig,
    InputConfig,
    Mast3rConfig,
    OutputConfig,
    ReconstructionConfig as VisionConfig,   # public alias
    RerunConfig,
    SurfaceConfig,
)


# =============================================================================
# Registration: pydantic-fied port of the legacy `config.yaml`.
# =============================================================================

# Map from the registration YAML's ArUco dict spelling (uppercase X, e.g. "4X4_50")
# into the vision module's spelling ("4x4_50") to validate the value cheaply.
_REG_TO_VISION_DICT = {k.upper(): k for k in _ARUCO_VISION_DICTS.keys()}


class RegistrationPaths(BaseModel):
    """Per-sample paths used by the registration pipeline.

    Any field left `None` is auto-derived by the orchestrator from
    ``sample_name`` + ``data_root`` + ``results_root`` + ``results_date``
    (see module docstring). Any field you DO set — relative or absolute — is
    used verbatim and never rewritten.
    """

    model_config = ConfigDict(extra="forbid")

    hsi_hdr:      Optional[Path] = None
    mesh:         Optional[Path] = None
    aruco_json:   Optional[Path] = None
    liveview_png: Optional[Path] = None
    output_dir:   Optional[Path] = None


class RegistrationDetection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hsi_extraction_method: Literal["mean", "visible_band", "pca"] = "mean"
    # The HSI cube and the 3D reconstruction see the SAME physical markers, so
    # both of these are inherited from `vision.aruco` when left None — see
    # `spectra.orchestrator.resolve_unified_paths`. Set them only to override.
    aruco_dict: Optional[str] = None
    marker_edge_length_mm: Optional[float] = Field(default=None, gt=0.0)
    use_subpix: bool = True
    subpix_winsize: int = Field(default=5, ge=1)
    # Contrast equalization for the 2D image fed to ArUco detection.
    # 'global' = cv2.equalizeHist (previous behavior), 'clahe' = local, 'none'.
    equalize_method: Literal["global", "clahe", "none"] = "global"
    # More permissive ArUco DetectorParameters for low-contrast/low-res images.
    tune_detector: bool = False

    @field_validator("aruco_dict")
    @classmethod
    def _validate_dict(cls, value: Optional[str]) -> Optional[str]:
        """Accept any capitalization; normalize to the canonical lowercase name."""
        if value is None:
            return None
        canonical = _REG_TO_VISION_DICT.get(value.upper())
        if canonical is None:
            raise ValueError(
                f"Unknown ArUco dictionary {value!r}; "
                f"valid options: {sorted(_ARUCO_VISION_DICTS.keys())}"
            )
        return canonical


class RegistrationRoiAlign(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["sift", "orb"] = "sift"
    min_matches: int = Field(default=20, ge=4)
    ransac_thresh: float = Field(default=3.0, gt=0.0)
    lowe_ratio: float = Field(default=0.75, gt=0.0, lt=1.0)
    fallback_orb: bool = True
    save_match_viz: bool = True


class RegistrationRender(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resolution_mm_per_px: float = Field(default=5.0, gt=0.0)
    resolution_reg_mm_per_px: float = Field(default=0.5, gt=0.0)
    margin_mm: float = Field(default=10.0, ge=0.0)
    ray_chunk_size: int = Field(default=16384, ge=1)
    face_chunk_size: int = Field(default=2048, ge=1)
    sort_rays_for_culling: bool = True


class RegistrationPointcloud(BaseModel):
    model_config = ConfigDict(extra="forbid")

    border_px: int = Field(default=2, ge=0)
    reflectance_norm: bool = True
    pc_chunk_size: int = Field(default=100_000, ge=1)


class RegistrationExport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_pointcloud: bool = True
    ply: bool = True
    npy: bool = True
    csv: bool = False
    csv_max_points: int = Field(default=500_000, ge=1)
    save_images: bool = True


class RegistrationBatch(BaseModel):
    """Batch mode parameters.

    Batch discovery scans ``data_root/input_registration/`` for every
    ``<sample>/<sample>_raw.hdr`` — the same stage-first layout used by
    ``single`` / ``roi`` / ``sweep``. Mesh and ArUco JSON come from that
    sample's vision run under ``results_root/<dd-mm-yyyy>/<sample>/vision/``.
    Per-sample output goes to
    ``results_root/<dd-mm-yyyy>/<sample>/registration/``; the aggregate Excel
    goes to ``results_root/<dd-mm-yyyy>/_batch/batch_summary.xlsx``.
    """

    model_config = ConfigDict(extra="forbid")

    roi_mode: bool = False
    sample_regex: Optional[str] = None   # e.g. "^SB\\d+" to restrict the scan


class RegistrationSweep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    roi_mode: bool = False
    resolution_pairs: List[List[float]] = Field(
        default_factory=lambda: [[5.0, 0.5], [2.0, 0.5], [1.0, 0.5], [0.5, 0.5]],
    )

    @field_validator("resolution_pairs")
    @classmethod
    def _validate_pairs(cls, value: List[List[float]]) -> List[List[float]]:
        if not value:
            raise ValueError("sweep.resolution_pairs must contain at least one [reg, pc] pair.")
        for i, item in enumerate(value):
            if len(item) != 2:
                raise ValueError(f"sweep.resolution_pairs[{i}] must be [reg, pc], got {item!r}")
            if item[0] <= 0 or item[1] <= 0:
                raise ValueError(f"sweep.resolution_pairs[{i}] must be positive, got {item!r}")
        return value


class RegistrationConfig(BaseModel):
    """Top-level registration config (HSI→mesh + spectral point cloud)."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["single", "roi", "batch", "sweep"] = "single"
    paths:       RegistrationPaths       = Field(default_factory=RegistrationPaths)
    detection:   RegistrationDetection   = Field(default_factory=RegistrationDetection)
    roi:         RegistrationRoiAlign    = Field(default_factory=RegistrationRoiAlign)
    render:      RegistrationRender      = Field(default_factory=RegistrationRender)
    pointcloud:  RegistrationPointcloud  = Field(default_factory=RegistrationPointcloud)
    export:      RegistrationExport      = Field(default_factory=RegistrationExport)
    batch:       RegistrationBatch       = Field(default_factory=RegistrationBatch)
    sweep:       RegistrationSweep       = Field(default_factory=RegistrationSweep)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RegistrationConfig":
        for fld in ("hsi_hdr", "mesh", "aruco_json", "liveview_png", "output_dir"):
            v = getattr(self.paths, fld)
            if v is not None:
                setattr(self.paths, fld, Path(v).expanduser())
        return self


# =============================================================================
# UnifiedConfig — top-level wrapper for `spectra full`.
# =============================================================================

class UnifiedConfig(BaseModel):
    """Top-level configuration for the unified Spectra pipeline.

    A standalone Vision config and a standalone Registration config remain
    valid as the ``vision:`` / ``registration:`` sub-trees. The extra
    top-level fields (``sample_name``, ``data_root``, ``results_root``,
    ``stages``) drive automatic path derivation for ``spectra full``.
    """

    model_config = ConfigDict(extra="forbid")

    sample_name: str = "SAMPLE1"
    data_root: Path = Path("DATA")
    results_root: Path = Path("RESULTS")
    #: Run-date folder inserted between ``results_root`` and ``sample_name``:
    #: ``RESULTS/<dd-mm-yyyy>/<sample>/{vision,registration}``. ``None`` means
    #: today. Pin an older date to re-run registration against that day's
    #: vision outputs.
    results_date: Optional[str] = None
    stages: List[Literal["vision", "registration"]] = Field(
        default_factory=lambda: ["vision", "registration"],
    )

    vision:       VisionConfig       = Field(...)
    registration: RegistrationConfig = Field(default_factory=RegistrationConfig)
    #: Robotic acquisition (``spectra/acquisition/acquire.py``). Kept as a
    #: free-form mapping: it is consumed only by the acquisition script (which
    #: runs on the robot PC) and never by vision/registration. ArUco dictionary
    #: and marker edge length are NOT repeated here — acquisition reads them
    #: from ``vision.aruco`` (single source of truth).
    acquisition:  Optional[dict[str, Any]] = None

    @field_validator("results_date")
    @classmethod
    def _validate_results_date(cls, value: Optional[str]) -> Optional[str]:
        if value is None or str(value).strip() == "":
            return None
        value = str(value).strip()
        try:
            datetime.strptime(value, "%d-%m-%Y")
        except ValueError as exc:
            raise ValueError(
                f"results_date must be formatted dd-mm-yyyy, got {value!r}"
            ) from exc
        return value

    @model_validator(mode="after")
    def _resolve_roots(self) -> "UnifiedConfig":
        self.data_root = Path(self.data_root).expanduser()
        self.results_root = Path(self.results_root).expanduser()
        return self

    # ------------------------------------------------------------------
    # Serialization / overrides (same dotted-path API as Vision's config).
    # ------------------------------------------------------------------
    def to_yaml_dict(self) -> dict[str, Any]:
        return _jsonable(self.model_dump(mode="python"))

    def with_overrides(self, overrides: Mapping[str, Any]) -> "UnifiedConfig":
        base = copy.deepcopy(self.model_dump(mode="python"))
        for dotted_key, value in overrides.items():
            _set_dotted(base, dotted_key, value)
        return UnifiedConfig.model_validate(base)


# =============================================================================
# YAML I/O helpers
# =============================================================================

def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, BaseModel):
        return _jsonable(obj.model_dump(mode="python"))
    return obj


def _set_dotted(root: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: Any = root
    for part in parts[:-1]:
        if not isinstance(cursor, MutableMapping) or part not in cursor:
            raise KeyError(f"Unknown config path {dotted_key!r} at segment {part!r}")
        cursor = cursor[part]
    if not isinstance(cursor, MutableMapping) or parts[-1] not in cursor:
        raise KeyError(f"Unknown config path {dotted_key!r}")
    cursor[parts[-1]] = value


def load_unified_config(path: str | Path) -> UnifiedConfig:
    """Load and validate a unified YAML config (must contain `vision:` at least)."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data).__name__}")
    if "vision" not in data:
        raise ValueError(
            f"Config {path} is missing a top-level `vision:` section. "
            "If you have a legacy vision-only config, wrap it under `vision:`."
        )
    return UnifiedConfig.model_validate(data)


def save_unified_config(cfg: UnifiedConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.to_yaml_dict(), f, sort_keys=False, default_flow_style=False)
    return path


def save_unified_config_json(cfg: UnifiedConfig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_yaml_dict(), indent=2), encoding="utf-8")
    return path


# =============================================================================
# Back-compat re-exports
# =============================================================================
# Legacy code (and the docs) reference `ReconstructionConfig` / `load_config` /
# `save_config`. Keep them working by re-exporting under their old names.

ReconstructionConfig = VisionConfig


def load_config(path: str | Path) -> VisionConfig:
    """Legacy loader: parse a *vision-only* YAML and return a VisionConfig.

    Used by tests and any script that hasn't been migrated to the unified
    config yet. Prefer :func:`load_unified_config` going forward.
    """
    from .vision.config import load_config as _legacy_load
    return _legacy_load(path)


def save_config(cfg: VisionConfig, path: str | Path) -> Path:
    from .vision.config import save_config as _legacy_save
    return _legacy_save(cfg, path)


def save_config_json(cfg: VisionConfig, path: str | Path) -> Path:
    from .vision.config import save_config_json as _legacy_save_json
    return _legacy_save_json(cfg, path)


__all__ = [
    # Vision (legacy + alias)
    "ArucoConfig",
    "InputConfig",
    "Mast3rConfig",
    "OutputConfig",
    "RerunConfig",
    "SurfaceConfig",
    "VisionConfig",
    "ReconstructionConfig",
    # Registration
    "RegistrationBatch",
    "RegistrationConfig",
    "RegistrationDetection",
    "RegistrationExport",
    "RegistrationPaths",
    "RegistrationPointcloud",
    "RegistrationRender",
    "RegistrationRoiAlign",
    "RegistrationSweep",
    # Unified
    "UnifiedConfig",
    # I/O
    "load_config",
    "load_unified_config",
    "save_config",
    "save_config_json",
    "save_unified_config",
    "save_unified_config_json",
]