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
(e.g. ``--set vision.aruco.min_views_per_marker=3``,
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

#: The one registration key that must never be user-set (single marker source).
REGISTRATION_MARKER_EDGE_KEY = "registration.detection.marker_edge_length_mm"


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
from .vision.config import DEFAULT_CONFIG_PATH, with_default_marker_edge


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


class RegistrationRerun(BaseModel):
    """Spectral Rerun recording written after the point-cloud export.

    ``<output_dir>/spectral.rrd`` (+ ``spectral.rbl``): top view = one band,
    chosen on the ``wavelength_nm`` timeline, in ``colormap``; bottom view = PCA
    PC1/PC2/PC3 as RGB. Both views spin around the sample. See :mod:`spectra.registration.rerun_spectral`.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    max_points: Optional[int] = Field(default=150_000, ge=1000)   # None = all
    band_step: int = Field(default=2, ge=1)
    colormap: Literal["gray", "viridis", "turbo", "inferno", "magma",
                      "plasma", "cividis", "jet"] = "gray"     # band view only
    point_radius_px: float = Field(default=2.0, gt=0.0)
    spin_speed: float = Field(default=0.3, ge=0.0)
    save_rbl: bool = True
    rrd_basename: str = "spectral.rrd"
    rbl_basename: str = "spectral.rbl"

    @field_validator("rrd_basename", "rbl_basename")
    @classmethod
    def _plain_basename(cls, value: str, info) -> str:
        suffix = ".rrd" if info.field_name == "rrd_basename" else ".rbl"
        if Path(value).name != value or not value.endswith(suffix):
            raise ValueError(
                f"registration.rerun.{info.field_name} must be a plain file name "
                f"ending in {suffix}, got {value!r}"
            )
        return value


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
    rerun:       RegistrationRerun       = Field(default_factory=RegistrationRerun)
    batch:       RegistrationBatch       = Field(default_factory=RegistrationBatch)
    sweep:       RegistrationSweep       = Field(default_factory=RegistrationSweep)

    @model_validator(mode="after")
    def _resolve_paths(self) -> "RegistrationConfig":
        for fld in ("hsi_hdr", "mesh", "aruco_json", "liveview_png", "output_dir"):
            v = getattr(self.paths, fld)
            if v is not None:
                setattr(self.paths, fld, Path(v).expanduser())
        return self

    @model_validator(mode="after")
    def _rerun_needs_pointcloud(self) -> "RegistrationConfig":
        exp = self.export
        if self.rerun.enabled and not (exp.save_pointcloud and (exp.ply or exp.npy)):
            raise ValueError(
                "registration.rerun.enabled=true reads the exported point cloud: set "
                "registration.export.save_pointcloud=true and ply or npy to true "
                "(or registration.rerun.enabled=false)."
            )
        return self


# =============================================================================
# Calibration — hand-eye + intrinsics of the robot camera (spectra calibrate).
# Robot / camera stream are NOT repeated: they come from acquisition.robot and
# acquisition.camera. Data layout is fixed: <data_root>/camera_calibration/.
# =============================================================================

class CalibrationBoard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chessboard_size: List[int]                   # INNER corners (columns, rows)
    square_size_mm: float = Field(gt=0.0)        # measured square side

    @field_validator("chessboard_size")
    @classmethod
    def _two_sizes(cls, value: List[int]) -> List[int]:
        if len(value) != 2 or min(value) < 2:
            raise ValueError(f"calibration.board.chessboard_size must be [cols, rows] "
                             f"of inner corners (>= 2), got {value!r}")
        return value


class CalibrationIntrinsics(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["realsense", "opencv", "file"] = "realsense"
    file: Optional[str] = None                   # JSON {"K", "D"}, source = file only
    min_views_warn: int = Field(default=10, ge=1)

    @model_validator(mode="after")
    def _file_when_needed(self) -> "CalibrationIntrinsics":
        if self.source == "file" and not self.file:
            raise ValueError("calibration.intrinsics.source = file needs calibration.intrinsics.file")
        return self


class CalibrationHandEye(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: Literal["PARK", "TSAI", "HORAUD", "ANDREFF", "DANIILIDIS"] = "PARK"
    outlier_sigma: float = Field(default=0.5, ge=0.0)
    max_reproj_err_px: float = Field(default=0.4, gt=0.0)
    only_images: List[str] = Field(default_factory=list)


class CalibrationAcquisition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_calib_target: int = Field(default=30, ge=1)
    n_val_target: int = Field(default=20, ge=1)
    preview_margin_px: int = Field(default=60, ge=0)


class CalibrationValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    save_overlay: bool = True


class CalibrationCamera(BaseModel):
    """Colour-sensor settings for the calibration views only (resolution, fps
    and frame averaging come from acquisition.camera)."""

    model_config = ConfigDict(extra="forbid")

    frame_timeout_ms: int = Field(default=10000, ge=1)
    auto_exposure: bool = False
    exposure: Optional[float] = 120
    gain: Optional[float] = 16
    auto_white_balance: bool = True
    white_balance: Optional[float] = None


class CalibrationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    board:       CalibrationBoard
    intrinsics:  CalibrationIntrinsics  = Field(default_factory=CalibrationIntrinsics)
    hand_eye:    CalibrationHandEye     = Field(default_factory=CalibrationHandEye)
    acquisition: CalibrationAcquisition = Field(default_factory=CalibrationAcquisition)
    validation:  CalibrationValidation  = Field(default_factory=CalibrationValidation)
    camera:      CalibrationCamera      = Field(default_factory=CalibrationCamera)


# =============================================================================
# Mufasa — where vision / registration run.
# =============================================================================

# =============================================================================
# Validation (spectra validate) — ground-truth phantoms, protocol phase 1 (vision).
# Data layout is fixed: <data_root>/validation/ (marker table + STL assemblies).
# =============================================================================

class ValidationPhantom(BaseModel):
    """One phantom: the STL of the cassette with this phantom in the recess
    (CAD frame, mm, Z up), in <data_root>/validation/."""

    model_config = ConfigDict(extra="forbid")

    stl: str
    #: XY box [xmin, xmax, ymin, ymax] of the evaluated region, CAD frame (mm).
    #: null = the STL above the marker plane + validation.vision.roi_margin_mm.
    roi_mm: Optional[List[float]] = None

    @field_validator("roi_mm")
    @classmethod
    def _box(cls, v):
        if v is not None and (len(v) != 4 or v[0] >= v[1] or v[2] >= v[3]):
            raise ValueError(f"roi_mm must be [xmin, xmax, ymin, ymax], got {v}")
        return v


class ValidationICP(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    max_corr_mm: float = Field(default=2.0, gt=0.0)    # pairs farther than this are ignored
    iters: int = Field(default=50, ge=1)


class ValidationFiducials(BaseModel):
    """Per-bersaglio error (validation/fiducials.py): the bersagli, holes of the
    phantom STL, looked for in the photos of the vision run near their expected
    position, then hit on the reconstructed mesh."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    #: bright = bersaglio lighter than the phantom (white on black) | dark.
    polarity: Literal["bright", "dark"] = "bright"
    search_radius_mm: float = Field(default=2.0, gt=0.0)        # around the expected position
    max_view_angle_deg: float = Field(default=60.0, gt=0.0, lt=90.0)   # view vs hole axis
    min_contrast: float = Field(default=20.0, ge=0.0)          # spot peak - background, grey 0-255
    outlier_mm: float = Field(default=1.0, gt=0.0)             # views this far from the median: dropped
    min_views: int = Field(default=2, ge=1)
    save_sheet: bool = True                                    # fiducials.png


class ValidationVision(BaseModel):
    """Phase 1: the reconstruction against the phantom STL."""

    model_config = ConfigDict(extra="forbid")

    source: Literal["mesh", "cloud"] = "mesh"     # surface_mesh.ply (what registration uses) | cloud.ply
    primary: Literal["frame", "kabsch", "icp"] = "icp"
    sample_spacing_mm: float = Field(default=0.25, gt=0.0)
    roi_margin_mm: float = Field(default=5.0, ge=0.0)
    occluder_margin_mm: float = Field(default=20.0, ge=0.0)
    icp: ValidationICP = Field(default_factory=ValidationICP)
    thresholds_mm: List[float] = Field(default_factory=lambda: [1.0, 2.0])
    height_bin_mm: float = Field(default=2.0, gt=0.0)
    visibility_tol_mm: float = Field(default=1.0, ge=0.0)
    error_map_max_mm: float = Field(default=2.0, gt=0.0)
    save_error_map: bool = True
    save_rrd: bool = True
    fiducials: ValidationFiducials = Field(default_factory=ValidationFiducials)


class ValidationConfig(BaseModel):
    """``validation:`` section (spectra validate, stage ``validation``)."""

    model_config = ConfigDict(extra="forbid")

    fixture_table: str = "Tabella_marker_cassetta.xlsx"
    phantoms: dict[str, ValidationPhantom] = Field(default_factory=dict)
    #: sample_name -> phantom, for samples not named as sample_pattern.
    cases: dict[str, str] = Field(default_factory=dict)
    #: Groups phantom (required), pose, rep: pose/rep pair the same case across models.
    sample_pattern: str = r"^(?P<phantom>[^_]+)_(?P<pose>[^_]+)_(?P<rep>[^_]+)$"
    #: Name of the reconstruction model in the summary; null = backend + pipeline variant.
    model_label: Optional[str] = None
    vision: ValidationVision = Field(default_factory=ValidationVision)

    @field_validator("sample_pattern")
    @classmethod
    def _pattern(cls, v: str) -> str:
        import re

        if "phantom" not in re.compile(v).groupindex:
            raise ValueError("validation.sample_pattern needs a (?P<phantom>...) group")
        return v


class MufasaConfig(BaseModel):
    """``mufasa:`` section: run vision / registration on this PC or on Mufasa.

    ``enabled: true`` → ``spectra vision``, ``spectra registration`` and the
    vision / registration part of ``spectra full`` are submitted to Mufasa as
    a SLURM job (``spectra/acquisition/remote.py``): inputs and results live in
    ``repo_dir`` on Mufasa (DATA/, RESULTS/) and only ``pull_patterns`` come back. ``enabled: false`` → everything runs on this PC. The acquisition
    always runs on this PC. The run config sent with each job has no
    ``mufasa:`` section, so the job itself always runs locally on the GPU node.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    host: Optional[str] = None
    port: int = Field(default=22, ge=1, le=65535)
    user: Optional[str] = None
    ssh_key_path: Optional[str] = None          # None = ask for the password
    repo_dir: Optional[str] = None              # repo clone on Mufasa: DATA/ + RESULTS/ there
    slurm_bin_dirs: List[str] = Field(default_factory=list)
    partition: str = "jobs"
    qos: str = "gpuwide"
    gres: str = "gpu:4g.20gb:1"
    cpus: int = Field(default=8, ge=1)
    mem: str = "64G"
    time: str = "12:00:00"
    job_name: str = "spectra"
    wait: bool = True
    poll_s: int = Field(default=20, ge=1)
    pull_results: bool = True
    #: Downloaded after the job, relative to RESULTS/<date>/<sample>/ on Mufasa
    #: ("dir/glob"); everything else stays on Mufasa.
    pull_patterns: List[str] = Field(default_factory=lambda: [
        "registration/*.xlsx", "registration/*.rrd", "registration/*.rbl",
        "validation/vision/*"])

    @model_validator(mode="after")
    def _required_when_enabled(self) -> "MufasaConfig":
        if self.enabled:
            missing = [k for k in ("host", "user", "repo_dir")
                       if not getattr(self, k)]
            if missing:
                raise ValueError(f"mufasa.enabled=true needs mufasa.{missing} to be set.")
        return self


# =============================================================================
# UnifiedConfig — top-level wrapper for `spectra full`.
# =============================================================================

class UnifiedConfig(BaseModel):
    """Top-level configuration for the unified Spectra pipeline.

    A standalone Vision config and a standalone Registration config remain
    valid as the ``vision:`` / ``registration:`` sub-trees. The extra
    top-level fields (``sample_name``, ``data_root``, ``results_root``,
    ``stages``) drive automatic path derivation for ``spectra full``;
    ``mufasa`` decides where vision / registration run.
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
    #: Stages run by ``spectra full``. "acquisition" (robot, always on this
    #: PC) is opt-in so a config without it never moves the robot.
    stages: List[Literal["acquisition", "vision", "registration", "validation"]] = Field(
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
    #: Where vision / registration run (this PC or Mufasa). ``None`` = this PC.
    mufasa:       Optional[MufasaConfig] = None
    #: Hand-eye + intrinsics calibration of the robot camera (spectra calibrate).
    calibration:  Optional[CalibrationConfig] = None
    #: Validation against ground-truth phantoms (spectra validate, stage "validation").
    validation:   Optional[ValidationConfig] = None

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
    def _check_stages_and_sections(self) -> "UnifiedConfig":
        if not self.stages:
            raise ValueError("stages must list at least one of acquisition / vision / "
                             "registration / validation.")
        if "validation" in self.stages and self.validation is None:
            raise ValueError("stages contains 'validation' but the config has no `validation:` section.")
        if isinstance(self.acquisition, dict) and "remote" in self.acquisition:
            raise ValueError(
                "acquisition.remote has moved to the top-level `mufasa:` section "
                "(with `enabled: true|false`); `submit_after_acquisition` no longer "
                "exists: `spectra full` submits after the acquisition when "
                "mufasa.enabled is true."
            )
        return self

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
    # The marker edge length is set ONLY in configs/default.yaml: any other
    # config (e.g. the run_config.yaml written by acquisition/remote.py) gets
    # it from there, and must not set it itself.
    if path.resolve() != DEFAULT_CONFIG_PATH:
        data["vision"] = with_default_marker_edge(data["vision"], f"{path} (vision)")
    reg_detection = (data.get("registration") or {}).get("detection") or {}
    if reg_detection.get("marker_edge_length_mm") is not None:
        raise ValueError(
            f"Config {path}: {REGISTRATION_MARKER_EDGE_KEY} cannot be set. The marker "
            f"edge length is set only in {DEFAULT_CONFIG_PATH} "
            "(vision.aruco.marker_edge_length_mm); remove it from `registration.detection`."
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
    "RegistrationRerun",
    "RegistrationRoiAlign",
    "RegistrationSweep",
    "MufasaConfig",
    "CalibrationConfig",
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