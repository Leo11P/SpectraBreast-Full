"""MASt3R-SfM reconstruction with ArUco-stabilized metric alignment.

- `run_reconstruction(cfg)` — images → dense cloud, ArUco 2D/3D, surface, exports.
- `ReconstructionConfig` / `load_config` — YAML configuration.
- `ArucoDetector`, `detect_folder` — standalone 2D detection.
"""

from __future__ import annotations

from .aruco import (
    ARUCO_DICTIONARIES,
    ArucoDetector,
    MarkerDetection,
    annotate_image,
    color_for_id,
    color_for_id_rgb,
    detect_folder,
    detect_image,
    read_detections_json,
)
from .calibration import calibrate_intrinsics
from .config import (
    ArucoConfig,
    InputConfig,
    Mast3rConfig,
    OutputConfig,
    ReconstructionConfig,
    SurfaceConfig,
    load_config,
    save_config,
)


def __getattr__(name: str):
    """``ReconstructionResult`` / ``run_reconstruction`` need torch: imported on
    first use, so ``spectra.vision.aruco`` & co. work without it."""
    if name in ("ReconstructionResult", "run_reconstruction"):
        from . import pipeline

        value = getattr(pipeline, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_viewer(*args, **kwargs):
    """Lazy re-export of :func:`spectra.vision.viewer.run_viewer`."""
    from .viewer import run_viewer as _run_viewer
    return _run_viewer(*args, **kwargs)


__all__ = [
    "ARUCO_DICTIONARIES",
    "ArucoConfig",
    "ArucoDetector",
    "InputConfig",
    "MarkerDetection",
    "Mast3rConfig",
    "OutputConfig",
    "ReconstructionConfig",
    "ReconstructionResult",
    "SurfaceConfig",
    "annotate_image",
    "calibrate_intrinsics",
    "color_for_id",
    "color_for_id_rgb",
    "detect_folder",
    "detect_image",
    "load_config",
    "read_detections_json",
    "run_reconstruction",
    "run_viewer",
    "save_config",
]
