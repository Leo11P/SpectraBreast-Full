"""Spectra: unified pipeline (MASt3R-SfM + ArUco vision + HSI→mesh registration).

Public API
==========

End-to-end::

    from spectra import run_full, load_unified_config

    cfg = load_unified_config("configs/default.yaml")
    result = run_full(cfg, sample_name="SB019")

Single stages::

    from spectra import run_vision_stage, run_registration_stage

Vision-only convenience (legacy compatibility, mirrors the original
``spectra`` API exactly)::

    from spectra import run_reconstruction, load_config
"""

from __future__ import annotations

# -------- Unified config + orchestrator --------------------------------------
from .config import (
    ArucoConfig,
    InputConfig,
    Mast3rConfig,
    OutputConfig,
    ReconstructionConfig,
    RegistrationConfig,
    RerunConfig,
    SurfaceConfig,
    UnifiedConfig,
    VisionConfig,
    load_config,
    load_unified_config,
    save_config,
    save_config_json,
    save_unified_config,
    save_unified_config_json,
)
from .orchestrator import (
    resolve_unified_paths,
    run_full,
    run_registration_stage,
    run_vision_stage,
    vision_outputs_present,
)

# -------- Vision (legacy public API, unchanged) ------------------------------
from .vision import (
    ARUCO_DICTIONARIES,
    ArucoDetector,
    MarkerDetection,
    annotate_image,
    calibrate_intrinsics,
    color_for_id,
    color_for_id_rgb,
    detect_folder,
    detect_image,
    read_detections_json,
)
from .vision.pipeline import ReconstructionResult, run_reconstruction

# -------- Registration -------------------------------------------------------
from .registration import (
    extract_suspicious_centroids,
    load_mesh,
    render_orthographic_topview_gpu,
    run_full_pipeline,
    run_full_pipeline_roi,
    run_registration,
)


def run_viewer(*args, **kwargs):
    """Lazy re-export of :func:`spectra.vision.viewer.run_viewer`."""
    from .vision.viewer import run_viewer as _run_viewer
    return _run_viewer(*args, **kwargs)


__all__ = [
    # Unified
    "UnifiedConfig",
    "load_unified_config",
    "save_unified_config",
    "save_unified_config_json",
    "resolve_unified_paths",
    "run_full",
    "run_vision_stage",
    "run_registration_stage",
    "vision_outputs_present",
    # Vision
    "ARUCO_DICTIONARIES",
    "ArucoConfig",
    "ArucoDetector",
    "InputConfig",
    "MarkerDetection",
    "Mast3rConfig",
    "OutputConfig",
    "ReconstructionConfig",
    "ReconstructionResult",
    "RerunConfig",
    "SurfaceConfig",
    "VisionConfig",
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
    "save_config_json",
    # Registration
    "RegistrationConfig",
    "extract_suspicious_centroids",
    "load_mesh",
    "render_orthographic_topview_gpu",
    "run_full_pipeline",
    "run_full_pipeline_roi",
    "run_registration",
]

__version__ = "0.2.0"
