"""Dataclasses shared by every reconstruction back-end: the contract between
the pipeline and the models.

The orchestrator in ``spectra/vision/pipeline.py`` only sees these two shapes,
so it applies the ArUco Sim3 / BA, the surface reconstruction and writes the
outputs without knowing which model ran.

Conventions (every back-end):

- Images: ``image_paths`` are already EXIF-upright and, with
  ``vision.input.undistort``, ideal pinhole images with K = ``K_orig_gt``
  (``distortion_coeffs`` is then zero). A back-end never undistorts.
- Poses: 4x4 **camera-to-world**, OpenCV camera axes (x right, y down,
  z forward), in and out. A model working in world-to-camera inverts inside
  its back-end.
- Intrinsics: ``K_orig_gt`` and ``K_per_view_orig`` are in the pixels of the
  ORIGINAL (full-resolution) images: the ArUco triangulation uses them with the
  2D detections. Resizing / cropping for the network is the back-end's own
  business (``common.ImageGeometry``).
- Everything model-specific (input resolution, zoom, confidence thresholds,
  fusion voxel, ...) is read by the back-end from its own config section
  ``vision.<backend>``; the pipeline never reads that section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List

import numpy as np


@dataclass
class BackendInputs:
    """Resolved inputs passed to a back-end's ``run(cfg, inputs)``."""

    image_paths: List[Path]
    pose_dir: Path | None = None
    camera_params_dir: Path | None = None
    # Optional pre-loaded tensors (inputs.build_backend_inputs fills these in):
    T_world_cam_gt: np.ndarray | None = None     # [V, 4, 4], camera-to-world (robot poses)
    K_orig_gt: np.ndarray | None = None          # [3, 3], shared intrinsics, original pixels
    cam2ee: np.ndarray | None = None             # [4, 4], camera-in-EE
    distortion_coeffs: np.ndarray | None = None  # [5], zero once the images are undistorted
    # ArUco detections on the full-resolution images, per view (list of
    # aruco.MarkerDetection): a back-end may use them (e.g. MASt3R's zoom).
    aruco_detections: List[list] | None = None
    # The run folder: where a back-end may save what it fed the network.
    work_dir: Path | None = None


@dataclass
class RawReconstruction:
    """Output of a back-end, expressed in one consistent "world" frame.

    - Point maps, clouds and ``T_world_cam`` share that frame (any origin,
      orientation and scale; metric only if ``metric_frame``).
    - Per-view maps are organised on the back-end's pixel grid ("network"
      grid, ``[V, Hn, Wn, ...]``), padded to a common size with
      ``valid_masks`` False in the padding.
    - Confidence: >= 0, higher = better; its scale is the model's own.
    - ``extra``: back-end specific. A per-view entry (array or list) must have
      length V: it is sliced with the views when some are rejected.
    """

    fused_points: np.ndarray              # [N, 3] float32
    fused_colors: np.ndarray              # [N, 3] uint8
    fused_confidence: np.ndarray          # [N]   float32
    point_map_world: np.ndarray           # [V, Hn, Wn, 3] float32
    valid_masks: np.ndarray               # [V, Hn, Wn] bool
    T_world_cam: np.ndarray               # [V, 4, 4] float32, camera-to-world (OpenCV)
    K_per_view_orig: np.ndarray           # [V, 3, 3] float32 (original pixels)
    K_per_view_network: np.ndarray        # [V, 3, 3] float32 (network pixels)
    network_image_sizes: np.ndarray       # [V, 2] int32, (width, height)
    original_image_sizes: np.ndarray      # [V, 2] int32, (width, height)
    images_network_uint8: np.ndarray      # [V, Hn, Wn, 3] uint8
    confidence_maps_network: np.ndarray   # [V, Hn, Wn] float32
    frame_description: str                # e.g. 'sfm' | 'mst' | 'gt-aligned'
    backend_name: str                     # registry name, e.g. 'mast3r'
    #: variant of the method that ran (e.g. 'sfm' / 'dense'), None if one only
    variant: str | None = None
    #: True if the frame is already metric (e.g. reconstructed on the robot
    #: poses with the calibrated K): the ArUco scale is then only checked, not
    #: applied. False: the scale comes from the markers.
    metric_frame: bool = False
    #: voxel (in the back-end frame) the back-end fused with: the pipeline
    #: re-fuses with the same one after the ArUco BA. 0 = no voxel fusion.
    voxel_size: float = 0.0
    #: percentile for the confidence colour map (Rerun): depends on the scale
    #: of the model's confidence.
    confidence_percentile: float = 99.0
    #: intermediate folders the pipeline deletes at the end of the run
    #: (vision.output.clean_sfm_cache)
    cache_dirs: List[Path] = field(default_factory=list)
    alignment_info: dict | None = None    # back-end specific info, saved in the metadata
    extra: dict[str, Any] = field(default_factory=dict)


__all__ = ["BackendInputs", "RawReconstruction"]
