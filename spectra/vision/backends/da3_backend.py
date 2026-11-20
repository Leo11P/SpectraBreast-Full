"""Depth Anything 3 back-end (``vision.backend: da3``).

DA3 (Lin et al., 2025; github.com/ByteDance-Seed/Depth-Anything-3, Apache-2.0
code; giant / nested weights CC BY-NC 4.0 on Hugging Face depth-anything/*)
is feed-forward and multi-view: per view a depth map, its confidence and the
camera (world-to-camera, OpenCV; intrinsics). It can be conditioned on the
cameras (intrinsics + poses together): then, as in the authors' API
(align_to_input_ext_scale), the output is in the input poses' frame and scale.

Everything it needs is read from ``vision.da3``:

- ``variant``: the checkpoint (``nested-giant-large-1.1`` = any-view giant +
  metric depth, the authors' strongest; ``giant-1.1``; ``large-1.1``);
  ``conditioning``: ``none`` or ``kpose`` (calibrated K + robot poses). Both go
  in the results folder name (``da3-nested-giant-large-1.1-kpose``).
- input: zoom window as MASt3R's, centred on the calibrated principal point,
  then the authors' resize (long side ``process_res``, sides multiple of 14).
  The images are handed to the API already at that size, so its own
  preprocessing changes nothing and the geometry stays ours.
- points: depth unprojected with the predicted (or given) cameras; kept above
  the ``conf_percentile`` of the confidence and away from depth edges.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import Da3Config, ReconstructionConfig
from .common import (
    assemble_feedforward,
    depth_edges,
    gt_conditioning,
    invert_rigid,
    prepare_feedforward_images,
    resolve_hf_or_path,
    unproject_depth,
    zoom_window,
)
from .types import BackendInputs, RawReconstruction

REPOS = {
    "nested-giant-large-1.1": "depth-anything/DA3NESTED-GIANT-LARGE-1.1",
    "giant-1.1": "depth-anything/DA3-GIANT-1.1",
    "large-1.1": "depth-anything/DA3-LARGE-1.1",
}
PATCH = 14
ZOOM_DIR = "inputs_da3_zoom"


def _target_shape(process_res: int):
    """The authors' upper_bound_resize: long side process_res, then each side
    to the nearest multiple of 14 (input_processor.py)."""
    def nearest(x: int) -> int:
        down = (x // PATCH) * PATCH
        up = down + PATCH
        return max(PATCH, up if abs(up - x) <= abs(x - down) else down)

    def shape(w: int, h: int):
        if max(w, h) != process_res:
            s = process_res / float(max(w, h))
            w, h = max(1, int(round(w * s))), max(1, int(round(h * s)))
        return nearest(w), nearest(h)
    return shape


def run_da3(cfg: ReconstructionConfig, inputs: BackendInputs) -> RawReconstruction:
    dcfg: Da3Config = cfg.da3
    try:
        from depth_anything_3.api import DepthAnything3
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Depth Anything 3 is not installed ({exc}): bash Setup_env.sh da3 "
                           "(or uv sync --extra processing --extra da3 in .venv-da3).") from exc
    num_views = len(inputs.image_paths)
    if num_views < 2:
        raise RuntimeError("DA3 back-end needs at least two images.")
    K_gt, T_gt = gt_conditioning(dcfg.conditioning, inputs, "da3")

    window, zoom_info = zoom_window(dcfg, inputs, dcfg.process_res, "da3", "DA3", ZOOM_DIR)
    images, geometries, originals, input_window = prepare_feedforward_images(
        inputs.image_paths, window, inputs.K_orig_gt, dcfg.center_principal_point,
        _target_shape(dcfg.process_res),
        save_dir=(Path(inputs.work_dir) / ZOOM_DIR) if (window is not None and inputs.work_dir) else None,
    )
    Hn, Wn = images.shape[1:3]
    print(f"DA3 {dcfg.variant} ({dcfg.conditioning}): {num_views} images -> {Wn}x{Hn}")

    extrinsics = intrinsics = None
    if T_gt is not None:
        extrinsics = invert_rigid(T_gt).astype(np.float32)             # world-to-camera
        intrinsics = np.stack([g.original_to_network_intrinsics(K_gt) for g in geometries]).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = resolve_hf_or_path(dcfg.checkpoint or REPOS[dcfg.variant])
    model = DepthAnything3.from_pretrained(src).to(device).eval()
    pred = model.inference(
        image=list(images),
        extrinsics=extrinsics,
        intrinsics=intrinsics,
        align_to_input_ext_scale=True,
        use_ray_pose=bool(dcfg.use_ray_pose),
        ref_view_strategy=dcfg.ref_view_strategy,
        process_res=int(max(Wn, Hn)),                 # already at size: no further resize
        process_res_method="upper_bound_resize",
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    depth = np.asarray(pred.depth, dtype=np.float32)                    # [V, H, W] z-depth
    if depth.shape != (num_views, Hn, Wn):
        raise RuntimeError(f"DA3 output {depth.shape} != input {(num_views, Hn, Wn)}: its "
                           "preprocessing resized the images (process_res / patch rule changed?).")
    conf = np.asarray(pred.conf, dtype=np.float32)
    w2c = np.asarray(pred.extrinsics, dtype=np.float32)
    T_cw = np.tile(np.eye(4, dtype=np.float32), (num_views, 1, 1))
    T_cw[:, :3, :] = w2c[:, :3, :]
    T_world_cam = invert_rigid(T_cw)
    K_net = np.asarray(pred.intrinsics, dtype=np.float32)
    point_maps = unproject_depth(depth, K_net, T_world_cam)

    valid = np.isfinite(point_maps).all(-1) & np.isfinite(conf) & (depth > 0)
    if dcfg.filter_depth_edges:
        valid &= ~depth_edges(np.where(depth > 0, depth, 1.0), float(dcfg.depth_edge_rtol))
    conf_thr = float(np.percentile(conf[valid], dcfg.conf_percentile)) if valid.any() else 0.0
    valid &= conf >= conf_thr

    return assemble_feedforward(
        backend_name="da3",
        variant="-".join(x for x in (dcfg.variant, dcfg.conditioning) if x != "none"),
        point_maps=point_maps, valid=valid, confidence=np.maximum(conf, 0.0),
        T_world_cam=T_world_cam, K_network=K_net,
        geometries=geometries, originals=originals,
        voxel_size_rel=dcfg.voxel_size_rel, max_points=dcfg.max_points,
        confidence_percentile=dcfg.confidence_percentile,
        metric_frame=False,             # the markers set the scale (DA3's own is reported by them)
        K_orig=None if K_gt is None else np.repeat(K_gt[None], num_views, 0),
        alignment_info={
            "pipeline_variant": dcfg.variant,
            "conditioning": dcfg.conditioning,
            "checkpoint": src,
            "is_metric": int(getattr(pred, "is_metric", 0) or 0),
            "gt_poses_used": T_gt is not None,
            "gt_intrinsics_used": K_gt is not None,
            "principal_point_centered": bool(dcfg.center_principal_point and inputs.K_orig_gt is not None),
            "input_window_px": [int(c) for c in input_window],
            "conf_threshold": conf_thr,
            "network_zoom": zoom_info,
        },
    )


#: entry point used by the registry (backends/registry.py)
run = run_da3

__all__ = ["REPOS", "run", "run_da3"]
