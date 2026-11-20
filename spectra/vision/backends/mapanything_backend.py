"""MapAnything back-end (``vision.backend: mapanything``).

MapAnything (Keetha et al., 2025; github.com/facebookresearch/map-anything,
Apache-2.0 code; weights facebook/map-anything CC BY-NC 4.0, or
facebook/map-anything-apache Apache 2.0) is a feed-forward metric
reconstruction model that takes images plus, optionally, intrinsics and
camera poses. Per view: world points, camera (camera-to-world, OpenCV),
intrinsics, confidence and a validity mask.

Everything it needs is read from ``vision.mapanything``:

- ``variant``: ``nc`` or ``apache`` weights; ``conditioning``: ``none``,
  ``k`` (calibrated intrinsics), ``kpose`` (+ robot poses, metric). Both go
  in the results folder name (``mapanything-nc-kpose``).
- input: zoom window as MASt3R's, centred on the calibrated principal point,
  then the authors' fixed resolution for the aspect ratio
  (``resolution_set`` 518: 4:3 -> 518x392). The images are handed to its
  preprocessing already at that size, so it changes nothing.
- points: the model's world points under its mask (non-ambiguous + edges,
  ``mask_edges``; confidence mask optional, as the authors' defaults).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import MapAnythingConfig, ReconstructionConfig
from .common import (
    assemble_feedforward,
    gt_conditioning,
    prepare_feedforward_images,
    resolve_hf_or_path,
    zoom_window,
)
from .types import BackendInputs, RawReconstruction

REPOS = {"nc": "facebook/map-anything", "apache": "facebook/map-anything-apache"}
ZOOM_DIR = "inputs_mapanything_zoom"


def run_mapanything(cfg: ReconstructionConfig, inputs: BackendInputs) -> RawReconstruction:
    mcfg: MapAnythingConfig = cfg.mapanything
    try:
        from mapanything.models import MapAnything
        from mapanything.utils.image import find_closest_aspect_ratio, preprocess_inputs
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"MapAnything is not installed ({exc}): bash Setup_env.sh mapanything "
                           "(or uv sync --extra processing --extra mapanything in .venv-mapanything).") from exc
    num_views = len(inputs.image_paths)
    if num_views < 2:
        raise RuntimeError("MapAnything back-end needs at least two images.")
    K_gt, T_gt = gt_conditioning(mcfg.conditioning, inputs, "mapanything")

    def target_shape(w, h):
        return tuple(int(x) for x in find_closest_aspect_ratio(w / h, mcfg.resolution_set))

    window, zoom_info = zoom_window(mcfg, inputs, mcfg.resolution_set, "mapanything", "MapAnything", ZOOM_DIR)
    images, geometries, originals, input_window = prepare_feedforward_images(
        inputs.image_paths, window, inputs.K_orig_gt, mcfg.center_principal_point, target_shape,
        resample=Image.Resampling.LANCZOS,
        save_dir=(Path(inputs.work_dir) / ZOOM_DIR) if (window is not None and inputs.work_dir) else None,
    )
    Hn, Wn = images.shape[1:3]
    print(f"MapAnything {mcfg.variant} ({mcfg.conditioning}): {num_views} images -> {Wn}x{Hn}")

    K_net_gt = (np.stack([g.original_to_network_intrinsics(K_gt) for g in geometries])
                if K_gt is not None else None)
    views = []
    for i in range(num_views):
        view = {"img": torch.from_numpy(images[i])}                     # (H, W, 3) uint8
        if K_net_gt is not None:
            view["intrinsics"] = torch.from_numpy(K_net_gt[i]).float()
        if T_gt is not None:
            view["camera_poses"] = torch.from_numpy(T_gt[i]).float()    # camera-to-world, metres
            view["is_metric_scale"] = torch.tensor([True])
        views.append(view)
    processed = preprocess_inputs(views, resize_mode="fixed_size", size=(int(Wn), int(Hn)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    src = resolve_hf_or_path(mcfg.checkpoint or REPOS[mcfg.variant])
    model = MapAnything.from_pretrained(src).to(device).eval()
    preds = model.infer(
        processed,
        memory_efficient_inference=True,
        use_amp=bool(mcfg.use_amp and device.type == "cuda"),
        amp_dtype="bf16",
        apply_mask=True,
        mask_edges=bool(mcfg.mask_edges),
        apply_confidence_mask=bool(mcfg.apply_confidence_mask),
        confidence_percentile=float(mcfg.conf_percentile),
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    def get(p, key):
        return p[key][0].detach().float().cpu().numpy()

    point_maps = np.stack([get(p, "pts3d") for p in preds])            # [V, H, W, 3]
    T_world_cam = np.stack([get(p, "camera_poses") for p in preds])    # [V, 4, 4]
    K_net = np.stack([get(p, "intrinsics") for p in preds])            # [V, 3, 3]
    conf = np.stack([get(p, "conf") for p in preds])                   # [V, H, W]
    mask = np.stack([get(p, "mask").reshape(Hn, Wn) for p in preds]) > 0.5
    if point_maps.shape[1:3] != (Hn, Wn):
        raise RuntimeError(f"MapAnything output {point_maps.shape[1:3]} != input {(Hn, Wn)}: its "
                           "preprocessing resized the images.")
    depth = np.stack([get(p, "depth_z").reshape(Hn, Wn) for p in preds])
    valid = mask & np.isfinite(point_maps).all(-1) & np.isfinite(conf) & (depth > 0)

    return assemble_feedforward(
        backend_name="mapanything",
        variant="-".join(x for x in (mcfg.variant, mcfg.conditioning) if x != "none"),
        point_maps=point_maps, valid=valid, confidence=np.maximum(conf, 0.0),
        T_world_cam=T_world_cam, K_network=K_net if K_net_gt is None else K_net_gt,
        geometries=geometries, originals=originals,
        voxel_size_rel=mcfg.voxel_size_rel, max_points=mcfg.max_points,
        confidence_percentile=mcfg.confidence_percentile,
        metric_frame=False,             # the markers set the scale (its own metric scale is reported by them)
        K_orig=None if K_gt is None else np.repeat(K_gt[None], num_views, 0),
        alignment_info={
            "pipeline_variant": mcfg.variant,
            "conditioning": mcfg.conditioning,
            "checkpoint": src,
            "gt_poses_used": T_gt is not None,
            "gt_intrinsics_used": K_gt is not None,
            "principal_point_centered": bool(mcfg.center_principal_point and inputs.K_orig_gt is not None),
            "input_window_px": [int(c) for c in input_window],
            "network_zoom": zoom_info,
        },
    )


#: entry point used by the registry (backends/registry.py)
run = run_mapanything

__all__ = ["REPOS", "run", "run_mapanything"]
