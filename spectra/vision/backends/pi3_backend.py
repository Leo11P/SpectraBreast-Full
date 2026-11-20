"""Pi3 / Pi3X back-end (``vision.backend: pi3``).

Pi3 (Wang et al., 2025; github.com/yyfz/Pi3, BSD-3-Clause code; weights on
Hugging Face yyfz233/Pi3X, yyfz233/Pi3) is feed-forward, multi-view and
permutation-equivariant (no reference view). Per view it predicts the camera
(camera-to-world, OpenCV) and a local point map with its confidence (a logit).
Pi3X, the authors' recommended version, can also be conditioned on the
intrinsics and on the camera poses.

Everything it needs is read from ``vision.pi3``:

- ``variant``: ``x`` (Pi3X) or ``base`` (Pi3); ``conditioning`` (Pi3X only):
  ``none``, ``k`` (calibrated intrinsics), ``kpose`` (+ robot poses). Both
  go in the results folder name (``pi3-x``, ``pi3-x-kpose``, ...).
- input: zoom window as MASt3R's, centred on the calibrated principal point,
  then the authors' resize (~``pixel_limit`` pixels, sides multiple of 14).
- points: the model's world points; kept where sigmoid(confidence) >
  ``conf_threshold`` and away from depth edges (as the authors' example).
- intrinsics: the calibrated ones when given in input, else a pinhole fit to
  the predicted rays. The scale is the model's (approximately metric for
  Pi3X): the ArUco markers set it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ..config import Pi3Config, ReconstructionConfig
from .common import (
    assemble_feedforward,
    depth_edges,
    gt_conditioning,
    intrinsics_from_point_maps,
    prepare_feedforward_images,
    resolve_hf_or_path,
    zoom_window,
)
from .types import BackendInputs, RawReconstruction

REPOS = {"x": "yyfz233/Pi3X", "base": "yyfz233/Pi3"}
PATCH = 14
ZOOM_DIR = "inputs_pi3_zoom"


def _target_shape(pixel_limit: int):
    """The authors' resize (pi3.utils.basic.load_images_as_tensor): about
    pixel_limit pixels, aspect kept, sides multiple of 14."""
    def shape(w: int, h: int):
        scale = math.sqrt(pixel_limit / (w * h))
        wt, ht = w * scale, h * scale
        k, m = round(wt / PATCH), round(ht / PATCH)
        while (k * PATCH) * (m * PATCH) > pixel_limit:
            if k / m > wt / ht:
                k -= 1
            else:
                m -= 1
        return max(1, k) * PATCH, max(1, m) * PATCH
    return shape


def _load_model(pcfg: Pi3Config, conditioned: bool, device):
    try:
        if pcfg.variant == "x":
            from pi3.models.pi3x import Pi3X as Model
        else:
            from pi3.models.pi3 import Pi3 as Model
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"Pi3 is not installed ({exc}): bash Setup_env.sh pi3 "
                           "(or uv sync --extra processing --extra pi3 in .venv-pi3).") from exc
    src = resolve_hf_or_path(pcfg.checkpoint or REPOS[pcfg.variant])
    if src.endswith(".safetensors") and Path(src).is_file():
        from safetensors.torch import load_file

        model = Model(use_multimodal=conditioned) if pcfg.variant == "x" else Model()
        model.load_state_dict(load_file(src), strict=False)
    else:
        model = Model.from_pretrained(src)
    if pcfg.variant == "x" and not conditioned:
        model.disable_multimodal()        # authors: frees the conditioning branches
    return model.to(device).eval(), src


def run_pi3(cfg: ReconstructionConfig, inputs: BackendInputs) -> RawReconstruction:
    pcfg: Pi3Config = cfg.pi3
    if pcfg.variant == "base" and pcfg.conditioning != "none":
        raise ValueError("vision.pi3.conditioning needs variant 'x' (Pi3X): Pi3 takes images only.")
    num_views = len(inputs.image_paths)
    if num_views < 2:
        raise RuntimeError("Pi3 back-end needs at least two images.")
    K_gt, T_gt = gt_conditioning(pcfg.conditioning, inputs, "pi3")

    target_shape = _target_shape(pcfg.pixel_limit)
    window, zoom_info = zoom_window(pcfg, inputs, int(math.sqrt(pcfg.pixel_limit)), "pi3", "Pi3", ZOOM_DIR)
    images, geometries, originals, input_window = prepare_feedforward_images(
        inputs.image_paths, window, inputs.K_orig_gt, pcfg.center_principal_point, target_shape,
        resample=Image.Resampling.LANCZOS,
        save_dir=(Path(inputs.work_dir) / ZOOM_DIR) if (window is not None and inputs.work_dir) else None,
    )
    Hn, Wn = images.shape[1:3]
    print(f"Pi3 {pcfg.variant} ({pcfg.conditioning}): {num_views} images -> {Wn}x{Hn}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src = _load_model(pcfg, pcfg.conditioning != "none", device)
    imgs = torch.from_numpy(images).float().div(255.0).permute(0, 3, 1, 2)[None].to(device)
    kwargs = {}
    K_net_gt = None
    if K_gt is not None:
        K_net_gt = np.stack([g.original_to_network_intrinsics(K_gt) for g in geometries])
        K_in = K_net_gt.copy()
        K_in[:, 0, 2] += 0.5          # Pi3X: pixel (u, v) at (u + 0.5, v + 0.5)
        K_in[:, 1, 2] += 0.5
        kwargs["intrinsics"] = torch.from_numpy(K_in).float()[None].to(device)
    if T_gt is not None:
        kwargs["poses"] = torch.from_numpy(T_gt).float()[None].to(device)
    amp = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) \
        if device.type == "cuda" else None
    with torch.no_grad():
        with torch.amp.autocast("cuda", dtype=amp, enabled=amp is not None):
            res = model(imgs, **kwargs) if pcfg.variant == "x" else model(imgs)
    point_maps = res["points"][0].float().cpu().numpy()                 # [V, H, W, 3] world
    local = res["local_points"][0].float().cpu().numpy()                # [V, H, W, 3] camera
    conf = torch.sigmoid(res["conf"][0, ..., 0].float()).cpu().numpy()  # [V, H, W] in [0, 1]
    T_world_cam = res["camera_poses"][0].float().cpu().numpy()          # [V, 4, 4] camera-to-world
    del model, res, imgs
    if device.type == "cuda":
        torch.cuda.empty_cache()

    z = local[..., 2]
    valid = np.isfinite(point_maps).all(-1) & np.isfinite(conf) & (z > 0)
    valid &= conf > float(pcfg.conf_threshold)
    if pcfg.filter_depth_edges:
        valid &= ~depth_edges(np.where(z > 0, z, 1.0), float(pcfg.depth_edge_rtol))
    K_net = K_net_gt if K_net_gt is not None else intrinsics_from_point_maps(point_maps, T_world_cam, valid)

    return assemble_feedforward(
        backend_name="pi3",
        variant="-".join(x for x in (pcfg.variant, pcfg.conditioning) if x != "none"),
        point_maps=point_maps, valid=valid, confidence=conf,
        T_world_cam=T_world_cam, K_network=K_net,
        geometries=geometries, originals=originals,
        voxel_size_rel=pcfg.voxel_size_rel, max_points=pcfg.max_points,
        confidence_percentile=pcfg.confidence_percentile,
        metric_frame=False,
        K_orig=None if K_gt is None else np.repeat(K_gt[None], num_views, 0),
        alignment_info={
            "pipeline_variant": pcfg.variant,
            "conditioning": pcfg.conditioning,
            "checkpoint": src,
            "gt_poses_used": T_gt is not None,
            "gt_intrinsics_used": K_gt is not None,
            "principal_point_centered": bool(pcfg.center_principal_point and inputs.K_orig_gt is not None),
            "input_window_px": [int(c) for c in input_window],
            "network_zoom": zoom_info,
        },
    )


#: entry point used by the registry (backends/registry.py)
run = run_pi3

__all__ = ["REPOS", "run", "run_pi3"]
