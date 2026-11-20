"""VGGT-Omega back-end (``vision.backend: vggt_omega``).

VGGT-Omega (Wang et al., CVPR 2026; github.com/facebookresearch/vggt-omega,
FAIR Noncommercial Research License) is feed-forward and multi-view: ONE
forward pass over all the images gives, per view, the camera (pose + focal)
and a depth map with its confidence. No pairs, no global optimisation, no
pose or intrinsics in input: the robot poses and the calibrated K are not
used, the frame is the model's (first camera, arbitrary scale) and the ArUco
markers set the metric scale.

Everything it needs is read from ``vision.vggt_omega``:

- ``variant`` picks the released checkpoint and its input resolution:
  ``512`` = vggt_omega_1b_512.pt (the authors' choice for in-the-wild use),
  ``416`` = vggt_omega_1b_416_reproduce.pt (their benchmark reference).
- input: the zoom window (as MASt3R's, on the ArUco detections), then the
  model's own resize (``resize_mode`` balanced: ~image_resolution^2 pixels,
  sides multiple of 16). Intrinsics and outputs are mapped back to the
  original pixels (``common.ImageGeometry``).
- principal point: the model's cameras have it at the image centre. With the
  calibrated K (``center_principal_point``) the window is re-centred on the
  calibrated principal point, so that assumption holds for our camera
  (otherwise ~10-20 px of offset on the D456 colour camera).
- points: depth unprojected with the predicted cameras; kept above the
  ``conf_percentile`` of the confidence and away from depth edges (as the
  authors' demo).
- fusion voxel: ``voxel_size_rel`` x the median depth, because the model's
  scale is arbitrary (0.006 ~ 1.2 mm at 200 mm).

Convention of the model: extrinsics camera-from-world, OpenCV axes; the
principal point at the image centre. The contract wants camera-to-world.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from ..config import ReconstructionConfig, VggtOmegaConfig
from .common import (
    assemble_feedforward,
    depth_edges,
    invert_rigid,
    prepare_feedforward_images,
    unproject_depth,
    zoom_window,
)
from .types import BackendInputs, RawReconstruction

#: variant -> (Hugging Face repo, checkpoint file, input resolution)
CHECKPOINTS = {
    "512": ("facebook/VGGT-Omega", "vggt_omega_1b_512.pt", 512),
    "416": ("facebook/VGGT-Omega", "vggt_omega_1b_416_reproduce.pt", 416),
}
PATCH_SIZE = 16
ZOOM_DIR = "inputs_vggt_omega_zoom"


def _import_vggt_omega():
    try:
        from vggt_omega.models import VGGTOmega
        from vggt_omega.utils.load_fn import _balanced_target_shape, _max_size_target_shape
        from vggt_omega.utils.pose_enc import encoding_to_camera
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"VGGT-Omega is not installed ({exc}). Install it in this back-end's environment: "
            "bash Setup_env.sh vggt_omega (or: UV_PROJECT_ENVIRONMENT=.venv-vggt_omega "
            "uv sync --extra processing --extra vggt-omega)."
        ) from exc
    return VGGTOmega, _balanced_target_shape, _max_size_target_shape, encoding_to_camera


def resolve_checkpoint(vcfg: VggtOmegaConfig) -> Path:
    """Local path of the checkpoint: ``checkpoint`` if it is a file, else
    ``<hf repo>/<file>`` (default: the variant's) from the Hugging Face cache
    (HF_HOME), downloaded the first time. The repo is gated: request access on
    huggingface.co/facebook/VGGT-Omega and log in once (`hf auth login`)."""
    repo, filename, _ = CHECKPOINTS[vcfg.variant]
    if vcfg.checkpoint:
        p = Path(vcfg.checkpoint).expanduser()
        if p.is_file():
            return p
        parts = str(vcfg.checkpoint).strip("/").split("/")
        if len(parts) < 3:
            raise FileNotFoundError(
                f"vision.vggt_omega.checkpoint {vcfg.checkpoint!r}: not a file, and not "
                "'<hf user>/<hf repo>/<file>'.")
        repo, filename = "/".join(parts[:2]), "/".join(parts[2:])
    from huggingface_hub import hf_hub_download

    try:
        return Path(hf_hub_download(repo_id=repo, filename=filename))
    except Exception as exc:
        raise RuntimeError(
            f"VGGT-Omega checkpoint {repo}/{filename} not available ({type(exc).__name__}: {exc}). "
            "The Hugging Face repo is gated: request access on "
            "https://huggingface.co/facebook/VGGT-Omega, log in once with `hf auth login` "
            "(same HF_HOME as the jobs), or set vision.vggt_omega.checkpoint to a local .pt."
        ) from exc


def run_vggt_omega(cfg: ReconstructionConfig, inputs: BackendInputs) -> RawReconstruction:
    vcfg: VggtOmegaConfig = cfg.vggt_omega
    VGGTOmega, balanced_shape, max_size_shape, encoding_to_camera = _import_vggt_omega()
    _, _, default_res = CHECKPOINTS[vcfg.variant]
    image_resolution = int(vcfg.image_resolution or default_res)
    if image_resolution % PATCH_SIZE:
        raise ValueError(f"vision.vggt_omega.image_resolution must be a multiple of {PATCH_SIZE}")
    num_views = len(inputs.image_paths)
    if num_views < 2:
        raise RuntimeError("VGGT-Omega back-end needs at least two images.")

    def target_shape(w, h):
        # the model's loader centre-crops aspect ratios outside [0.5, 2]: same here
        ar = min(max(h / max(w, 1), 0.5), 2.0)
        fn = balanced_shape if vcfg.resize_mode == "balanced" else max_size_shape
        th, tw = fn(ar, image_resolution, PATCH_SIZE)
        return tw, th

    window, zoom_info = zoom_window(vcfg, inputs, image_resolution, "vggt_omega", "VGGT-Omega", ZOOM_DIR)
    images, geometries, originals, input_window = prepare_feedforward_images(
        inputs.image_paths, window, inputs.K_orig_gt, vcfg.center_principal_point, target_shape,
        save_dir=(Path(inputs.work_dir) / ZOOM_DIR) if (window is not None and inputs.work_dir) else None,
    )
    Hn, Wn = images.shape[1:3]
    print(f"VGGT-Omega {vcfg.variant}: {num_views} images -> {Wn}x{Hn} ({vcfg.resize_mode})")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = resolve_checkpoint(vcfg)
    model = VGGTOmega(autocast=bool(vcfg.autocast and device.type == "cuda")).eval()
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    del state
    model = model.to(device)
    x = torch.from_numpy(images).float().div(255.0).permute(0, 3, 1, 2).to(device)
    with torch.inference_mode():
        pred = model(x)
        w2c, K_net = encoding_to_camera(pred["pose_enc"], (Hn, Wn))
    w2c = w2c[0].float().cpu().numpy()                       # [V, 3, 4] camera-from-world
    K_net = K_net[0].float().cpu().numpy()                   # [V, 3, 3] network pixels
    depth = pred["depth"][0, ..., 0].float().cpu().numpy()   # [V, Hn, Wn]
    conf = pred["depth_conf"][0].float().cpu().numpy()       # [V, Hn, Wn], >= 1
    del model, pred, x
    if device.type == "cuda":
        torch.cuda.empty_cache()

    T_cw = np.tile(np.eye(4, dtype=np.float32), (num_views, 1, 1))
    T_cw[:, :3, :] = w2c
    T_world_cam = invert_rigid(T_cw)                          # camera-to-world
    point_maps = unproject_depth(depth, K_net, T_world_cam)

    valid = np.isfinite(point_maps).all(-1) & np.isfinite(conf) & (depth > 0)
    if vcfg.filter_depth_edges:
        valid &= ~depth_edges(depth, float(vcfg.depth_edge_rtol))
    conf_thr = float(np.percentile(conf[valid], vcfg.conf_percentile)) if valid.any() else 0.0
    valid &= conf >= conf_thr

    return assemble_feedforward(
        backend_name="vggt_omega",
        variant=vcfg.variant,
        point_maps=point_maps, valid=valid, confidence=conf,
        T_world_cam=T_world_cam, K_network=K_net,
        geometries=geometries, originals=originals,
        voxel_size_rel=vcfg.voxel_size_rel, max_points=vcfg.max_points,
        confidence_percentile=vcfg.confidence_percentile,
        metric_frame=False,             # its own frame and scale: the markers set the scale
        alignment_info={
            "pipeline_variant": vcfg.variant,
            "checkpoint": str(Path(ckpt).name),
            "image_resolution": image_resolution,
            "resize_mode": vcfg.resize_mode,
            "gt_poses_used": False,
            "gt_intrinsics_used": False,
            "principal_point_centered": bool(vcfg.center_principal_point and inputs.K_orig_gt is not None),
            "input_window_px": [int(c) for c in input_window],
            "conf_threshold": conf_thr,
            "network_zoom": zoom_info,
        },
    )


#: entry point used by the registry (backends/registry.py)
run = run_vggt_omega

__all__ = ["CHECKPOINTS", "resolve_checkpoint", "run", "run_vggt_omega"]
