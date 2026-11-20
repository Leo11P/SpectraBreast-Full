"""Helpers shared by the back-ends (and by the pipeline's re-fusion after the
ArUco BA). Nothing here is tied to one model: each back-end decides whether and
with which parameters (from its own config section) to use them."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Original image <-> network input
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ImageGeometry:
    """How one original image maps onto the network input: resize by
    (scale_x, scale_y), then a crop starting at (crop_left, crop_top) in the
    resized image. Each back-end builds it with its own resize rule."""

    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    network_height: int
    network_width: int
    scale_x: float
    scale_y: float
    crop_left: float
    crop_top: float

    def original_to_network_intrinsics(self, K_orig: np.ndarray) -> np.ndarray:
        A = np.array(
            [
                [self.scale_x, 0.0, -self.crop_left],
                [0.0, self.scale_y, -self.crop_top],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return A @ K_orig.astype(np.float32)

    def network_to_original_intrinsics(self, K_net: np.ndarray) -> np.ndarray:
        """Invert `original_to_network_intrinsics`: pulls K back into original pixel coords."""
        A_inv = np.array(
            [
                [1.0 / self.scale_x, 0.0, self.crop_left / self.scale_x],
                [0.0, 1.0 / self.scale_y, self.crop_top / self.scale_y],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        return A_inv @ K_net.astype(np.float32)

    def network_to_original(self, uv_net: np.ndarray) -> np.ndarray:
        uv_orig = uv_net.astype(np.float32).copy()
        uv_orig[:, 0] = (uv_orig[:, 0] + self.crop_left) / self.scale_x
        uv_orig[:, 1] = (uv_orig[:, 1] + self.crop_top) / self.scale_y
        return uv_orig

    def network_grid_to_original(self) -> tuple[np.ndarray, np.ndarray]:
        uu, vv = np.meshgrid(
            np.arange(self.network_width, dtype=np.float32),
            np.arange(self.network_height, dtype=np.float32),
        )
        uu = (uu + self.crop_left) / self.scale_x
        vv = (vv + self.crop_top) / self.scale_y
        return uu, vv

    def warp_network_map_to_original(self, map_net: np.ndarray) -> np.ndarray:
        canvas = np.zeros((self.resized_height, self.resized_width), dtype=np.float32)
        left = int(round(self.crop_left))
        top = int(round(self.crop_top))
        canvas[top : top + self.network_height, left : left + self.network_width] = map_net.astype(np.float32)
        return cv2.resize(
            canvas,
            (self.original_width, self.original_height),
            interpolation=cv2.INTER_LINEAR,
        )


def colors_on_network_grid(geom: ImageGeometry, rgb_orig: np.ndarray) -> np.ndarray:
    """[Hn, Wn, 3] uint8: the original image's colour at each network pixel
    (nearest pixel of the full-resolution image)."""
    uu_orig, vv_orig = geom.network_grid_to_original()
    u_idx = np.clip(np.round(uu_orig).astype(np.int32), 0, geom.original_width - 1)
    v_idx = np.clip(np.round(vv_orig).astype(np.int32), 0, geom.original_height - 1)
    return rgb_orig[v_idx, u_idx].astype(np.uint8)


# --------------------------------------------------------------------------- #
# Poses
# --------------------------------------------------------------------------- #
def invert_rigid(T: np.ndarray) -> np.ndarray:
    """[..., 4, 4] rigid inverse: world-to-camera <-> camera-to-world."""
    T = np.asarray(T, dtype=np.float64)
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    out = np.zeros_like(T)
    Rt = np.swapaxes(R, -1, -2)
    out[..., :3, :3] = Rt
    out[..., :3, 3] = -np.einsum("...ij,...j->...i", Rt, t)
    out[..., 3, 3] = 1.0
    return out.astype(np.float32)


# --------------------------------------------------------------------------- #
# Fusion
# --------------------------------------------------------------------------- #
def weighted_voxel_downsample(
    points: np.ndarray,
    colors: np.ndarray,
    confidence: np.ndarray,
    voxel_size: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One point per voxel: confidence-weighted mean position and colour, mean
    confidence."""
    if len(points) == 0 or voxel_size <= 0:
        return points, colors, confidence

    keys = np.floor(points / float(voxel_size)).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)  # numpy>=2.0 may return shape [N,1] for axis=0
    num_vox = int(inv.max()) + 1
    weights = np.clip(confidence.astype(np.float64), 1e-6, None)
    weight_sum = np.bincount(inv, weights=weights, minlength=num_vox)
    counts = np.bincount(inv, minlength=num_vox)

    out_pts = np.zeros((num_vox, 3), dtype=np.float64)
    out_cols = np.zeros((num_vox, 3), dtype=np.float64)
    for d in range(3):
        out_pts[:, d] = np.bincount(
            inv, weights=weights * points[:, d].astype(np.float64), minlength=num_vox
        ) / np.maximum(weight_sum, 1e-12)
        out_cols[:, d] = np.bincount(
            inv, weights=weights * colors[:, d].astype(np.float64), minlength=num_vox
        ) / np.maximum(weight_sum, 1e-12)
    out_conf = np.bincount(
        inv, weights=confidence.astype(np.float64), minlength=num_vox
    ) / np.maximum(counts, 1)

    return (
        out_pts.astype(np.float32),
        np.clip(out_cols, 0.0, 255.0).astype(np.uint8),
        out_conf.astype(np.float32),
    )


def fuse_points(
    points: Sequence[np.ndarray],
    colors: Sequence[np.ndarray],
    confidence: Sequence[np.ndarray],
    voxel_size: float,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate the valid points of every view, voxel-fuse them
    (``voxel_size`` > 0) and keep at most ``max_points``, the most confident."""
    fused_points = np.concatenate(points, axis=0)
    fused_colors = np.concatenate(colors, axis=0)
    fused_confidence = np.concatenate(confidence, axis=0)
    if voxel_size > 0.0:
        fused_points, fused_colors, fused_confidence = weighted_voxel_downsample(
            fused_points, fused_colors, fused_confidence, voxel_size
        )
    if len(fused_points) > max_points:
        keep = np.argsort(fused_confidence)[-max_points:][::-1]
        fused_points = fused_points[keep]
        fused_colors = fused_colors[keep]
        fused_confidence = fused_confidence[keep]
    return fused_points, fused_colors, fused_confidence


def pad_view_maps_for_stacking(
    point_map_world: List[np.ndarray],
    valid_masks: List[np.ndarray],
    images_network_uint8: List[np.ndarray],
    confidence_maps_network: List[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pad per-view maps to a common [V, Hmax, Wmax, ...] shape.

    Mixed portrait/landscape inputs can produce heterogeneous network shapes.
    We pad each view to the global max shape and keep padded regions invalid.
    """
    max_h = max(int(mask.shape[0]) for mask in valid_masks)
    max_w = max(int(mask.shape[1]) for mask in valid_masks)

    point_map_world_np = np.full((len(point_map_world), max_h, max_w, 3), np.nan, dtype=np.float32)
    valid_masks_np = np.zeros((len(valid_masks), max_h, max_w), dtype=bool)
    images_net_np = np.zeros((len(images_network_uint8), max_h, max_w, 3), dtype=np.uint8)
    conf_maps_np = np.zeros((len(confidence_maps_network), max_h, max_w), dtype=np.float32)

    for idx, (pts, mask, rgb, conf) in enumerate(
        zip(point_map_world, valid_masks, images_network_uint8, confidence_maps_network)
    ):
        h, w = int(mask.shape[0]), int(mask.shape[1])
        point_map_world_np[idx, :h, :w, :] = pts.astype(np.float32, copy=False)
        valid_masks_np[idx, :h, :w] = mask.astype(bool, copy=False)
        images_net_np[idx, :h, :w, :] = rgb.astype(np.uint8, copy=False)
        conf_maps_np[idx, :h, :w] = conf.astype(np.float32, copy=False)

    return point_map_world_np, valid_masks_np, images_net_np, conf_maps_np


# --------------------------------------------------------------------------- #
# Input zoom: the same central window of every image, sized on the markers
# --------------------------------------------------------------------------- #
def network_zoom(
    zoom: float | str,
    zoom_max: float,
    zoom_margin: float,
    image_size: int,
    detections_per_view,
    image_paths,
    config_key: str = "zoom",
) -> Dict[str, Any] | None:
    """The central window of every image that the network sees, or None for
    the whole image (zoom 1).

    One window for every image: the servo centres the markers, and shared
    intrinsics need a single camera. 'auto': the largest zoom (<= zoom_max)
    whose window holds every detected marker corner with zoom_margin around
    it. Extents are fractions of the half image from its centre (1 = touches
    the border). ``image_size``: the network's long side, only to flag a window
    smaller than it (upsampled). ``config_key``: for the messages."""
    from rich import print

    if zoom != "auto" and float(zoom) <= 1.0:
        return None
    from PIL import Image

    sizes = set()
    for p in image_paths:
        with Image.open(p) as im:
            sizes.add(im.size)
    if len(sizes) != 1:
        print(f"[yellow]{config_key} ignored: the images do not all have the same size "
              f"({sorted(sizes)}).[/yellow]")
        return None
    W, H = sizes.pop()
    ext: List[float | None] = []
    for dets in (detections_per_view or [[] for _ in image_paths]):
        if not dets:
            ext.append(None)
            continue
        c = np.concatenate([np.asarray(d.corners_xy, dtype=np.float64).reshape(-1, 2) for d in dets])
        ext.append(float(max(np.abs(c[:, 0] - W / 2).max() / (W / 2),
                             np.abs(c[:, 1] - H / 2).max() / (H / 2))))
    seen = [e for e in ext if e is not None]
    need = [None if e is None else e * (1.0 + float(zoom_margin)) for e in ext]
    if zoom == "auto":
        if not seen:
            print(f"[yellow]{config_key} auto: no ArUco marker detected, the network gets the "
                  "whole images.[/yellow]")
            return None
        z = min(float(zoom_max), 1.0 / max(n for n in need if n is not None))
        if z < 1.05:
            print(f"[dim]{config_key} auto: the markers reach {max(seen):.2f} of the half "
                  "image: no zoom.[/dim]")
            return None
    else:
        z = float(zoom)
    # even sizes, rounded up so that the window holds what 'auto' asked for
    w, h = (min(n, int(np.ceil(n / z / 2 - 1e-9)) * 2) for n in (W, H))
    x0, y0 = (W - w) // 2, (H - h) // 2
    half = min(w / W, h / H)                 # window half-size, fraction of the half image
    return {
        "mode": "auto" if zoom == "auto" else "fixed",
        "zoom_config": zoom,
        "zoom": float(W / w),
        "crop_px": [int(x0), int(y0), int(w), int(h)],
        "image_wh": [int(W), int(H)],
        "margin": float(zoom_margin),
        "marker_extent_max": max(seen) if seen else None,
        "limiting_view": int(np.argmax([-1.0 if e is None else e for e in ext])) if seen else None,
        "views_at_risk": [i for i, n in enumerate(need) if n is not None and n > half + 1e-9],
        "upsampled": bool(max(w, h) < int(image_size)),
    }


def report_network_zoom(zoom_info: Dict[str, Any], image_size: int, out_dir_name: str,
                        config_key: str = "zoom", model: str = "the network") -> None:
    """Prints the zoom chosen by `network_zoom` and its warnings."""
    from rich import print

    x0, y0, w, h = zoom_info["crop_px"]
    ext = zoom_info["marker_extent_max"]
    print(
        f"Input zoom [green]{zoom_info['zoom']:.2f}x[/green] ({zoom_info['mode']}): "
        f"{model} gets the central {w}x{h} px of the "
        f"{zoom_info['image_wh'][0]}x{zoom_info['image_wh'][1]} images ({out_dir_name}/)"
        + ("" if ext is None else f"; the detected markers reach {ext:.2f} of the half image "
           f"(view {zoom_info['limiting_view']})")
    )
    if zoom_info["views_at_risk"]:
        print(
            f"[yellow]Zoom {zoom_info['zoom']:.2f}x: in views {zoom_info['views_at_risk']} the "
            f"detected markers x (1 + zoom_margin) go beyond the window: the sample may be cut. "
            f"Lower {config_key} or use 'auto'.[/yellow]"
        )
    if zoom_info["upsampled"]:
        print(
            f"[yellow]Zoom {zoom_info['zoom']:.2f}x: the window is smaller than image_size "
            f"({image_size} px): {model} gets an upsampled image, no detail is gained.[/yellow]"
        )


def write_network_crops(image_paths, crop, out_dir: Path) -> List[Path]:
    """The zoom window of every image, as PNG (lossless) in ``out_dir``."""
    from PIL import Image

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    x0, y0, w, h = (int(c) for c in crop)
    out = []
    for p in image_paths:
        q = out_dir / (Path(p).stem + ".png")
        with Image.open(p) as im:
            im.convert("RGB").crop((x0, y0, x0 + w, y0 + h)).save(q)
        out.append(q)
    return out


__all__ = [
    "ImageGeometry",
    "assemble_feedforward",
    "camera_depth",
    "centered_window",
    "crop_to_aspect",
    "depth_edges",
    "gt_conditioning",
    "intrinsics_from_point_maps",
    "prepare_feedforward_images",
    "resolve_hf_or_path",
    "unproject_depth",
    "zoom_window",
    "colors_on_network_grid",
    "fuse_points",
    "invert_rigid",
    "network_zoom",
    "pad_view_maps_for_stacking",
    "report_network_zoom",
    "weighted_voxel_downsample",
    "write_network_crops",
]


# --------------------------------------------------------------------------- #
# Feed-forward multi-view models (VGGT-Omega, Pi3, DA3, MapAnything): shared
# input preparation and output assembly. Each back-end brings its own resize
# rule, model call and point/confidence semantics.
# --------------------------------------------------------------------------- #
def centered_window(window, width: int, height: int, K=None):
    """(x0, y0, w, h): ``window`` (or the whole image) re-centred on the
    principal point of ``K`` when given (shrunk to stay inside the image),
    even sizes; the centre x0 + w/2 then lies within half a pixel of (cx, cy).
    For models whose cameras have the principal point at the image centre."""
    x0, y0, w, h = window if window is not None else (0, 0, width, height)
    if K is None:
        return int(x0), int(y0), int(w), int(h)
    cx, cy = float(K[0, 2]), float(K[1, 2])
    hw = min(w / 2.0, cx, width - cx)
    hh = min(h / 2.0, cy, height - cy)
    w, h = 2 * int(np.floor(hw)), 2 * int(np.floor(hh))
    x0 = int(np.clip(round(cx - w / 2.0), 0, width - w))
    y0 = int(np.clip(round(cy - h / 2.0), 0, height - h))
    return x0, y0, w, h


def crop_to_aspect(window, aspect_wh: float):
    """Centre crop of ``window`` (x0, y0, w, h) to the aspect ratio w/h of the
    network input, so that the resize scales x and y alike (square pixels)."""
    x0, y0, w, h = window
    if w / h > aspect_wh + 1e-9:
        nw = min(w, max(2, 2 * int(round(h * aspect_wh / 2.0))))
        return x0 + (w - nw) // 2, y0, nw, h
    if w / h < aspect_wh - 1e-9:
        nh = min(h, max(2, 2 * int(round(w / aspect_wh / 2.0))))
        return x0, y0 + (h - nh) // 2, w, nh
    return x0, y0, w, h


def prepare_feedforward_images(
    image_paths,
    window,
    K_orig,
    center_principal_point: bool,
    target_shape,
    resample=None,
    match_aspect: bool = True,
    save_dir: Path | None = None,
):
    """Network inputs of a feed-forward model, with their geometry.

    ``window``: zoom window (x0, y0, w, h) or None; re-centred on the
    calibrated principal point if ``center_principal_point`` and ``K_orig``;
    ``target_shape(w, h) -> (net_w, net_h)``: the model's own resize rule;
    with ``match_aspect`` the window is then cropped to the network aspect.
    Returns (images [V, Hn, Wn, 3] uint8, geometries, originals [V, H, W, 3]
    uint8, window used (x0, y0, w, h))."""
    from PIL import Image

    resample = Image.Resampling.BICUBIC if resample is None else resample
    nets, geometries, originals, used = [], [], [], None
    for path in image_paths:
        with Image.open(path) as im:
            rgb = im.convert("RGB")
        W0, H0 = rgb.size
        originals.append(np.asarray(rgb, dtype=np.uint8))
        win = centered_window(window, W0, H0, K_orig if center_principal_point else None)
        tw, th = target_shape(win[2], win[3])
        if match_aspect:
            win = crop_to_aspect(win, tw / th)
        x0, y0, wc, hc = win
        used = used or win
        crop = rgb.crop((x0, y0, x0 + wc, y0 + hc))
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            crop.save(Path(save_dir) / (Path(path).stem + ".png"))
        nets.append(np.asarray(crop.resize((tw, th), resample), dtype=np.uint8))
        sx, sy = tw / float(wc), th / float(hc)
        geometries.append(ImageGeometry(
            original_height=H0, original_width=W0,
            resized_height=int(round(H0 * sy)), resized_width=int(round(W0 * sx)),
            network_height=th, network_width=tw,
            scale_x=sx, scale_y=sy, crop_left=x0 * sx, crop_top=y0 * sy,
        ))
    shapes = {n.shape for n in nets}
    if len(shapes) != 1:
        raise RuntimeError(f"Different network sizes {sorted(shapes)}: one batch needs one size "
                           "(images of one camera).")
    return np.stack(nets), geometries, originals, used


def depth_edges(depth: np.ndarray, rtol: float) -> np.ndarray:
    """[V, H, W] bool: pixels whose 3x3 neighbourhood spans a relative depth
    jump > rtol (flying points at occlusion boundaries)."""
    pad = np.pad(depth, ((0, 0), (1, 1), (1, 1)), mode="edge")
    H, W = depth.shape[1:]
    win = np.stack([pad[:, y:y + H, x:x + W] for y in range(3) for x in range(3)])
    return (win.max(0) - win.min(0)) / np.maximum(np.abs(depth), 1e-6) > rtol


def unproject_depth(depth: np.ndarray, K: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """[V, H, W, 3] world points of z-depth maps [V, H, W] (pixel (u, v) at
    integer coordinates, as the intrinsics of ImageGeometry)."""
    V, H, W = depth.shape
    v, u = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    out = np.empty((V, H, W, 3), dtype=np.float32)
    for i in range(V):
        d = depth[i].astype(np.float64)
        Xc = np.stack([(u - K[i, 0, 2]) / K[i, 0, 0] * d, (v - K[i, 1, 2]) / K[i, 1, 1] * d, d], -1)
        R, t = T_world_cam[i, :3, :3].astype(np.float64), T_world_cam[i, :3, 3].astype(np.float64)
        out[i] = (Xc @ R.T + t).astype(np.float32)
    return out


def camera_depth(point_maps: np.ndarray, T_world_cam: np.ndarray) -> np.ndarray:
    """[V, H, W] z-depth of world point maps in their own camera."""
    out = np.empty(point_maps.shape[:3], dtype=np.float32)
    for i in range(point_maps.shape[0]):
        R, t = T_world_cam[i, :3, :3].astype(np.float64), T_world_cam[i, :3, 3].astype(np.float64)
        out[i] = ((point_maps[i].astype(np.float64) - t) @ R)[..., 2].astype(np.float32)
    return out


def intrinsics_from_point_maps(point_maps: np.ndarray, T_world_cam: np.ndarray,
                               valid: np.ndarray) -> np.ndarray:
    """[V, 3, 3] pinhole K (network pixels, integer pixel centres) fitted by
    least squares to the camera-frame directions of the point maps: for
    models that predict points/rays rather than a camera matrix."""
    V, H, W = point_maps.shape[:3]
    v, u = np.meshgrid(np.arange(H, dtype=np.float64), np.arange(W, dtype=np.float64), indexing="ij")
    Ks = np.zeros((V, 3, 3), dtype=np.float32)
    for i in range(V):
        R, t = T_world_cam[i, :3, :3].astype(np.float64), T_world_cam[i, :3, 3].astype(np.float64)
        Xc = (point_maps[i].astype(np.float64) - t) @ R
        m = valid[i] & np.isfinite(Xc).all(-1) & (Xc[..., 2] > 1e-9)
        x, y = Xc[..., 0][m] / Xc[..., 2][m], Xc[..., 1][m] / Xc[..., 2][m]
        fx, cx = np.polyfit(x, u[m], 1)
        fy, cy = np.polyfit(y, v[m], 1)
        Ks[i] = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
    return Ks


def assemble_feedforward(
    *,
    backend_name: str,
    variant: str | None,
    point_maps: np.ndarray,          # [V, Hn, Wn, 3] in the model's world frame
    valid: np.ndarray,               # [V, Hn, Wn] bool
    confidence: np.ndarray,          # [V, Hn, Wn] >= 0, higher = better
    T_world_cam: np.ndarray,         # [V, 4, 4] camera-to-world
    K_network: np.ndarray,           # [V, 3, 3] network pixels
    geometries,
    originals,
    voxel_size_rel: float,
    max_points: int,
    confidence_percentile: float,
    alignment_info: dict,
    metric_frame: bool = False,
    K_orig: np.ndarray | None = None,
    frame_description: str | None = None,
):
    """RawReconstruction of a feed-forward model: colours from the original
    images, fusion with voxel = voxel_size_rel x median camera depth (the
    model's scale is arbitrary), K back in original pixels (or ``K_orig`` when
    the model was given the calibrated one)."""
    from .types import RawReconstruction

    V = point_maps.shape[0]
    if not valid.any():
        raise RuntimeError(f"{backend_name} produced no valid points.")
    colors = [colors_on_network_grid(g, rgb) for g, rgb in zip(geometries, originals)]
    depth = camera_depth(point_maps, T_world_cam)
    median_depth = float(np.median(depth[valid]))
    voxel = float(voxel_size_rel) * median_depth
    fused_points, fused_colors, fused_confidence = fuse_points(
        [point_maps[i][valid[i]] for i in range(V)],
        [colors[i][valid[i]] for i in range(V)],
        [confidence[i][valid[i]].astype(np.float32) for i in range(V)],
        voxel, max_points,
    )
    pm, vm, imgs, cm = pad_view_maps_for_stacking(
        list(point_maps), list(valid), colors, [c.astype(np.float32) for c in confidence])
    if K_orig is None:
        K_orig = np.stack([g.network_to_original_intrinsics(K_network[i])
                           for i, g in enumerate(geometries)])
    info = dict(alignment_info)
    info.update({"median_depth_backend": median_depth, "voxel_size_backend": voxel,
                 "network_size_wh": [int(geometries[0].network_width), int(geometries[0].network_height)]})
    return RawReconstruction(
        fused_points=fused_points,
        fused_colors=fused_colors,
        fused_confidence=fused_confidence,
        point_map_world=pm,
        valid_masks=vm,
        T_world_cam=np.asarray(T_world_cam, dtype=np.float32),
        K_per_view_orig=np.asarray(K_orig, dtype=np.float32),
        K_per_view_network=np.asarray(K_network, dtype=np.float32),
        network_image_sizes=np.array([[g.network_width, g.network_height] for g in geometries], np.int32),
        original_image_sizes=np.array([[g.original_width, g.original_height] for g in geometries], np.int32),
        images_network_uint8=imgs,
        confidence_maps_network=cm,
        frame_description=frame_description or backend_name,
        backend_name=backend_name,
        variant=variant,
        metric_frame=bool(metric_frame),
        voxel_size=voxel,
        confidence_percentile=float(confidence_percentile),
        alignment_info=info,
    )


def zoom_window(fcfg, inputs, image_size: int, key: str, model: str, out_dir_name: str):
    """(zoom window (x0, y0, w, h) or None, zoom info) of a feed-forward
    back-end's config section (fields zoom / zoom_max / zoom_margin)."""
    info = network_zoom(fcfg.zoom, fcfg.zoom_max, fcfg.zoom_margin, image_size,
                        inputs.aruco_detections, inputs.image_paths, config_key=f"vision.{key}.zoom")
    if info is None:
        return None, {"zoom": 1.0, "mode": "off"}
    report_network_zoom(info, image_size, out_dir_name, config_key=f"vision.{key}.zoom", model=model)
    return tuple(int(c) for c in info["crop_px"]), info


def gt_conditioning(conditioning: str, inputs, key: str):
    """(K_orig or None, T_world_cam or None) the model is conditioned on:
    'k' = calibrated intrinsics, 'kpose' = + robot camera poses. Refuses when
    they are missing (no silent fallback to RGB only)."""
    if conditioning == "none":
        return None, None
    if inputs.K_orig_gt is None:
        raise RuntimeError(f"vision.{key}.conditioning={conditioning!r} needs the calibrated "
                           "intrinsics (camera_parameters/intrinsics.npy), which are missing.")
    if conditioning == "k":
        return np.asarray(inputs.K_orig_gt, dtype=np.float64), None
    if inputs.T_world_cam_gt is None or len(inputs.T_world_cam_gt) != len(inputs.image_paths):
        raise RuntimeError(f"vision.{key}.conditioning='kpose' needs one robot pose per image "
                           "(poses/pose_*.txt), which are missing.")
    return np.asarray(inputs.K_orig_gt, dtype=np.float64), np.asarray(inputs.T_world_cam_gt, dtype=np.float64)


def resolve_hf_or_path(name: str) -> str:
    """A local file/folder as an absolute path, else the name unchanged
    (Hugging Face repo id, resolved by the model's from_pretrained)."""
    p = Path(str(name)).expanduser()
    return str(p.resolve()) if p.exists() else str(name)
