"""Shared helpers for loading RGB images, GT poses, and GT intrinsics.

These load routines are used by both back-ends and by the top-level
orchestrator. Missing files are handled gracefully so the pipeline can run in
"RGB-only" mode when poses/intrinsics are absent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from PIL.ImageOps import exif_transpose

from .aruco import IMAGE_EXTENSIONS
from .backends import BackendInputs
from .transforms import xyzeuler_to_hmat


def list_rgb_images(rgb_dir: Path | None) -> List[Path]:
    """Return a sorted list of image paths in a folder."""
    if rgb_dir is None:
        raise FileNotFoundError(
            "No RGB folder resolved. Either set `vision.input.rgb_dir` in the "
            "YAML (relative or absolute), or place the images under "
            "DATA/input_vision/<sample_name>[_<device>]/rgb so it can be "
            "derived automatically."
        )
    rgb_dir = Path(rgb_dir)
    if not rgb_dir.is_dir():
        raise FileNotFoundError(f"RGB folder does not exist: {rgb_dir}")
    files = sorted(
        p for p in rgb_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not files:
        raise FileNotFoundError(f"No image files found in {rgb_dir}")
    return files


def canonicalize_images_with_exif(
    image_paths: List[Path],
    cache_dir: Path,
) -> List[Path]:
    """Bake EXIF orientation into pixel buffers and write to ``cache_dir``.

    iPhones and other cameras often store photos in a fixed sensor orientation
    and use the EXIF ``Orientation`` tag to tell viewers how to rotate/flip
    them for correct display. ``cv2.imread`` / ``torchvision.io.read_image`` /
    ``PIL.Image.open`` all ignore that tag by default, so without this step
    downstream code sees raw buffers that may be upside-down or mirrored.

    Mixed orientations break dense-prediction MASt3R (and similar) models whose
    pose heads assume upright, non-mirrored images. Applying EXIF upfront
    guarantees every consumer (back-end, ArUco detector, Rerun, TUI) works in
    the same image frame.

    The output directory is cleared before writing, and paths are preserved
    in the same sorted order as ``image_paths``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    for existing in cache_dir.iterdir():
        if existing.is_file():
            existing.unlink()

    canonical_paths: List[Path] = []
    for src in image_paths:
        # Reserve the sidecar .png extension for everything so we don't have
        # to deal with lossy re-encoding of JPEG->JPEG (which would change
        # pixel values and invalidate ArUco subpixel positions).
        dst = cache_dir / (src.stem + ".png")
        with Image.open(src) as pil_img:
            canonical = exif_transpose(pil_img).convert("RGB")
            canonical.save(dst, format="PNG", compress_level=1)
        canonical_paths.append(dst)
    return canonical_paths


def _fix_3x4_to_4x4(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=np.float32)
    if T.shape == (4, 4):
        return T
    if T.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :4] = T
        return out
    raise ValueError(f"Expected (4,4) or (3,4), got {T.shape}")


def load_gt_cameras(
    pose_dir: Path | None,
    camera_params_dir: Path | None,
    num_images: int,
    device: torch.device | None = None,
) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Load GT camera-to-world poses and intrinsics if available.

    Returns ``(T_world_cam, K_orig, cam2ee, distortion)`` where each element
    is ``None`` when the corresponding file is missing.
    """
    T_world_cam: np.ndarray | None = None
    K_orig: np.ndarray | None = None
    cam2ee: np.ndarray | None = None
    distortion: np.ndarray | None = None

    if pose_dir is not None:
        pose_dir = Path(pose_dir)
        if pose_dir.exists():
            pose_files = sorted(pose_dir.glob("pose_*.txt"))
            if len(pose_files) == num_images:
                poses6 = np.stack(
                    [np.asarray(pf.read_text().strip().split(), dtype=np.float32) for pf in pose_files],
                    axis=0,
                )  # [V, 6]
                dev = device if device is not None else torch.device("cpu")
                poses6_t = torch.as_tensor(poses6, dtype=torch.float32, device=dev)
                T_world_ee = xyzeuler_to_hmat(
                    poses6_t,
                    convention="ROLLPITCHYAW",
                    translation_scale=1.0,
                )  # [V, 4, 4]

                if camera_params_dir is not None:
                    cam2ee_path = Path(camera_params_dir) / "camera2ee.npy"
                    if cam2ee_path.exists():
                        cam2ee = _fix_3x4_to_4x4(np.load(cam2ee_path))
                if cam2ee is None:
                    cam2ee = np.eye(4, dtype=np.float32)

                T_ee_cam_t = torch.as_tensor(cam2ee, dtype=torch.float32, device=dev)
                T_ee_cam_batch = T_ee_cam_t.unsqueeze(0).repeat(num_images, 1, 1)
                T_world_cam = torch.matmul(T_world_ee, T_ee_cam_batch).detach().cpu().numpy().astype(np.float32)
            else:
                # Poses are zipped to images by sorted order, so a count
                # mismatch makes every pose unusable. Say so instead of
                # silently falling back to RGB-only SfM.
                print(
                    f"[WARNING] {pose_dir}: found {len(pose_files)} pose_*.txt for "
                    f"{num_images} images -> GT poses IGNORED, running RGB-only SfM "
                    f"(scale from ArUco only). Pose files must match images 1:1."
                )

    if camera_params_dir is not None:
        cparams = Path(camera_params_dir)
        intrinsics_path = cparams / "intrinsics.npy"
        distortions_path = cparams / "distortions.npy"
        if intrinsics_path.exists():
            K_orig = np.load(intrinsics_path).astype(np.float32)
            if K_orig.shape != (3, 3):
                raise ValueError(f"intrinsics.npy must be [3,3], got {K_orig.shape}")
        if distortions_path.exists():
            distortion = np.load(distortions_path).astype(np.float32).reshape(-1)

    return T_world_cam, K_orig, cam2ee, distortion


def build_backend_inputs(
    rgb_dir: Path,
    pose_dir: Path | None,
    camera_params_dir: Path | None,
    device: torch.device | None = None,
) -> BackendInputs:
    """Gather everything the back-end needs in a single `BackendInputs`."""
    image_paths = list_rgb_images(rgb_dir)
    T_world_cam_gt, K_orig_gt, cam2ee, distortion = load_gt_cameras(
        pose_dir=pose_dir,
        camera_params_dir=camera_params_dir,
        num_images=len(image_paths),
        device=device,
    )
    return BackendInputs(
        image_paths=image_paths,
        pose_dir=pose_dir,
        camera_params_dir=camera_params_dir,
        T_world_cam_gt=T_world_cam_gt,
        K_orig_gt=K_orig_gt,
        cam2ee=cam2ee,
        distortion_coeffs=distortion,
    )


# ---------------------------------------------------------------------------
# Lens undistortion of the input images
# ---------------------------------------------------------------------------
# Every stage after this one (MASt3R, ArUco triangulation, validation) uses a
# PINHOLE camera with K = camera_parameters/intrinsics.npy. The images are
# therefore remapped to an ideal pinhole camera with the SAME K, so
# intrinsics.npy stays valid for them. The distortion model depends on where
# the intrinsics came from (camera_parameters/intrinsics_source.json, written
# by the acquisition):
#   source calibration / override (OpenCV calibration, e.g. the GC1020C):
#       OpenCV (k1, k2, p1, p2, k3): pixel = distort(ideal)  -> cv2 maps
#   source realsense_factory, model *inverse_brown_conrady* (RealSense colour):
#       librealsense applies the polynomial to the DISTORTED coordinates to get
#       the ideal ones (rs2_deproject_pixel_to_point), so the map is its inverse,
#       solved by fixed-point iteration
#   *brown_conrady* / *modified_brown_conrady*: treated as OpenCV
#   *none*: nothing to do.  Other models (kannala_brandt, ftheta): not supported.
_UNDISTORT_ITERS = 30


def _distortion_model(camera_params_dir: Path | None) -> tuple[str | None, dict]:
    """('opencv' | 'inverse_brown_conrady' | 'none' | None if unknown, intrinsics_source)."""
    src = {}
    if camera_params_dir is not None:
        p = Path(camera_params_dir) / "intrinsics_source.json"
        if p.is_file():
            src = json.loads(p.read_text(encoding="utf-8"))
    source = str(src.get("source") or "")
    model = str(src.get("model") or "").lower()
    if source in ("calibration", "override"):
        return "opencv", src
    if source.endswith("_factory"):
        if "inverse_brown_conrady" in model:
            return "inverse_brown_conrady", src
        if "brown_conrady" in model:
            return "opencv", src
        if model.endswith("none"):
            return "none", src
        raise ValueError(f"intrinsics_source.json: distortion model {src.get('model')!r} is not "
                         "supported by the undistortion (vision.input.undistort)")
    return None, src


def _undistort_maps(K: np.ndarray, dist: np.ndarray, model: str, width: int, height: int):
    """(map_x, map_y) float32: for each pixel of the ideal pinhole image (same K),
    where to read it in the original image."""
    import cv2

    K = np.asarray(K, dtype=np.float64)
    d = np.zeros(5)
    d[: min(5, len(dist))] = np.asarray(dist, dtype=np.float64).ravel()[:5]
    if model == "opencv":
        mx, my = cv2.initUndistortRectifyMap(K, d, None, K, (width, height), cv2.CV_32FC1)
        return mx, my
    if model != "inverse_brown_conrady":
        raise ValueError(model)
    k1, k2, p1, p2, k3 = d
    u, v = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    xu = (u - K[0, 2]) / K[0, 0]
    yu = (v - K[1, 2]) / K[1, 1]

    def ideal_from_distorted(x, y):          # librealsense rs2_deproject_pixel_to_point
        r2 = x * x + y * y
        f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        return (x * f + 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x),
                y * f + 2.0 * p2 * x * y + p1 * (r2 + 2.0 * y * y))

    xd, yd = xu.copy(), yu.copy()
    for _ in range(_UNDISTORT_ITERS):
        gx, gy = ideal_from_distorted(xd, yd)
        xd, yd = xd - (gx - xu), yd - (gy - yu)
    gx, gy = ideal_from_distorted(xd, yd)
    res_px = float(np.max(np.hypot((gx - xu) * K[0, 0], (gy - yu) * K[1, 1])))
    if res_px > 0.01:
        raise RuntimeError(f"inverse Brown-Conrady undistortion did not converge ({res_px:.3f} px)")
    return ((xd * K[0, 0] + K[0, 2]).astype(np.float32),
            (yd * K[1, 1] + K[1, 2]).astype(np.float32))


def undistort_images_inplace(
    image_paths: List[Path],
    K: np.ndarray | None,
    dist: np.ndarray | None,
    camera_params_dir: Path | None,
    report_path: Path | None = None,
) -> dict:
    """Remaps the (EXIF-canonical, already copied) images in place to an ideal
    pinhole camera with the same K. Returns a report dict ('applied' False with
    the reason when nothing is done); written to ``report_path`` if given."""
    import cv2

    report: dict = {"applied": False}

    def _done(reason):
        report["reason"] = reason
        print(f"[undistort] skipped: {reason}")
        if report_path is not None:
            Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    if K is None or dist is None:
        return _done("no camera_parameters/intrinsics.npy + distortions.npy")
    model, src = _distortion_model(camera_params_dir)
    report.update(model=model, coeffs=np.asarray(dist, dtype=float).ravel().tolist(),
                  intrinsics_source=src.get("source"), intrinsics_model=src.get("model"))
    if model == "none" or not np.any(np.abs(np.asarray(dist)) > 1e-9):
        return _done("zero distortion")
    if model is None:
        return _done("distortions.npy is not zero but camera_parameters/intrinsics_source.json "
                     "does not say its model (OpenCV or RealSense inverse Brown-Conrady): NOT "
                     "undistorted, applying the wrong model would double the error")

    with Image.open(image_paths[0]) as im0:
        width, height = im0.size
    want = (src.get("width"), src.get("height"))
    if None not in want and tuple(int(x) for x in want) != (width, height):
        raise ValueError(f"[undistort] the intrinsics are for {want[0]}x{want[1]} images, the "
                         f"input images are {width}x{height}")
    mx, my = _undistort_maps(K, dist, model, width, height)
    gu, gv = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    shift = np.hypot(mx - gu, my - gv)
    outside = (mx < 0) | (my < 0) | (mx > width - 1) | (my > height - 1)

    for p in image_paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(p)
        if img.shape[1] != width or img.shape[0] != height:
            raise ValueError(f"[undistort] {p.name} is {img.shape[1]}x{img.shape[0]}, "
                             f"expected {width}x{height} like the other images")
        out = cv2.remap(img, mx, my, interpolation=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if not cv2.imwrite(str(p), out):
            raise OSError(f"cannot write {p}")

    report.update(applied=True, width=width, height=height, n_images=len(image_paths),
                  max_shift_px=float(shift.max()), centre_shift_px=float(shift[height // 2, width // 2]),
                  black_border_pct=float(100.0 * outside.mean()))
    print(f"[undistort] {len(image_paths)} images remapped to a pinhole camera with the same K "
          f"({model}): shift max {report['max_shift_px']:.1f} px, centre "
          f"{report['centre_shift_px']:.2f} px, black border {report['black_border_pct']:.2f}% of the image")
    if report["max_shift_px"] > 0.05 * max(width, height):
        print(f"[undistort] WARNING: shift up to {report['max_shift_px']:.0f} px: check the "
              "distortion coefficients (a calibration that did not cover the image corners "
              "extrapolates badly there).")
    if report_path is not None:
        Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


__all__ = [
    "build_backend_inputs",
    "canonicalize_images_with_exif",
    "undistort_images_inplace",
    "list_rgb_images",
    "load_gt_cameras",
]
