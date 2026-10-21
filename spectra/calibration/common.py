# -*- coding: utf-8 -*-
"""Shared by acquisition, intrinsics, hand-eye and validation: settings from the
unified config, pose conventions, chessboard detection, file pairing and
intrinsics loading.

View files: imgXXXX.png + posaXXXX.txt (flange pose x,y,z mm; rx,ry,rz deg,
Fairino Euler convention), TCP = bare flange (tool 0).
"""

from __future__ import annotations

import glob
import json
import math
import os

import cv2 as cv
import numpy as np

CB_FLAGS = (cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE +
            cv.CALIB_CB_FILTER_QUADS)
SUBPIX = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# -------------------------------------------------------------- settings ---
def settings(ucfg) -> dict:
    """Plain dict used by the calibration modules, from a UnifiedConfig.

    Robot (ip, tool, euler_order) and camera stream (width, height, fps,
    n_avg_frames) come from `acquisition.robot` / `acquisition.camera`; the
    rest from `calibration:`. Paths are fixed under
    <data_root>/camera_calibration/{input,output}/{calibration,validation,intrinsics}.
    """
    if ucfg.calibration is None:
        raise SystemExit("[calibration] `calibration:` section missing in the config")
    cal = ucfg.calibration.model_dump()
    acq = ucfg.acquisition or {}
    robot, cam = acq.get("robot") or {}, acq.get("camera") or {}
    euler = str(robot.get("euler_order", "ZYX")).upper()
    if euler not in ("ZYX", "XYZ"):
        raise SystemExit(f"[calibration] acquisition.robot.euler_order is {euler!r}: the "
                         "calibration needs the explicit convention, ZYX or XYZ.")

    root = os.path.abspath(os.path.join(str(ucfg.data_root), "camera_calibration"))
    paths = {
        "root": root,
        "cal_in": os.path.join(root, "input", "calibration"),
        "val_in": os.path.join(root, "input", "validation"),
        "cal_out": os.path.join(root, "output", "calibration"),
        "intr_out": os.path.join(root, "output", "intrinsics"),
        "val_out": os.path.join(root, "output", "validation"),
    }
    for d in list(paths.values()):
        os.makedirs(d, exist_ok=True)
    paths["intr_realsense"] = os.path.join(paths["cal_in"], "intrinsics_realsense.json")
    paths["intr_opencv"] = os.path.join(paths["intr_out"], "intrinsics_opencv.json")
    f = cal["intrinsics"]["file"]
    paths["intr_file"] = os.path.abspath(os.path.expanduser(f)) if f else None
    paths["cam2gripper"] = os.path.join(paths["cal_out"], "cam2gripper.txt")

    cal["hand_eye"]["euler_order"] = euler
    cal["acquisition"].update(robot_ip=robot.get("ip", "192.168.58.2"),
                              expected_tool=int(robot.get("tool", 0)))
    cal["camera"].update(width=int(cam.get("width", 1280)), height=int(cam.get("height", 800)),
                         fps=int(cam.get("fps", 15)),
                         n_avg_frames=int(cam.get("n_avg_frames", 5)))
    cal["_paths"] = paths
    return cal


# ----------------------------------------------------------------- poses ---
def euler_deg_to_R(rx, ry, rz, order="ZYX"):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx) if order == "ZYX" else (Rx @ Ry @ Rz)


def to_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.ravel(t)
    return T


def mean_rotation(Rs):
    """Rotation closest to the sum of Rs (SVD projection)."""
    U, _, Vt = np.linalg.svd(sum(Rs))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def pose_to_T(pose, order="ZYX"):
    return to_T(euler_deg_to_R(pose[3], pose[4], pose[5], order), pose[:3])


def read_pose(path):
    with open(path) as f:
        txt = f.read().strip().replace('[', '').replace(']', '').replace(';', ',')
    vals = [float(v) for v in (txt.split(',') if ',' in txt else txt.split())]
    if len(vals) < 6:
        raise ValueError(f"Invalid pose file: {path}")
    return vals[:6]


# ------------------------------------------------------------ chessboard ---
def board_size(cfg):
    return tuple(cfg["board"]["chessboard_size"])


def board_points(cfg):
    """3D chessboard corners (mm) in the board frame."""
    cols, rows = board_size(cfg)
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return objp * float(cfg["board"]["square_size_mm"])


def detect_board(img, size):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ok, corners = cv.findChessboardCorners(gray, size, CB_FLAGS)
    if not ok:
        return None
    return cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), SUBPIX)


# ----------------------------------------------------------------- files ---
def list_images(folder):
    return sorted(glob.glob(os.path.join(folder, "img*.png")) +
                  glob.glob(os.path.join(folder, "img*.jpg")))


def list_views(folder, only=None):
    """[(num, img_path, pose_path)] for every imgXXXX with a matching posaXXXX.txt."""
    imgs = list_images(folder)
    if only:
        imgs = [p for p in imgs if os.path.basename(p) in set(only)]
    out = []
    for ip in imgs:
        num = os.path.splitext(os.path.basename(ip))[0][3:]
        pp = os.path.join(folder, f"posa{num}.txt")
        if os.path.exists(pp):
            out.append((num, ip, pp))
        else:
            print(f"  [skip] img{num}: posa{num}.txt missing")
    return out


def next_index(folder):
    """Progressive numbering: continue after the files already in the folder."""
    nums = [int(s) for s in (os.path.splitext(os.path.basename(p))[0][3:]
                             for p in glob.glob(os.path.join(folder, "img*.png")))
            if s.isdigit()]
    return (max(nums) if nums else 0) + 1


# ------------------------------------------------------------ intrinsics ---
def realsense_intrinsics(profile, rs):
    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    return {"K": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
            "D": [list(intr.coeffs)],
            "img_size": [intr.width, intr.height],
            "source": "realsense factory"}


def _read_realsense_live(cfg):
    import pyrealsense2 as rs
    C = cfg["camera"]
    pipe, rcfg = rs.pipeline(), rs.config()
    rcfg.enable_stream(rs.stream.color, C["width"], C["height"], rs.format.bgr8, C["fps"])
    prof = pipe.start(rcfg)
    try:
        return realsense_intrinsics(prof, rs)
    finally:
        pipe.stop()


def load_intrinsics(cfg):
    """K, D and info dict from calibration.intrinsics.source. The same values
    are used for calibration AND validation."""
    P = cfg["_paths"]
    source = cfg["intrinsics"]["source"]
    if source == "realsense":
        path = P["intr_realsense"]
        if not os.path.exists(path):
            print(f"    {os.path.basename(path)} not found: reading factory "
                  "intrinsics from the connected RealSense...")
            with open(path, "w") as f:
                json.dump(_read_realsense_live(cfg), f, indent=2)
    elif source == "opencv":
        path = P["intr_opencv"]
        if not os.path.exists(path):
            raise SystemExit(f"{path} not found: run `spectra calibrate --modality intrinsics` first.")
    else:   # file
        path = P["intr_file"]
        if not path or not os.path.exists(path):
            raise SystemExit(f"calibration.intrinsics.file not found: {path}")

    with open(path) as f:
        d = json.load(f)
    K, D = np.array(d["K"], float), np.array(d["D"], float)
    info = {"source": source, "path": path, "img_size": d.get("img_size")}
    print(f"    Intrinsics [{source}] from {path}")
    print(f"    fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    return K, D, info


def check_image_size(img, info, name):
    """Intrinsics are valid only at the resolution they were computed for."""
    size = info.get("img_size")
    if size and [img.shape[1], img.shape[0]] != list(size):
        raise SystemExit(f"{name}: image is {img.shape[1]}x{img.shape[0]} but intrinsics "
                         f"are for {size[0]}x{size[1]}.")
