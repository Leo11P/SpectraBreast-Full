# -*- coding: utf-8 -*-
"""
common.py
=========
Helpers shared by acquisition, intrinsics, hand-eye and validation:
pose conventions, chessboard detection, file pairing and intrinsics loading.

File convention: imgXXXX.png + posaXXXX.txt (x,y,z in mm; rx,ry,rz in deg,
Fairino Euler ZYX), TCP = bare flange (tool 0).
"""

import glob
import json
import math
import os

import cv2 as cv
import numpy as np

CB_FLAGS = (cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE +
            cv.CALIB_CB_FILTER_QUADS)
SUBPIX = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ----------------------------------------------------------------- poses ---
def euler_deg_to_R(rx, ry, rz, order="ZYX"):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx) if order == "ZYX" else (Rx @ Ry @ Rz)


def pose_to_T(pose, order="ZYX"):
    T = np.eye(4)
    T[:3, :3] = euler_deg_to_R(pose[3], pose[4], pose[5], order)
    T[:3, 3] = pose[:3]
    return T


def read_pose(path):
    with open(path) as f:
        txt = f.read().strip().replace('[', '').replace(']', '').replace(';', ',')
    sep = ',' if ',' in txt else None
    vals = [float(v) for v in (txt.split(sep) if sep else txt.split())]
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
def list_views(folder, only=None):
    """[(num, img_path, pose_path)] for every imgXXXX with a matching posaXXXX.txt."""
    imgs = sorted(glob.glob(os.path.join(folder, "img*.png")) +
                  glob.glob(os.path.join(folder, "img*.jpg")))
    if only:
        wanted = set(only)
        imgs = [p for p in imgs if os.path.basename(p) in wanted]
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
    nums = []
    for p in glob.glob(os.path.join(folder, "img*.png")):
        stem = os.path.splitext(os.path.basename(p))[0][3:]
        if stem.isdigit():
            nums.append(int(stem))
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
    R = cfg["realsense"]
    pipe, rcfg = rs.pipeline(), rs.config()
    rcfg.enable_stream(rs.stream.color, int(R["width"]), int(R["height"]),
                       rs.format.bgr8, int(R["fps"]))
    prof = pipe.start(rcfg)
    try:
        return realsense_intrinsics(prof, rs)
    finally:
        pipe.stop()


def load_intrinsics(cfg):
    """K, D and info dict, from the source chosen in config.yaml.
    The same values are used for calibration AND validation."""
    P = cfg["_paths"]
    source = str(cfg["intrinsics"]["source"]).lower()
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
            raise SystemExit(f"{path} not found: run 'python main.py intrinsics' first.")
    elif source == "file":
        path = P["intr_file"]
        if not path or not os.path.exists(path):
            raise SystemExit(f"intrinsics.file not found: {path}")
    else:
        raise SystemExit(f"Unknown intrinsics.source: {source} (realsense | opencv | file)")

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
