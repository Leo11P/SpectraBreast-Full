# -*- coding: utf-8 -*-
"""OpenCV intrinsic calibration of the RealSense from the chessboard images in
input/calibration. Writes ONLY output/intrinsics/intrinsics_opencv.json (no
input file is touched). Used when calibration.intrinsics.source is `opencv`.
"""

from __future__ import annotations

import json
import os

import cv2 as cv
import numpy as np

from . import common


def run(cfg):
    P = cfg["_paths"]
    size = common.board_size(cfg)
    objp = common.board_points(cfg)
    min_views = int(cfg["intrinsics"]["min_views_warn"])

    imgs = common.list_images(P["cal_in"])
    if not imgs:
        raise SystemExit(f"No img*.png in {P['cal_in']}")
    print(f"\nImages found: {len(imgs)}")

    objpoints, imgpoints, used = [], [], []
    img_size = None
    for ip in imgs:
        img = cv.imread(ip)
        if img is None:
            continue
        if img_size is None:
            img_size = (img.shape[1], img.shape[0])
        elif (img.shape[1], img.shape[0]) != img_size:
            print(f"  [skip] {os.path.basename(ip)}: different resolution")
            continue
        corners = common.detect_board(img, size)
        if corners is None:
            continue
        objpoints.append(objp)
        imgpoints.append(corners)
        used.append(os.path.basename(ip))

    print(f"Chessboard detected in {len(used)} images (resolution {img_size})")
    if len(used) < min_views:
        print(f"WARNING: fewer than {min_views} valid views, the intrinsics may be "
              "unreliable. Prefer 15+ views from different angles.")
    if len(used) < 3:
        raise SystemExit("Too few views: cannot calibrate.")

    rms, K, D, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    tot_err, tot_pts, per_img = 0.0, 0, []
    for i in range(len(objpoints)):
        proj, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, D)
        e = cv.norm(imgpoints[i], proj, cv.NORM_L2) / len(proj)
        per_img.append((used[i], e))
        tot_err += e**2 * len(proj)
        tot_pts += len(proj)
    mean_reproj = np.sqrt(tot_err / tot_pts)

    print("\n" + "-" * 64)
    print(f"RMS (OpenCV):              {rms:.4f} px")
    print(f"Mean reprojection error:   {mean_reproj:.4f} px")
    print(f"\nfx={K[0,0]:.3f}  fy={K[1,1]:.3f}  cx={K[0,2]:.3f}  cy={K[1,2]:.3f}")
    print(f"Distortion [k1 k2 p1 p2 k3]: {np.round(D.ravel(), 5)}")

    per_img.sort(key=lambda r: r[1], reverse=True)
    print("\nViews with the highest error (candidates to check):")
    for name, e in per_img[:5]:
        print(f"  {name:>16}  {e:.4f} px")

    out = {"K": K.tolist(), "D": D.tolist(), "img_size": list(img_size),
           "source": "opencv calibrateCamera", "rms_px": float(rms),
           "mean_reproj_px": float(mean_reproj), "n_views": len(used)}
    with open(P["intr_opencv"], "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[Output] {P['intr_opencv']}")
    if cfg["intrinsics"]["source"] != "opencv":
        print("Not used by hand-eye / validation: calibration.intrinsics.source is "
              f"{cfg['intrinsics']['source']!r} (set it to opencv to use it).")
    return K, D, mean_reproj
