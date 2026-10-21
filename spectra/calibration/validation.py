# -*- coding: utf-8 -*-
"""Hand-eye validation on NEW images (input/validation).

For each view the chessboard pose is observed with solvePnP, then its corners
are PREDICTED through the chain
    T_target2cam_pred = inv(T_cam2grip) @ inv(T_grip2base) @ T_target2base
and compared with the observed ones.
  [X] T_target2base estimated from the VALIDATION images (self-consistency)
  [Y] T_target2base estimated from the CALIBRATION images (out-of-sample, stricter)

Intrinsics: same source as the calibration (same K, D for both sets). Nothing
is written in the input folders. Output: output/validation/validation_report.csv
(+ overlay_X/, overlay_Y/).
"""

from __future__ import annotations

import json
import os

import cv2 as cv
import numpy as np

from . import common


def median_transform(Ts):
    return common.to_T(common.mean_rotation([T[:3, :3] for T in Ts]),
                       np.median(np.array([T[:3, 3] for T in Ts]), axis=0))


def collect_views(cfg, folder, K, D, info):
    """[(num, corners, T_grip2base, T_target2cam_observed, img)]."""
    objp, size = common.board_points(cfg), common.board_size(cfg)
    order = cfg["hand_eye"]["euler_order"]
    views = []
    for num, ip, pp in common.list_views(folder):
        img = cv.imread(ip)
        if img is None:
            continue
        common.check_image_size(img, info, f"img{num}")
        corners = common.detect_board(img, size)
        if corners is None:
            print(f"  [skip] chessboard not detected in img{num}")
            continue
        _, rvec, tvec = cv.solvePnP(objp, corners, K, D)
        T_t2c = common.to_T(cv.Rodrigues(rvec)[0], tvec)
        views.append((num, corners, common.pose_to_T(common.read_pose(pp), order), T_t2c, img))
    return views


def reproj_stats(cfg, views, T_target2base, K, D, T_cam2grip, label, overlay_dir=None):
    """Per view: corner error in px, the same error in mm on the board plane
    (px * Z/f), and the 3D error (mm) between observed and predicted board position."""
    objp = common.board_points(cfg)
    if overlay_dir:
        os.makedirs(overlay_dir, exist_ok=True)
    f_mean = (K[0, 0] + K[1, 1]) / 2.0
    per_view, all_px, all_mm = [], [], []
    for num, corners, T_g2b, T_t2c, img in views:
        T_pred = np.linalg.inv(T_cam2grip) @ np.linalg.inv(T_g2b) @ T_target2base
        rp, _ = cv.Rodrigues(T_pred[:3, :3])
        proj, _ = cv.projectPoints(objp, rp, T_pred[:3, 3], K, D)
        proj, det = proj.reshape(-1, 2), corners.reshape(-1, 2)
        err_px = np.linalg.norm(proj - det, axis=1)

        Z = (T_pred[:3, :3] @ objp.T + T_pred[:3, 3:4])[2]
        err_mm = err_px * Z / f_mean
        err3d_mm = float(np.linalg.norm(T_t2c[:3, 3] - T_pred[:3, 3]))

        all_px.append(err_px)
        all_mm.append(err_mm)
        per_view.append((num, err_px.mean(), np.median(err_px), err_px.std(),
                         err_px.max(), err_mm.mean(), err3d_mm))
        if overlay_dir:
            vis = img.copy()
            for (u, v) in det:
                cv.circle(vis, (int(u), int(v)), 4, (0, 255, 0), 1)     # observed: green
            for (u, v) in proj:
                cv.circle(vis, (int(u), int(v)), 2, (0, 0, 255), -1)    # predicted: red
            cv.imwrite(os.path.join(overlay_dir, f"{label}_{num}.png"), vis)
    return per_view, np.concatenate(all_px), np.concatenate(all_mm)


def print_report(title, per_view, all_px, all_mm):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)
    print(f"GLOBAL px : mean={all_px.mean():.3f}  median={np.median(all_px):.3f}  "
          f"std={all_px.std():.3f}  max={all_px.max():.3f}")
    print(f"GLOBAL mm : mean={all_mm.mean():.3f}  median={np.median(all_mm):.3f}  "
          f"std={all_mm.std():.3f}  max={all_mm.max():.3f}   (on the chessboard plane)")
    err3d = np.array([r[6] for r in per_view])
    print(f"3D ERROR  : mean={err3d.mean():.3f}  median={np.median(err3d):.3f}  "
          f"std={err3d.std():.3f}  max={err3d.max():.3f}  mm  (observed vs predicted board position)")
    print(f"\n  {'view':>8}  {'px_mean':>7}  {'px_med':>7}  {'px_std':>6}  {'px_max':>7}  "
          f"{'mm_mean':>7}  {'3D_mm':>7}")
    for num, m, md, sd, mx, mmm, e3d in sorted(per_view, key=lambda r: r[1], reverse=True):
        print(f"  img{num:>5}  {m:7.3f}  {md:7.3f}  {sd:6.3f}  {mx:7.3f}  {mmm:7.3f}  {e3d:7.3f}")


def _check_same_intrinsics(P, K, D):
    used = os.path.join(P["cal_out"], "intrinsics_used.json")
    if not os.path.exists(used):
        return
    with open(used) as f:
        d = json.load(f)
    if not (np.allclose(np.array(d["K"]), K) and np.allclose(np.array(d["D"]), D)):
        print(f"\n  WARNING: these intrinsics differ from the ones used by the calibration "
              f"({d.get('source')}, {d.get('path')}). Re-run the hand-eye "
              "(`spectra calibrate --modality calibration calibrate`) or change "
              "calibration.intrinsics.source, otherwise X/Y are not comparable.")


def run(cfg):
    P = cfg["_paths"]
    out_dir = P["val_out"]
    overlay = bool(cfg["validation"]["save_overlay"])

    if not os.path.exists(P["cam2gripper"]):
        raise SystemExit(f"{P['cam2gripper']} not found: run the hand-eye first "
                         "(`spectra calibrate --modality calibration calibrate`).")
    T_cam2grip = np.loadtxt(P["cam2gripper"])
    print(f"cam2gripper t (mm): {np.round(T_cam2grip[:3, 3], 2)}")

    K, D, info = common.load_intrinsics(cfg)
    _check_same_intrinsics(P, K, D)

    val_views = collect_views(cfg, P["val_in"], K, D, info)
    if len(val_views) < 2:
        raise SystemExit("Fewer than 2 valid validation images.")
    print(f"Valid validation images: {len(val_views)}")

    # [X] chessboard pose from the VALIDATION images
    T_X = median_transform([G @ T_cam2grip @ Ct for (_, _, G, Ct, _) in val_views])
    pv_X, px_X, mm_X = reproj_stats(cfg, val_views, T_X, K, D, T_cam2grip, "X",
                                    os.path.join(out_dir, "overlay_X") if overlay else None)
    print_report("METRIC X - self-consistency (board pose from the validation images)",
                 pv_X, px_X, mm_X)
    print(f"\n  Board in base [X] (t, mm): {np.round(T_X[:3, 3], 1)}")

    # [Y] chessboard pose from the CALIBRATION images (out-of-sample)
    pv_Y = []
    calib_views = collect_views(cfg, P["cal_in"], K, D, info)
    if len(calib_views) >= 2:
        T_Y = median_transform([G @ T_cam2grip @ Ct for (_, _, G, Ct, _) in calib_views])
        pv_Y, px_Y, mm_Y = reproj_stats(cfg, val_views, T_Y, K, D, T_cam2grip, "Y",
                                        os.path.join(out_dir, "overlay_Y") if overlay else None)
        print_report("METRIC Y - out-of-sample (board pose from the CALIBRATION images)",
                     pv_Y, px_Y, mm_Y)
        print(f"\n  Board in base [Y] (t, mm): {np.round(T_Y[:3, 3], 1)}")
        print(f"  Distance between X and Y board estimates: "
              f"{np.linalg.norm(T_X[:3, 3] - T_Y[:3, 3]):.2f} mm")
    else:
        print("\n[Y skipped] fewer than 2 valid calibration views.")

    csv = os.path.join(out_dir, "validation_report.csv")
    nan3 = (float('nan'),) * 3
    dX = {r[0]: (r[1], r[5], r[6]) for r in pv_X}
    dY = {r[0]: (r[1], r[5], r[6]) for r in pv_Y}
    with open(csv, "w") as f:
        f.write("view,X_px_mean,X_mm_mean,X_3Dmm,Y_px_mean,Y_mm_mean,Y_3Dmm\n")
        for n in sorted(dX):
            xp, xm, x3 = dX[n]
            yp, ym, y3 = dY.get(n, nan3)
            f.write(f"img{n},{xp:.4f},{xm:.4f},{x3:.4f},{yp:.4f},{ym:.4f},{y3:.4f}\n")
    print(f"\n[Output] {csv}")
    if overlay:
        print(f"[Output] overlays in {out_dir} (green = observed, red = predicted)")
