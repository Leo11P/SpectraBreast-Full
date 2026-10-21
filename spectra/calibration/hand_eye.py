# -*- coding: utf-8 -*-
"""Hand-eye calibration (eye-in-hand) from input/calibration: solvePnP per view
+ cv.calibrateHandEye, with chessboard-origin disambiguation (a 7x4 board is
symmetric under 180 deg) and iterative outlier rejection.

Views whose PnP reprojection error exceeds calibration.hand_eye.max_reproj_err_px
are EXCLUDED before the hand-eye solve.

Output (output/calibration): cam2gripper.txt (4x4, mm), matrici_4x4.txt
(target2cam), gripper2base.txt, kept_images.txt, intrinsics_used.json.
The sample acquisition reads cam2gripper.txt when
acquisition.hand_eye.T_cam2gripper is null.
"""

from __future__ import annotations

import json
import math
import os

import cv2 as cv
import numpy as np

from . import common
from .common import to_T as _T

METHODS = {"PARK": cv.CALIB_HAND_EYE_PARK, "TSAI": cv.CALIB_HAND_EYE_TSAI,
           "HORAUD": cv.CALIB_HAND_EYE_HORAUD, "ANDREFF": cv.CALIB_HAND_EYE_ANDREFF,
           "DANIILIDIS": cv.CALIB_HAND_EYE_DANIILIDIS}


def reproj_error_px(objp, corners, rvec, tvec, K, D):
    proj, _ = cv.projectPoints(objp, rvec, tvec, K, D)
    return float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) -
                                         corners.reshape(-1, 2))**2, axis=1))))


def collect_pairs(cfg, K, D, info):
    P, H = cfg["_paths"], cfg["hand_eye"]
    objp, size = common.board_points(cfg), common.board_size(cfg)
    max_err = float(H["max_reproj_err_px"])

    pairs, names = [], []
    n_nopat = n_reproj = 0
    for num, ip, pp in common.list_views(P["cal_in"], H["only_images"] or None):
        img = cv.imread(ip)
        if img is None:
            continue
        common.check_image_size(img, info, f"img{num}")
        corners = common.detect_board(img, size)
        if corners is None:
            n_nopat += 1
            continue

        v = common.read_pose(pp)
        Rg = common.euler_deg_to_R(v[3], v[4], v[5], H["euler_order"])
        tg = np.array([[v[0]], [v[1]], [v[2]]])      # mm

        # both corner orderings: the board origin is ambiguous (180 deg symmetry)
        cands, errs = [], []
        for c in (corners, corners[::-1].copy()):
            _, rvec, tvec = cv.solvePnP(objp, c, K, D)
            R, _ = cv.Rodrigues(rvec)
            cands.append((R, tvec))
            errs.append(reproj_error_px(objp, c, rvec, tvec, K, D))
        if min(errs) > max_err:
            n_reproj += 1
            print(f"  [excluded] img{num}: reproj {min(errs):.3f} px > {max_err} px")
            continue

        pairs.append((Rg, tg, cands[0], cands[1]))
        names.append(num)

    print(f"  Valid pairs: {len(pairs)}  (discarded: {n_nopat} without chessboard, "
          f"{n_reproj} reproj > {max_err} px)")
    return pairs, names


def _solve(pairs, choice, mask, method):
    A, B, Cr, Ct = [], [], [], []
    for (Rg, tg, c1, c2), ch, m in zip(pairs, choice, mask):
        if not m:
            continue
        A.append(Rg)
        B.append(tg)
        R, t = c1 if ch == 0 else c2
        Cr.append(R)
        Ct.append(t)
    return cv.calibrateHandEye(A, B, Cr, Ct, method=method)


def _target2base(pairs, choice, Tcg):
    out = []
    for (Rg, tg, c1, c2), ch in zip(pairs, choice):
        Rc, tc = c1 if ch == 0 else c2
        out.append(_T(Rg, tg) @ Tcg @ _T(Rc, tc))
    return out


def resolve_origin_and_calibrate(pairs, method, outlier_sigma):
    n = len(pairs)
    choice = [0] * n
    mask = [True] * n

    # 1) pick, per view, the corner ordering consistent with the consensus target pose
    for _ in range(50):
        R, t = _solve(pairs, choice, mask, method)
        Tcg = _T(R, t)
        t2b = _target2base(pairs, choice, Tcg)
        Rref = t2b[0][:3, :3]
        tref = np.median(np.array([T[:3, 3] for T in t2b]), axis=0)
        changed = 0
        for i, (Rg, tg, c1, c2) in enumerate(pairs):
            best, bestcost = choice[i], 1e18
            for cc in (0, 1):
                Rc, tc = c1 if cc == 0 else c2
                T = _T(Rg, tg) @ Tcg @ _T(Rc, tc)
                ang = np.degrees(np.arccos(np.clip((np.trace(T[:3, :3] @ Rref.T) - 1) / 2, -1, 1)))
                cost = ang + 0.1 * np.linalg.norm(T[:3, 3] - tref)
                if cost < bestcost:
                    bestcost, best = cost, cc
            if best != choice[i]:
                changed += 1
            choice[i] = best
        if changed == 0:
            break

    # 2) iterative outlier rejection on the target position in the base frame
    for _ in range(3):
        R, t = _solve(pairs, choice, mask, method)
        poss = np.array([T[:3, 3] for T in _target2base(pairs, choice, _T(R, t))])
        tref = np.median(poss, axis=0)
        errs = np.linalg.norm(poss - tref, axis=1)
        thr = np.median(errs) + outlier_sigma * np.std(errs)
        new = [bool(e <= thr) for e in errs]
        if new == mask:
            break
        mask = new

    R, t = _solve(pairs, choice, mask, method)
    return R, t, choice, mask


def per_view_residuals(G_list, C_list, X):
    """target2base_i = G_i @ X @ C_i must be the same for every view (board
    still): per view, rotational (deg) and translational (mm) deviation from
    the robust consensus."""
    T = [G @ X @ C for G, C in zip(G_list, C_list)]
    t_ref = np.median(np.array([Ti[:3, 3] for Ti in T]), axis=0)
    R_ref = common.mean_rotation([Ti[:3, :3] for Ti in T])
    rot_dev = [math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(R_ref.T @ Ti[:3, :3]) - 1) / 2))))
               for Ti in T]
    tr_dev = [float(np.linalg.norm(Ti[:3, 3] - t_ref)) for Ti in T]
    return np.array(rot_dev), np.array(tr_dev)


def format_residuals(names, rot_dev, tr_dev, rot_sigma=2.0):
    thr = float(np.median(rot_dev) + rot_sigma * np.std(rot_dev))
    L = ["    view      rot_dev(deg)   tr_dev(mm)   flag", "    " + "-" * 46]
    for i in np.argsort(rot_dev)[::-1]:
        flag = "  <== ROT OUTLIER" if rot_dev[i] > thr else ""
        L.append(f"    {str(names[i]):>8s}   {rot_dev[i]:9.3f}     {tr_dev[i]:8.2f}{flag}")
    L.append("")
    L.append(f"    rot_dev: median={np.median(rot_dev):.3f}  max={rot_dev.max():.3f}  "
             f"outlier_threshold={thr:.3f} deg")
    L.append(f"    tr_dev : median={np.median(tr_dev):.2f}  max={tr_dev.max():.2f} mm")
    return "\n".join(L)


def save_outputs(out_dir, R, t, pairs, choice, mask, names, info):
    np.savetxt(os.path.join(out_dir, "cam2gripper.txt"), _T(R, t), fmt="%.10f")

    with open(os.path.join(out_dir, "matrici_4x4.txt"), "w") as fcam, \
         open(os.path.join(out_dir, "gripper2base.txt"), "w") as fgb:
        for (Rg, tg, c1, c2), ch, m in zip(pairs, choice, mask):
            if not m:
                continue
            Rc, tc = c1 if ch == 0 else c2
            np.savetxt(fcam, _T(Rc, tc), fmt="%.6f")
            fcam.write("\n")
            np.savetxt(fgb, _T(Rg, tg), fmt="%.6f")
            fgb.write("\n")

    with open(os.path.join(out_dir, "kept_images.txt"), "w") as fk:
        fk.write(f"# Kept images: {sum(mask)}/{len(pairs)}\n")
        fk.write(f"# Flipped origin: {sum(c for c, m in zip(choice, mask) if m)}\n\n")
        for ch, m, name in zip(choice, mask, names):
            status = "OK" if m else "OUTLIER"
            flip = "flipped" if (m and ch == 1) else ""
            fk.write(f"{status:8s}  img{name}.png  {flip}\n")

    with open(os.path.join(out_dir, "intrinsics_used.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(f"[Output] cam2gripper.txt, matrici_4x4.txt, gripper2base.txt, "
          f"kept_images.txt, intrinsics_used.json in {out_dir}")


def run(cfg):
    P, H = cfg["_paths"], cfg["hand_eye"]
    method = METHODS[H["method"]]

    print("\n[1] Intrinsics")
    K, D, info = common.load_intrinsics(cfg)

    print(f"\n[2] Pairs from {P['cal_in']} (reproj filter: {H['max_reproj_err_px']} px)")
    pairs, names = collect_pairs(cfg, K, D, info)
    if len(pairs) < 3:
        raise SystemExit("Fewer than 3 valid pairs: cannot calibrate.")

    print(f"\n[3] Origin disambiguation + hand-eye {H['method']} ({len(pairs)} pairs)")
    R, t, choice, mask = resolve_origin_and_calibrate(pairs, method, float(H["outlier_sigma"]))
    print(f"    kept views: {sum(mask)}/{len(pairs)}, "
          f"flipped: {sum(c for c, m in zip(choice, mask) if m)}")

    Tcg = _T(R, t)
    kept = [(p, ch, nm) for p, ch, m, nm in zip(pairs, choice, mask, names) if m]
    G_list = [_T(Rg, tg) for (Rg, tg, _, _), _, _ in kept]
    C_list = [_T(*(c1 if ch == 0 else c2)) for (_, _, c1, c2), ch, _ in kept]
    poss = np.array([(G @ Tcg @ Cm)[:3, 3] for G, Cm in zip(G_list, C_list)])
    std = np.std(poss, axis=0)
    print(f"\n[4] Self-check chessboard->base: std={np.round(std, 2)} mm "
          f"(|std|={np.linalg.norm(std):.2f} mm) -> small = good")

    rot_dev, tr_dev = per_view_residuals(G_list, C_list, Tcg)
    print("\n[5] Per-view residuals (target->base consistency; high rot = rotational outlier):")
    print(format_residuals([nm for _, _, nm in kept], rot_dev, tr_dev))

    print("\n[Result] cam2gripper:")
    print("  t (mm):", np.round(t.ravel(), 2))
    print("  R:\n", np.round(R, 4))
    save_outputs(P["cal_out"], R, t, pairs, choice, mask, names,
                 {**info, "K": K.tolist(), "D": D.tolist()})
    return Tcg
