# -*- coding: utf-8 -*-
"""Manual acquisition in DRAG TEACH mode, for both calibration and validation:
- the robot stays in drag mode (you move it by hand)
- live preview of the camera in use (acquisition.camera.type) with chessboard
  detection
- ENTER -> save imgXXXX.png + posaXXXX.txt (only if the chessboard is detected)
- q / ESC -> stop (drag mode off, camera and robot closed)

Intrinsics:
- calibration, RealSense only: the factory intrinsics are written ONCE to
  input/calibration/intrinsics_realsense.json (never overwritten). The GC1020C
  has none: its intrinsics come from `--modality intrinsics`.
- validation: NOTHING but images and poses is written.

Needs the robot-PC install (`uv sync --extra acquisition`, plus Vimba X for the
GC1020C) and the Fairino SDK in spectra/acquisition/fairino/ (shared with the
sample acquisition).
"""

from __future__ import annotations

import json
import os
import sys
import time

import cv2 as cv
import numpy as np

from . import common

KEY_SAVE = (13,)            # ENTER
KEY_QUIT = (ord('q'), 27)   # q / ESC
FAIRINO_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "acquisition")


# ------------------------------------------------------------ intrinsics ---
def _store_factory_intrinsics(cfg, cam):
    path = cfg["_paths"]["intr_realsense"]
    new = common.realsense_intrinsics(cam.profile, cam.rs)
    if not os.path.exists(path):
        with open(path, "w") as f:
            json.dump(new, f, indent=2)
        print(f"[Intrinsics] factory values saved to {path}")
        return
    with open(path) as f:
        old = json.load(f)
    if not np.allclose(np.array(old["K"]), np.array(new["K"]), atol=1e-3):
        print(f"[Intrinsics] WARNING: {path} differs from the connected camera "
              "(different unit or resolution?). File NOT overwritten.")


# ----------------------------------------------------------- acquisition ---
def run(cfg, which):
    """which = 'calibration' | 'validation'. Returns the number of saved views."""
    if FAIRINO_DIR not in sys.path:
        sys.path.insert(0, FAIRINO_DIR)
    from fairino import Robot

    P, A, C = cfg["_paths"], cfg["acquisition"], cfg["camera"]
    save_dir = P["cal_in"] if which == "calibration" else P["val_in"]
    target = int(A["n_calib_target"] if which == "calibration" else A["n_val_target"])
    size = common.board_size(cfg)

    # calibration.camera (exposure / gain / white balance) applies to the RealSense only
    cam = common.cameras_module().open_camera(cfg["camera_cfg"], depth=False,
                                              rs_color_settings=C)
    if which == "calibration" and cam.kind == "realsense":
        _store_factory_intrinsics(cfg, cam)

    robot = Robot.RPC(A["robot_ip"])
    Robot.RPC.is_connect = True
    time.sleep(0.5)
    err, tool = robot.GetActualTCPNum()
    if err == 0 and int(tool) != int(A["expected_tool"]):
        robot.CloseRPC()
        cam.stop()
        raise SystemExit(f"Active tool is {tool}, expected {A['expected_tool']} "
                         "(acquisition.robot.tool): the saved poses would not be flange "
                         "poses. Set the tool on the controller.")
    robot.RobotEnable(1)
    robot.DragTeachSwitch(1)

    n_next = common.next_index(save_dir)
    print(f"\n[{which}] DRAG MODE on: move the robot by hand.")
    print(f"  output folder: {save_dir}  (numbering starts at {n_next:04d})")
    print(f"  ENTER = save, q/ESC = quit. Target: {target} views.\n")

    win = f"{cam.kind} - {which} (ENTER = save, q/ESC = quit)"
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    saved, frame_i, corners = 0, 0, None
    flash_msg, flash_until = "", 0.0
    try:
        while saved < target:
            img = cam.latest()
            if img is None:
                continue

            frame_i += 1
            if frame_i % 3 == 0:     # detection every 3 frames keeps the preview fluid
                corners = common.detect_board(img, size)

            vis = img.copy()
            H, W = vis.shape[:2]
            m = int(A["preview_margin_px"])
            cv.rectangle(vis, (m, m), (W - m, H - m), (200, 200, 200), 1)
            ok = corners is not None
            if ok:
                cv.drawChessboardCorners(vis, size, corners, True)
            cv.rectangle(vis, (0, 0), (W, 34), (0, 0, 0), -1)
            cv.putText(vis, f"{'board OK' if ok else 'board NOT detected'}  |  "
                            f"saved {saved}/{target}  |  next {n_next:04d}",
                       (10, 24), cv.FONT_HERSHEY_SIMPLEX, 0.6,
                       (0, 180, 0) if ok else (0, 0, 255), 2)
            if time.time() < flash_until:
                cv.putText(vis, flash_msg, (10, 70), cv.FONT_HERSHEY_SIMPLEX, 0.9,
                           (0, 0, 255), 2)
            cv.imshow(win, vis)
            key = cv.waitKey(1) & 0xFF

            if cv.getWindowProperty(win, cv.WND_PROP_VISIBLE) < 1 or key in KEY_QUIT:
                break
            if key not in KEY_SAVE:
                continue

            res = robot.GetActualTCPPose(1)
            if not isinstance(res, tuple) or res[0] != 0:
                print(f"  [ERROR] robot pose read failed ({res}) -> NOT saved")
                flash_msg, flash_until = "Pose read ERROR", time.time() + 1.5
                continue
            tcp = res[1]
            snap = cam.grab(C["n_avg_frames"])
            if snap is None:
                print("  [ERROR] no camera frame -> NOT saved")
                flash_msg, flash_until = "Camera ERROR", time.time() + 1.5
                continue
            if common.detect_board(snap, size) is None:
                print("  [WARN] chessboard not detected in the captured frame -> NOT saved")
                flash_msg, flash_until = "Board not detected: NOT saved", time.time() + 1.5
                continue

            tag = f"{n_next:04d}"
            cv.imwrite(os.path.join(save_dir, f"img{tag}.png"), snap)
            with open(os.path.join(save_dir, f"posa{tag}.txt"), "w") as fp:
                fp.write(",".join(f"{v:.6f}" for v in tcp))
            saved += 1
            n_next += 1
            print(f"  [OK] img{tag}.png + posa{tag}.txt ({saved}/{target})  "
                  f"TCP={[round(v, 1) for v in tcp]}")
            flash_msg, flash_until = f"Saved {tag}", time.time() + 1.0
    except KeyboardInterrupt:
        print("\n(interrupted)")
    finally:
        print("\n[Closing] drag teach off, stopping camera and robot...")
        for name, fn in (("DragTeachSwitch(0)", lambda: robot.DragTeachSwitch(0)),
                         ("CloseRPC", robot.CloseRPC)):
            try:
                fn()
            except Exception as e:
                print(f"  ({name}: {e})")
        cam.stop()
        cv.destroyAllWindows()
    print(f"[{which}] {saved} new views saved in {save_dir}")
    return saved
