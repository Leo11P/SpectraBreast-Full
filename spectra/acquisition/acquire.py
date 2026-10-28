#!/usr/bin/env python3
# =============================================================================
#  acquire.py  —  layered acquisition with look-at viewpoints
# -----------------------------------------------------------------------------
#  GUIDING PRINCIPLE: the Fairino CONTROLLER kinematics is the ONLY source of
#  truth. No URDF, no home-made FK, no rebuilding matrices from Euler angles
#  to read the robot state.
#    - To READ the flange pose from the joints -> GetForwardKin (native).
#    - To COMMAND the robot -> MoveCart with the same [x,y,z,rx,ry,rz] format
#      the controller uses and returns.
#    - The [rx,ry,rz] angle convention is DETERMINED empirically at runtime by
#      comparing candidate matrices with GetForwardKin (see
#      detect_euler_convention), so it is not assumed a priori.
#
#  COLLISIONS: handled by collision_guard.py (URDF+FCL model, validated
#  against GetForwardKin) plus the controller's native force/torque
#  anti-collision as a reactive backstop. Still: keep speeds low and
#  supervise the robot.
# =============================================================================
import os
import sys
import math
import json
import time
import argparse

import numpy as np
import cv2 as cv
import yaml

import re
import copy
import shutil
import hashlib

ROOT = os.path.dirname(os.path.abspath(__file__))              # spectra/acquisition
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))             # SpectraBreast-Full
# SINGLE pipeline config (`acquisition:` section + `vision.aruco`).
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")
# Sibling modules (collision_guard, pipeline_viz, rerun_viz, collision_fr5,
# fairino/) are imported "flat": the script runs on the robot PC WITHOUT importing
# the `spectra` package (which would pull in torch/open3d/MASt3R).
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


# ----------------------------------------------------------------------------- 
#  Basic math utilities
# -----------------------------------------------------------------------------
def _Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], float)


def _Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], float)


def _Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], float)


# the two candidate conventions (moving/intrinsic). Which one the controller
# uses is determined by comparing against GetForwardKin.
def _euler_to_R(rx, ry, rz, order):
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    if order == "XYZ":
        return _Rx(rx) @ _Ry(ry) @ _Rz(rz)
    elif order == "ZYX":
        return _Rz(rz) @ _Ry(ry) @ _Rx(rx)
    raise ValueError(order)


def _R_to_euler(R, order):
    """Inverse of _euler_to_R for the given convention. Returns (rx,ry,rz) deg."""
    if order == "XYZ":
        # R = Rx@Ry@Rz
        ry = math.asin(max(-1.0, min(1.0, R[0, 2])))
        if abs(R[0, 2]) < 0.999999:
            rx = math.atan2(-R[1, 2], R[2, 2])
            rz = math.atan2(-R[0, 1], R[0, 0])
        else:  # gimbal lock
            rx = math.atan2(R[2, 1], R[1, 1])
            rz = 0.0
    elif order == "ZYX":
        # R = Rz@Ry@Rx
        ry = math.asin(max(-1.0, min(1.0, -R[2, 0])))
        if abs(R[2, 0]) < 0.999999:
            rz = math.atan2(R[1, 0], R[0, 0])
            rx = math.atan2(R[2, 1], R[2, 2])
        else:  # gimbal lock
            rz = 0.0
            rx = math.atan2(-R[1, 2], R[1, 1])
    else:
        raise ValueError(order)
    return math.degrees(rx), math.degrees(ry), math.degrees(rz)


# controller Euler convention, determined at runtime
_EULER = None


def pose_to_T(pose):
    """[x,y,z,rx,ry,rz] (mm, deg, _EULER convention) -> 4x4 (mm)."""
    T = np.eye(4)
    T[:3, :3] = _euler_to_R(pose[3], pose[4], pose[5], _EULER)
    T[:3, 3] = pose[:3]
    return T


def T_to_pose(T):
    """4x4 (mm) -> [x,y,z,rx,ry,rz] (mm, deg, _EULER convention)."""
    rx, ry, rz = _R_to_euler(T[:3, :3], _EULER)
    return [float(T[0, 3]), float(T[1, 3]), float(T[2, 3]), rx, ry, rz]


def _cam_pose_xyzrpy_m_rad(T):
    """4x4 (mm, base frame) -> (x, y, z in METRES, roll, pitch, yaw in RADIANS).
    RPY = XYZ fixed-axis (ROS standard): R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
    Independent of the controller Euler convention. It is exactly the format
    Vision reads from poses/pose_*.txt (convention ROLLPITCHYAW,
    translation_scale=1, radians). Used for the per-shot FLANGE pose."""
    roll, pitch, yaw = _R_to_euler(T[:3, :3], "ZYX")   # degrees, XYZ fixed-axis
    return (float(T[0, 3]) / 1000.0, float(T[1, 3]) / 1000.0, float(T[2, 3]) / 1000.0,
            math.radians(roll), math.radians(pitch), math.radians(yaw))


def look_at_R(cam_pos, target, roll_deg=0.0):
    """Camera->base rotation with the +Z optical axis pointing from cam_pos to target.
    OpenCV camera convention: +Z forward (optical), +X right, +Y down."""
    z = np.asarray(target, float) - np.asarray(cam_pos, float)
    z = z / (np.linalg.norm(z) + 1e-12)
    world_up = np.array([0, 0, 1.0])
    if abs(float(z @ world_up)) > 0.95:      # almost vertical: use another reference
        world_up = np.array([0, 1.0, 0])
    x = np.cross(world_up, z); x = x / (np.linalg.norm(x) + 1e-12)
    y = np.cross(z, x)
    R = np.column_stack([x, y, z])
    if abs(roll_deg) > 1e-9:
        R = R @ _Rz(math.radians(roll_deg))
    return R


def _pair(r):
    """(err, value) of a Fairino call that answers (err, value) on success but
    may answer a bare error code (int) on failure (e.g. after a refused move)."""
    if isinstance(r, (list, tuple)) and len(r) == 2:
        return r[0], r[1]
    return (r if isinstance(r, int) else -1), None


# -----------------------------------------------------------------------------
#  Robot: minimal wrapper around the official Fairino library (Robot.py)
# -----------------------------------------------------------------------------
class RobotIO:
    """Thin wrapper around fairino.Robot.RPC. Exposes only what is needed
    here and centralises connection, error reset and native FK reads."""

    def __init__(self, ip):
        from fairino import Robot
        print(f"[robot] connecting to {ip} ...")
        self.rpc = Robot.RPC(ip)
        Robot.RPC.is_connect = True          # known CNDE bypass (see library notes)
        self.rpc.RobotEnable(1)
        self.rpc.Mode(0)
        self._Robot = Robot

    def reset_errors(self):
        try:
            self.rpc.ResetAllError()
        except Exception as e:
            print(f"[robot] ResetAllError failed: {e}")

    def safety_code(self):
        try:
            return int(self.rpc.GetSafetyCode())
        except Exception:
            return -1

    def get_joints(self):
        err, j = _pair(self.rpc.GetActualJointPosDegree(0))
        return list(j) if err == 0 and j is not None else None

    def get_tcp_pose(self):
        r = self.rpc.GetActualTCPPose(0)
        if isinstance(r, (list, tuple)) and len(r) == 2 and r[0] == 0:
            return list(r[1])
        return None

    def get_flange_pose(self):
        """Pose of the BARE FLANGE (tool-independent). Useful for comparison."""
        r = self.rpc.GetActualToolFlangePose(0)
        if isinstance(r, (list, tuple)) and len(r) == 2 and r[0] == 0:
            return list(r[1])
        return None

    def forward_kin(self, joints):
        """NATIVE controller FK: joints(deg) -> [x,y,z,rx,ry,rz] of the flange.
        THIS is our only kinematic truth."""
        err, pose = _pair(self.rpc.GetForwardKin(list(map(float, joints))))
        return list(pose) if err == 0 and pose is not None else None

    def inverse_kin_near(self, flange, q_ref):
        """Joints (deg) reaching the flange pose with the solution NEAREST to
        q_ref (same posture), or None."""
        err, q = _pair(self.rpc.GetInverseKinRef(0, list(map(float, flange)),
                                                 list(map(float, q_ref))))
        return list(q) if err == 0 and q is not None else None

    def move_cart(self, desc_pose, tool, user, vel):
        return self.rpc.MoveCart(list(map(float, desc_pose)), int(tool), int(user),
                                 vel=float(vel))

    def move_j(self, joint_pos, tool, user, vel, desc_pose=None):
        """PTP in joint space: no Cartesian straight-line constraint, so it
        does not run into the wrist singularities that stop MoveCart/MoveL along
        the intermediate path. 'desc_pose' is informative only for the controller
        (default [0]*6 if not given, MoveJ recomputes it from the joints anyway)."""
        dp = list(map(float, desc_pose)) if desc_pose is not None else [0.0] * 6
        return self.rpc.MoveJ(list(map(float, joint_pos)), int(tool), int(user),
                              desc_pos=dp, vel=float(vel))

    def move_l(self, desc_pose, tool, user, vel):
        return self.rpc.MoveL(list(map(float, desc_pose)), int(tool), int(user),
                              vel=float(vel))

    def close(self):
        try:
            self.rpc.CloseRPC()
        except Exception:
            pass


# -----------------------------------------------------------------------------
#  Euler convention: determined EMPIRICALLY against the native FK
# -----------------------------------------------------------------------------
def detect_euler_convention(robot):
    """Compares the rotation rebuilt from the TCP pose (via GetForwardKin on the
    current joints) with the two candidate conventions, and picks the consistent one.

    Unlike the old code (which compared against a potentially WRONG URDF FK),
    here the reference is GetForwardKin, i.e. the controller itself: pose and
    angles come from the SAME source, so the comparison is valid.
    Returns (order, residual_deg, gap_deg).

    NB on gimbal lock: if the current pose has |ry|~90, both conventions may
    give the same residual -> the user is warned to redo it from a pose with |ry|<60."""
    global _EULER
    joints = robot.get_joints()
    if joints is None:
        raise SystemExit("[euler] cannot read the joints.")
    fk = robot.forward_kin(joints)
    if fk is None:
        raise SystemExit("[euler] GetForwardKin failed.")
    # the native FK ALSO returns the angles: we need the 'true' matrix. We get
    # it by assuming GetForwardKin and GetActualTCPPose use the SAME controller
    # convention. We build the candidate R from the FK angles and check the
    # self-consistency of both conventions on the SAME angles: we pick the one
    # whose R, converted back to angles, reproduces the FK angles.
    rx, ry, rz = fk[3], fk[4], fk[5]
    resid = {}
    for order in ("XYZ", "ZYX"):
        R = _euler_to_R(rx, ry, rz, order)
        back = _R_to_euler(R, order)
        # angles->R->angles round-trip error (should be ~0 for the
        # right convention away from the lock)
        d = max(abs(((a - b + 180) % 360) - 180) for a, b in zip((rx, ry, rz), back))
        resid[order] = d
    order = min(resid, key=resid.get)
    gap = abs(resid["XYZ"] - resid["ZYX"])
    near_lock = abs(abs(ry) - 90.0) < 20.0
    _EULER = order
    print(f"[euler] chosen convention = {order} "
          f"(round-trip XYZ={resid['XYZ']:.3f} ZYX={resid['ZYX']:.3f} deg)")
    if gap < 1.0 or near_lock:
        print(f"[euler] *** WARNING: weak choice (ry={ry:.1f} deg close to "
              f"gimbal lock, or indistinguishable residuals). If the pointing is "
              f"wrong, force euler_order in default.yaml[acquisition] or redo it from a pose with "
              f"|ry| < 60 deg. ***")
    return order, resid[order], gap


# -----------------------------------------------------------------------------
#  Camera intrinsics, ArUco, RealSense
# -----------------------------------------------------------------------------
def load_intrinsics(path):
    """Loads K (3x3) and D (dist coeffs) from JSON, tolerant to several schemas
    (we do not know a priori how the file was saved). Tries, in order:
      1. 'camera_matrix' (3x3, nested list) + 'dist_coeffs'/'distortion'
      2. 'K' (3x3) + 'D'/'dist'
      3. 'mtx' (3x3, typical cv.calibrateCamera output saved by hand) + 'dist'
      4. flat fields fx,fy,cx,cy (+ optional k1,k2,p1,p2,k3)
    If no schema matches, raises an error listing the keys FOUND in the file,
    so the right schema can be added instead of guessing."""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)

    def _as_K(mat):
        arr = np.array(mat, float)
        if arr.shape == (9,):
            arr = arr.reshape(3, 3)
        if arr.shape != (3, 3):
            raise ValueError(f"intrinsics matrix with unexpected shape {arr.shape}")
        return arr

    matrix_keys = ["camera_matrix", "K", "mtx", "intrinsic_matrix", "cameraMatrix"]
    dist_keys = ["dist_coeffs", "distortion", "D", "dist", "distCoeffs"]

    for mk in matrix_keys:
        if mk in d:
            K = _as_K(d[mk])
            D = None
            for dk in dist_keys:
                if dk in d:
                    D = np.array(d[dk], float).ravel()
                    break
            if D is None:
                D = np.zeros(5)
                print(f"[intrinsics] no distortion key found "
                      f"({dist_keys}): assuming D=0 (probably wrong if "
                      f"the lens has noticeable distortion).")
            print(f"[intrinsics] loaded from key '{mk}': "
                  f"fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
            return K, D

    # flat format: fx,fy,cx,cy (+ optional k1,k2,p1,p2,k3)
    if all(k in d for k in ("fx", "fy", "cx", "cy")):
        K = np.array([[d["fx"], 0, d["cx"]], [0, d["fy"], d["cy"]], [0, 0, 1]], float)
        D = np.array([d.get(k, 0.0) for k in ("k1", "k2", "p1", "p2", "k3")], float)
        print(f"[intrinsics] loaded from flat fields fx/fy/cx/cy: "
              f"fx={d['fx']:.2f} fy={d['fy']:.2f} cx={d['cx']:.2f} cy={d['cy']:.2f}")
        return K, D

    raise SystemExit(
        f"[intrinsics] No known schema found in {path}.\n"
        f"Keys in the file: {sorted(d.keys())}\n"
        f"Add the schema (3x3 matrix key and distortion key) "
        f"to load_intrinsics().")


def load_intrinsics_override(path):
    """Optional override (config `acquisition.camera.intrinsics`): a JSON with
    K and D, e.g. DATA/camera_calibration/output/intrinsics/intrinsics_opencv.json
    written by `spectra calibrate --modality intrinsics` (accepted schemas: see
    load_intrinsics). Relative paths are resolved from the cwd, like every other path."""
    p = os.path.abspath(str(path))
    if not (p.lower().endswith(".json") and os.path.isfile(p)):
        raise SystemExit(f"[intrinsics] acquisition.camera.intrinsics must be an existing "
                         f"JSON file with K and D, got {p}")
    return load_intrinsics(p)


def resolve_intrinsics(cfg, rs_cam):
    """Camera intrinsics used for solvePnP and saved into the dataset.
      default  : RealSense factory intrinsics read from the running color
                 stream (always consistent with camera.width/height);
      override : `acquisition.camera.intrinsics` set to a calibration file
                 (see load_intrinsics_override).
    The override is checked against the factory values: a focal length or
    principal point off by more than 5% almost always means a calibration made
    at a different resolution -> abort instead of acquiring with a wrong K.
    Sets cfg['_intrinsics_source'] (saved as camera_parameters/intrinsics_source.json)."""
    K_f, D_f, model = rs_cam.factory_intrinsics()
    src = {"source": "realsense_factory", "model": model,
           "width": rs_cam.w, "height": rs_cam.h}
    override = cfg["camera"].get("intrinsics")
    if not override:
        K, D = K_f, D_f
    else:
        K, D = load_intrinsics_override(override)
        d_f = max(abs(K[0, 0] / K_f[0, 0] - 1.0), abs(K[1, 1] / K_f[1, 1] - 1.0))
        d_c = max(abs(K[0, 2] - K_f[0, 2]) / rs_cam.w, abs(K[1, 2] - K_f[1, 2]) / rs_cam.h)
        if d_f > 0.05 or d_c > 0.05:
            raise SystemExit(
                f"[intrinsics] override {override} is inconsistent with the "
                f"{rs_cam.w}x{rs_cam.h} stream (focal off by {100 * d_f:.1f}%, principal "
                f"point off by {100 * d_c:.1f}% of the image): probably calibrated at a "
                "different resolution. Fix camera.width/height or the calibration.")
        src = {"source": "override", "path": os.path.abspath(str(override)),
               "width": rs_cam.w, "height": rs_cam.h}
    src["K"] = np.asarray(K, float).tolist()
    src["D"] = np.asarray(D, float).ravel().tolist()
    cfg["_intrinsics_source"] = src
    print(f"[intrinsics] {src['source']}"
          + (f" ({src['path']})" if "path" in src else f" ({model})")
          + f": fx={K[0,0]:.2f} fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}")
    return K, D


def load_T_cam2gripper(cfg):
    """T_cam2gripper 4x4 (mm). It links the camera to the FLANGE (controller
    flange, tool 0), like the hand-eye calibration. `hand_eye.T_cam2gripper`:
    null -> output of `spectra calibrate` (camera_calibration/output/
    calibration/cam2gripper.txt under data_root); a 4x4 list -> used as is."""
    T = (cfg.get("hand_eye") or {}).get("T_cam2gripper")
    if T is None:
        path = os.path.join(cfg["_calib_dir"], "output", "calibration", "cam2gripper.txt")
        if not os.path.isfile(path):
            raise SystemExit(f"[hand-eye] {path} not found: run `spectra calibrate` (or "
                             "--modality calibration calibrate), or set "
                             "acquisition.hand_eye.T_cam2gripper to a 4x4 matrix.")
        T, src = np.loadtxt(path), path
    else:
        src = "acquisition.hand_eye.T_cam2gripper"
    T = np.array(T, float)
    if T.shape != (4, 4):
        raise SystemExit(f"[hand-eye] T_cam2gripper must be 4x4 ({src}).")
    print(f"[hand-eye] {src}: t(mm) = {np.round(T[:3,3],2).tolist()}")
    return T


def make_detector(dict_name):
    d = getattr(cv.aruco, dict_name)
    ad = cv.aruco.getPredefinedDictionary(d)
    params = cv.aruco.DetectorParameters()
    return cv.aruco.ArucoDetector(ad, params)


def detect_markers(detector, gray):
    corners, ids, _ = detector.detectMarkers(gray)
    return corners, ids


def _marker_object_points(L):
    h = L / 2.0
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], float)


def estimate_marker_pose(corner, K, D, L):
    objp = _marker_object_points(L)
    ok, rvec, tvec = cv.solvePnP(objp, corner.reshape(-1, 2), K, D,
                                 flags=cv.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    R, _ = cv.Rodrigues(rvec)
    return R, tvec.reshape(3)


class RS:
    """Minimal RealSense wrapper (color stream + average of N frames). If the device
    supports it, depth is ALSO enabled, aligned to color (same intrinsics as
    color -> consistent with intrinsics.npy): grab_depth() returns the depth map."""

    def __init__(self, cfg):
        import pyrealsense2 as rs
        self.rs = rs
        self.w = int(cfg["camera"]["width"]); self.h = int(cfg["camera"]["height"])
        fps = int(cfg["camera"].get("fps", 15))
        # optional depth (default ON): it must NEVER prevent color acquisition.
        self.depth_enabled = bool(cfg["camera"].get("depth", True))
        self.depth_scale = 1.0
        self.align = None

        def _start(with_depth):
            pipe = rs.pipeline()
            conf = rs.config()
            conf.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, fps)
            if with_depth:
                # no forced depth resolution: many cameras (e.g. D45x) do NOT
                # support depth at the color resolution (e.g. 1280x800). Let the
                # SDK pick a valid depth mode; alignment to color brings the
                # map back to the color resolution/intrinsics anyway.
                conf.enable_stream(rs.stream.depth)
            prof = pipe.start(conf)
            return pipe, prof

        self.pipe = None
        if self.depth_enabled:
            try:
                self.pipe, self.profile = _start(True)
                self.depth_scale = float(self.profile.get_device()
                                         .first_depth_sensor().get_depth_scale())
                self.align = rs.align(rs.stream.color)
                print(f"[realsense] depth ON (scale {self.depth_scale:.6f} m/unit, "
                      "aligned to color)")
            except Exception as e:
                print(f"[realsense] warning: depth unavailable ({e}); color only")
                self.depth_enabled = False
                if self.pipe is not None:
                    try: self.pipe.stop()
                    except Exception: pass
                self.pipe = None
        if self.pipe is None:
            self.pipe, self.profile = _start(False)
        # warm-up
        for _ in range(30):
            self.pipe.wait_for_frames(10000)

    def grab(self, n_avg=5):
        acc = None; got = 0
        for _ in range(max(1, n_avg)):
            try:
                frames = self.pipe.wait_for_frames(10000)
            except RuntimeError:
                continue
            c = frames.get_color_frame()
            if not c:
                continue
            img = np.asanyarray(c.get_data()).astype(np.float32)
            acc = img if acc is None else acc + img
            got += 1
        if got == 0:
            return None
        return (acc / got).astype(np.uint8)

    def grab_depth(self):
        """Depth map co-registered with color (same resolution/intrinsics).
        Returns a uint16 in MILLIMETRES (0 = no measurement/hole) or None if
        depth is unavailable. Single frameset: at shot time the robot is
        still, so no averaging (averaging depth with the zeros of the holes
        would corrupt it)."""
        if not self.depth_enabled or self.align is None:
            return None
        try:
            frames = self.pipe.wait_for_frames(10000)
        except RuntimeError:
            return None
        try:
            aligned = self.align.process(frames)
            d = aligned.get_depth_frame()
        except Exception:
            return None
        if not d:
            return None
        raw = np.asanyarray(d.get_data())                 # uint16, native units
        depth_mm = raw.astype(np.float32) * self.depth_scale * 1000.0
        return np.clip(depth_mm, 0, 65535).astype(np.uint16)

    def factory_intrinsics(self):
        """Factory intrinsics of the COLOR stream actually running (same
        resolution as the images). Returns (K 3x3, D (5,), model name)."""
        vsp = self.profile.get_stream(self.rs.stream.color).as_video_stream_profile()
        i = vsp.get_intrinsics()
        K = np.array([[i.fx, 0.0, i.ppx], [0.0, i.fy, i.ppy], [0.0, 0.0, 1.0]], float)
        D = np.asarray(list(i.coeffs)[:5], float)
        return K, D, str(i.model)

    def stop(self):
        try:
            self.pipe.stop()
        except Exception:
            pass


# -----------------------------------------------------------------------------
#  Transformation chain — EVERYTHING anchored to the controller's native FK
# -----------------------------------------------------------------------------
def camera_T_base(robot, T_cam2gripper):
    """T_cam2base (4x4, mm) using the controller's NATIVE FK for the flange.
        T_cam2base = T_flange2base(GetForwardKin) @ T_cam2gripper
    Returns (T_cam2base, joints) or (None, None)."""
    joints = robot.get_joints()
    if joints is None:
        return None, None
    fk = robot.forward_kin(joints)          # [x,y,z,rx,ry,rz] flange in base
    if fk is None:
        return None, None
    T_f2b = pose_to_T(fk)                    # uses the detected _EULER convention
    return T_f2b @ T_cam2gripper, joints


def cam_points_to_base(pts_cam_mm, T_cam2base):
    P = np.asarray(pts_cam_mm, float).reshape(-1, 3)
    Ph = np.hstack([P, np.ones((len(P), 1))])
    return (T_cam2base @ Ph.T).T[:, :3]


def base_points_to_cam(pts_base_mm, T_cam2base):
    P = np.asarray(pts_base_mm, float).reshape(-1, 3)
    Ph = np.hstack([P, np.ones((len(P), 1))])
    return (np.linalg.inv(T_cam2base) @ Ph.T).T[:, :3]


def gripper_pose_from_camera_T(T_cam2base, T_cam2gripper):
    """Given the desired camera->base pose, derives the FLANGE pose to command:
        T_flange2base = T_cam2base @ inv(T_cam2gripper)
    and converts it to [x,y,z,rx,ry,rz] for MoveCart (tool 0 = flange)."""
    T_f2b = T_cam2base @ np.linalg.inv(T_cam2gripper)
    return T_to_pose(T_f2b)


# -----------------------------------------------------------------------------
#  CHAIN CHECK (--verify): does not move the robot. Proves the chain is
#  correct BEFORE generating trajectories.
# -----------------------------------------------------------------------------
def run_verify(cfg, robot, T_c2g, K, D, rs):
    print("\n" + "=" * 64)
    print("  TRANSFORMATION CHAIN CHECK (no motion)")
    print("=" * 64)

    # 1) sanity: GetForwardKin(current joints) == GetActualTCPPose ? (tool 0)
    joints = robot.get_joints()
    fk = robot.forward_kin(joints)
    tcp = robot.get_tcp_pose()
    fl = robot.get_flange_pose()
    print(f"[verify] joints (deg)          = {np.round(joints,2).tolist()}")
    print(f"[verify] GetForwardKin  flange = {np.round(fk,2).tolist()}")
    print(f"[verify] GetActualTCPPose      = {np.round(tcp,2).tolist()}")
    print(f"[verify] GetActualToolFlange   = {np.round(fl,2).tolist() if fl else None}")
    dpos = np.linalg.norm(np.array(fk[:3]) - np.array(tcp[:3]))
    print(f"[verify] |FK - TCP| position  = {dpos:.2f} mm  "
          f"(expected ~0 with tool 0; if large, a TCP tool is active on the controller)")

    # 2) Euler round-trip: pose_to_T(fk) -> T_to_pose must give back fk
    T = pose_to_T(fk)
    back = T_to_pose(T)
    drt = max(abs(a - b) for a, b in zip(fk[:3], back[:3]))
    print(f"[verify] round-trip pose->T->pose (pos) err = {drt:.4f} mm "
          f"(convention {_EULER})")

    # 3) markers: centroid estimate in base and REPROJECTION into the current image.
    L = float(cfg["aruco"]["marker_length_mm"])
    cx, cy = K[0, 2], K[1, 2]
    detector = make_detector(cfg["aruco"]["dictionary"])
    img = rs.grab(int(cfg["camera"].get("n_avg_frames", 5)))
    if img is None:
        print("[verify] no RealSense frame."); return
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners, ids = detect_markers(detector, gray)
    if ids is None or len(corners) == 0:
        print("[verify] no marker in view: centre the camera on the markers.")
        return
    T_c2b, _ = camera_T_base(robot, T_c2g)
    centers_px, centers_cam, centers_base, dists = [], [], [], []
    for mc in corners:
        pose = estimate_marker_pose(mc, K, D, L)
        if pose is None:
            continue
        centers_px.append(mc.reshape(-1, 2).mean(axis=0))
        centers_cam.append(pose[1])
        centers_base.append(cam_points_to_base(pose[1], T_c2b)[0])
        dists.append(float(np.linalg.norm(pose[1])))
    centroid_px = np.mean(centers_px, axis=0)
    centroid_cam = np.mean(centers_cam, axis=0)
    centroid_base = np.mean(centers_base, axis=0)
    off = math.hypot(centroid_px[0] - cx, centroid_px[1] - cy)
    print(f"[verify] markers: {len(centers_base)}  distances(mm)={np.round(dists,1).tolist()}")
    print(f"[verify] ACTUAL centroid in pixels = ({centroid_px[0]:.0f},{centroid_px[1]:.0f}); "
          f"offset from centre = {off:.0f} px")
    print(f"[verify] 3D centroid in base (mm) = {np.round(centroid_base,1).tolist()}")

    # reproject the 3D centroid (camera frame) to pixels: round-trip
    uv, _ = cv.projectPoints(centroid_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    up = uv.reshape(2)
    err_px = math.hypot(up[0] - centroid_px[0], up[1] - centroid_px[1])
    print(f"[verify] round-trip pixel->3D->pixel = {err_px:.1f} px (expected <5)")

    # 4) CROSS-CHECK: if I commanded the camera to point at the centroid from the
    # CURRENT position, which flange pose would I get? Compare it with the current
    # flange: the position DIFFERENCE must be ~0 (same camera position),
    # and the predicted optical axis must point at the centroid.
    cam_pos_now = T_c2b[:3, 3]
    R_look = look_at_R(cam_pos_now, centroid_base, 0.0)
    T_look = np.eye(4); T_look[:3, :3] = R_look; T_look[:3, 3] = cam_pos_now
    flange_cmd = gripper_pose_from_camera_T(T_look, T_c2g)
    print("[verify] flange to command to point at the centroid (same camera pos):")
    print(f"         {np.round(flange_cmd,2).tolist()}")
    print(f"[verify] CURRENT flange (FK)      = {np.round(fk,2).tolist()}")
    dpos_cmd = np.linalg.norm(np.array(flange_cmd[:3]) - np.array(fk[:3]))
    print(f"[verify] |cmd - current| position = {dpos_cmd:.1f} mm "
          f"(the camera position does not change, so the flange should stay "
          f"close; a large difference means a hand-eye/convention inconsistency)")
    print("=" * 64)
    print("[verify] INTERPRETATION:")
    print("  - |FK-TCP| ~0 and Euler round-trip ~0  -> reading chain OK")
    print("  - small pixel round-trip (<5px)        -> intrinsics + solvePnP OK")
    print("  - if the px offset is large BUT the 3D centroid in base falls where you")
    print("    physically expect the dish, the camera is just off-centre")
    print("    (to be corrected by the servo), there is no chain error.")
    print("  - if the 3D centroid in base does NOT fall on the real dish: chain")
    print("    error (hand-eye or convention) -> do not generate trajectories.")


# -----------------------------------------------------------------------------
#  Bootstrap: sample centroid + normal estimate (median over several frames)
# -----------------------------------------------------------------------------
def bootstrap(robot, rs, cfg, T_c2g, K, D):
    L = float(cfg["aruco"]["marker_length_mm"])
    n_frames = int(cfg["target"].get("n_bootstrap_frames", 8))
    detector = make_detector(cfg["aruco"]["dictionary"])

    T_c2b, joints = camera_T_base(robot, T_c2g)
    if T_c2b is None:
        raise SystemExit("[bootstrap] native FK not readable.")
    print(f"[bootstrap] joints (deg) = {np.round(joints,1).tolist()}")
    print(f"[bootstrap] camera optical axis in base (+Z) = {np.round(T_c2b[:3,2],3).tolist()} "
          f"(expected ~[0,0,-1] if looking down)")
    print(f"[bootstrap] camera position in base (mm) = {np.round(T_c2b[:3,3],1).tolist()}")

    per_id_c, per_id_n, per_id_corn = {}, {}, {}
    seen = 0
    for _ in range(n_frames):
        img = rs.grab(2)
        if img is None:
            continue
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        corners, ids = detect_markers(detector, gray)
        if ids is None or len(corners) == 0:
            continue
        seen += 1
        for mc, mid in zip(corners, ids.ravel().tolist()):
            pose = estimate_marker_pose(mc, K, D, L)
            if pose is None:
                continue
            R, t = pose
            per_id_c.setdefault(mid, []).append(t)
            per_id_n.setdefault(mid, []).append(R @ np.array([0, 0, 1.0]))
            objp = _marker_object_points(L)
            per_id_corn.setdefault(mid, []).append((R @ objp.T + t.reshape(3, 1)).T)

    if seen == 0 or not per_id_c:
        raise SystemExit(f"[bootstrap] no marker seen in {n_frames} frames.")

    centers_cam, normals_cam, corners_cam, disp = [], [], [], []
    for mid in sorted(per_id_c):
        arr = np.array(per_id_c[mid])
        med = np.median(arr, axis=0)
        centers_cam.append(med)
        disp.append(float(np.linalg.norm(arr - med, axis=1).mean()))
        normals_cam.append(np.median(np.array(per_id_n[mid]), axis=0))
        corners_cam.append(np.median(np.array(per_id_corn[mid]), axis=0))

    centers_cam = np.array(centers_cam)
    centers_base = cam_points_to_base(centers_cam, T_c2b)
    normals_base = (T_c2b[:3, :3] @ np.array(normals_cam).T).T
    corners_base = np.vstack([cam_points_to_base(c, T_c2b) for c in corners_cam])
    centroid = centers_base.mean(axis=0)
    cam_pos = T_c2b[:3, 3]

    up = np.mean(normals_base, axis=0)
    if np.linalg.norm(up) < 1e-6:
        up = np.array([0, 0, 1.0])
    up = up / np.linalg.norm(up)
    if np.dot(up, cam_pos - centroid) < 0:
        up = -up

    print(f"[bootstrap] markers: {len(centers_base)} (in {seen} frames)  "
          f"distances(mm)={np.round(np.linalg.norm(centers_cam,axis=1),1).tolist()}")
    print(f"[bootstrap] per-marker spread (mm) = {np.round(disp,2).tolist()}")
    print(f"[bootstrap] base centroid (mm) = {np.round(centroid,1).tolist()}")
    print(f"[bootstrap] up = {np.round(up,3).tolist()}")
    # base position of every marker seen here: the recentring aims at the FIXED
    # centroid and uses these to measure the chain error at each pose.
    centers_by_id = {int(mid): centers_base[k] for k, mid in enumerate(sorted(per_id_c))}
    return {"centroid": centroid, "up": up, "corners_base": corners_base,
            "cam_pos": cam_pos, "centers_by_id": centers_by_id}


def _orthonormal_basis(up):
    up = up / (np.linalg.norm(up) + 1e-12)
    ref = np.array([1.0, 0, 0]) if abs(up[0]) < 0.9 else np.array([0, 1.0, 0])
    u = np.cross(ref, up); u = u / (np.linalg.norm(u) + 1e-12)
    v = np.cross(up, u)
    return u, v


# -----------------------------------------------------------------------------
#  Look-at pose generation (NO collisions: only visibility + gimbal + IK)
# -----------------------------------------------------------------------------
def _surface_coverage(surf_pts_base, T_c2b, K, D, W, H, margin):
    """Fraction [0..1] of the surface points (marker corners, from the bootstrap)
    that project INSIDE the image (with margin) and are IN FRONT of the camera. It
    is the geometrically correct criterion for "the markers are in view": it scales
    by itself with distance (far -> wide FOV -> they cover everything; close ->
    narrow FOV -> some fall out), unlike the centroid-only check (the centroid can
    be centred while the surrounding markers are out of frame)."""
    P = np.asarray(surf_pts_base, float).reshape(-1, 3)
    if len(P) == 0:
        return 1.0
    pts_cam = base_points_to_cam(P, T_c2b)          # base -> camera
    vis = 0
    for p in pts_cam:
        if p[2] <= 1.0:                              # behind the camera
            continue
        uv, _ = cv.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
        uu, vv = uv.reshape(2)
        if margin <= uu <= W - margin and margin <= vv <= H - margin:
            vis += 1
    return vis / len(P)


def _evaluate_pose(centroid, up, u, v, az, tilt, dist, roll,
                   robot, T_c2g, K, D, W, H, margin, gimbal_margin,
                   surf_pts=None, min_cov=0.0, q_ref=None, ik_config=-1):
    """Evaluates ONE parametric look-at pose. Returns (pose_dict, None) if valid,
    otherwise (None, reason) with reason in {'vis','cov','gimbal','ik'}.
    az/tilt in radians, dist in mm, roll in degrees. IK: the solution nearest
    to q_ref if given (GetInverseKinRef), else configuration ik_config
    (-1 = nearest to the robot's current joints, 0..7 = branch)."""
    direction = (math.cos(tilt) * up +
                 math.sin(tilt) * (math.cos(az) * u + math.sin(az) * v))
    cam_pos = centroid + dist * direction
    R = look_at_R(cam_pos, centroid, roll)
    T_c2b = np.eye(4); T_c2b[:3, :3] = R; T_c2b[:3, 3] = cam_pos
    # (a) centroid inside the image
    p_cam = base_points_to_cam(centroid, T_c2b)[0]
    if p_cam[2] <= 1.0:
        return None, "vis"
    uv, _ = cv.projectPoints(p_cam.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    uu, vv = uv.reshape(2)
    if not (margin <= uu <= W - margin and margin <= vv <= H - margin):
        return None, "vis"
    # (a2) MARKER COVERAGE: a minimum fraction of the marker corners must stay
    # in frame (a centred centroid is not enough). Filters out poses too close/
    # oblique where the markers leave the FOV -> at runtime you would see nothing.
    if surf_pts is not None and min_cov > 0.0:
        cov = _surface_coverage(surf_pts, T_c2b, K, D, W, H, margin)
        if cov < min_cov:
            return None, "cov"
    # (b) flange pose to command
    flange = gripper_pose_from_camera_T(T_c2b, T_c2g)
    # (c) avoid gimbal lock in |ry|
    if abs(abs(flange[4]) - 90.0) < gimbal_margin:
        return None, "gimbal"
    # (d) reachability: the native IK must have a solution, and we STORE it:
    # it is used for MoveJ (joint space), which imposes no Cartesian straight line
    # and so does not run into the wrist singularities that stop MoveCart/MoveL.
    if q_ref is not None:
        err, joints_sol = _pair(robot.rpc.GetInverseKinRef(0, list(map(float, flange)),
                                                           list(map(float, q_ref))))
    else:
        err, joints_sol = _pair(robot.rpc.GetInverseKin(0, list(map(float, flange)),
                                                        int(ik_config)))
    if err != 0 or joints_sol is None:
        return None, "ik"
    return {"flange": flange, "joints": list(map(float, joints_sol)),
            "cam_T_base": T_c2b.copy(), "view": [az, tilt, dist, roll]}, None


def _ik_retry_candidates(az, tilt, dist, roll, r):
    """Perturbations 'close' to a pose whose IK failed, sorted from the most
    similar to the most different. First only the roll changes (camera position
    and optical axis UNCHANGED: only the rotation about the optical axis changes ->
    same framing, but it often moves the wrist out of a singularity/failed IK);
    then small azimuth/tilt/distance nudges; finally combinations.
    az/tilt in radians, dist in mm, roll in degrees."""
    roll_step = float(r.get("roll_step_deg", 15.0))
    az_step = math.radians(float(r.get("az_step_deg", 4.0)))
    tilt_step = math.radians(float(r.get("tilt_step_deg", 4.0)))
    dfrac = float(r.get("dist_frac", 0.06))
    max_att = int(r.get("max_attempts", 24))

    cands = []
    # Phase A: roll sweep (same position + pointing) — the most similar
    for k in range(1, 7):
        for s in (+1, -1):
            cands.append((az, tilt, dist, roll + s * k * roll_step))
    # Phase B: small azimuth / tilt / distance nudges
    for daz in (az_step, -az_step, 2 * az_step, -2 * az_step):
        cands.append((az + daz, tilt, dist, roll))
    for dt in (tilt_step, -tilt_step, 2 * tilt_step, -2 * tilt_step):
        cands.append((az, tilt + dt, dist, roll))
    for f in (1 - dfrac, 1 + dfrac, 1 - 2 * dfrac, 1 + 2 * dfrac):
        cands.append((az, tilt, dist * f, roll))
    # Phase C: azimuth + roll combinations
    for daz in (az_step, -az_step):
        for s in (+1, -1):
            cands.append((az + daz, tilt, dist, roll + s * roll_step))
    return cands[:max_att]


def _planning_ctx(target, cfg, robot, T_c2g, K, D):
    """Evaluator of the look-at poses around the target (visibility, gimbal,
    IK), shared by generate_poses and plan_sequence:
    ev(az, tilt, dist, roll, q_ref=None, ik_config=-1) -> (pose|None, reason)."""
    centroid = target["centroid"]; up = target["up"]
    u, v = _orthonormal_basis(up)
    W = int(cfg["camera"]["width"]); H = int(cfg["camera"]["height"])
    vis = cfg.get("visibility", {})
    base_margin = float(vis.get("margin_px", 40))
    gimbal_margin = float(vis.get("gimbal_lock_margin_deg", 12))
    min_cov = float(vis.get("min_marker_coverage", 0.0))   # 0 = coverage check OFF
    # margin proportional to distance: CLOSE UP the object is large in the FOV and
    # it takes little to cross the border -> LARGER margin (more central
    # centroid); FAR AWAY the object is small -> smaller margin. Formula:
    # margin = base_margin * (ref_dist / dist), clamped to [min, max]. It is a
    # check on the CENTROID ONLY, not on the markers.
    margin_prop = bool(vis.get("margin_px_proportional", False))
    margin_ref_d = float(vis.get("margin_ref_distance_mm", 150.0))
    margin_min = float(vis.get("margin_px_min", 20.0))
    margin_max = float(vis.get("margin_px_max", 200.0))

    def margin_of(dist):
        if not margin_prop:
            return base_margin
        m = base_margin * margin_ref_d / max(1.0, float(dist))
        return float(min(margin_max, max(margin_min, m)))

    # marker corners estimated by the bootstrap: used ONLY if the coverage check
    # is active (min_marker_coverage > 0). OFF by default: the centroid is used.
    surf_pts = target.get("corners_base")
    if surf_pts is not None and len(np.asarray(surf_pts)) == 0:
        surf_pts = None

    def ev(az, tilt, dist, roll, q_ref=None, ik_config=-1):
        return _evaluate_pose(centroid, up, u, v, az, tilt, dist, roll,
                              robot, T_c2g, K, D, W, H, margin_of(dist), gimbal_margin,
                              surf_pts=surf_pts, min_cov=min_cov, q_ref=q_ref,
                              ik_config=ik_config)
    return ev


def generate_poses(target, cfg, robot, T_c2g, K, D):
    _eval = _planning_ctx(target, cfg, robot, T_c2g, K, D)
    min_cov = float(cfg.get("visibility", {}).get("min_marker_coverage", 0.0))
    ik_retry = cfg.get("ik_retry", {}) or {}
    retry_on = bool(ik_retry.get("enable", True))

    out = []
    skip = {"vis": 0, "cov": 0, "gimbal": 0, "ik": 0}
    recovered = 0
    # global default for depth saving (camera.depth): a layer can
    # override it with 'depth:' in its block (True/False) -> decided per layer.
    global_depth = bool(cfg.get("camera", {}).get("depth", True))
    for layer in cfg["layers"]:
        name = layer.get("name", "layer")
        depth_flag = bool(layer.get("depth", global_depth))
        dist = float(layer["distance_mm"]); tilt = math.radians(float(layer["tilt_deg"]))
        n = int(layer["n_poses"]); roll = float(layer.get("roll_deg", 0))
        span = math.radians(float(layer.get("azimuth_span_deg", 360)))
        start = math.radians(float(layer.get("azimuth_start_deg", 0)))
        full = abs(span - 2 * math.pi) < 1e-6
        denom = n if full else max(1, n - 1)
        kept = 0
        for i in range(n):
            az = start + span * (i / denom)
            pose, reason = _eval(az, tilt, dist, roll)
            az_used = az
            # RETRY on failed IK: look for a NEARBY pose (roll first) that
            # has an IK solution, instead of losing this azimuth entirely.
            if pose is None and reason == "ik" and retry_on:
                for (az2, tilt2, dist2, roll2) in _ik_retry_candidates(az, tilt, dist, roll, ik_retry):
                    pose2, reason2 = _eval(az2, tilt2, dist2, roll2)
                    if pose2 is not None:
                        pose, az_used = pose2, az2
                        recovered += 1
                        break
            if pose is None:
                skip[reason] += 1; continue
            pose.update({"layer": name, "azim_deg": round(math.degrees(az_used), 1),
                         "depth": depth_flag})
            out.append(pose)
            kept += 1
        print(f"[layer] {name:>14}: {kept}/{n} valid poses "
              f"(dist {dist:.0f}mm tilt {math.degrees(tilt):.0f}deg)")
    if any(skip.values()) or recovered:
        print(f"[layer] discarded: out-of-frame={skip['vis']} "
              f"insufficient-marker-coverage={skip['cov']} "
              f"gimbal-lock={skip['gimbal']} IK-failed={skip['ik']}"
              + (f"  | recovered by IK retry={recovered}" if recovered else ""))
        if skip["cov"]:
            print(f"[layer] NB: {skip['cov']} poses discarded because at that "
                  f"distance/tilt less than {min_cov:.0%} of the markers stay in the FOV "
                  "(the markers leave the frame). Move the layer further away "
                  "(distance_mm) or lower 'visibility.min_marker_coverage'.")
    return out


# -----------------------------------------------------------------------------
#  Closed-loop recentring (position+orientation, damping, min-marker)
# -----------------------------------------------------------------------------
def _project_px(P_base, T_c2b, K, D):
    """Pixel of a base point seen from the camera pose T_c2b, or None if behind it."""
    p = base_points_to_cam(P_base, T_c2b)[0]
    if p[2] <= 1.0:
        return None
    uv, _ = cv.projectPoints(p.reshape(1, 3), np.zeros(3), np.zeros(3), K, D)
    return uv.reshape(2)


def _rot_towards(a, b, frac=1.0):
    """frac x the smallest rotation taking direction a onto direction b."""
    a = np.asarray(a, float) / np.linalg.norm(a)
    b = np.asarray(b, float) / np.linalg.norm(b)
    axis = np.cross(a, b)
    s = float(np.linalg.norm(axis))
    if s < 1e-12:
        return np.eye(3)
    ang = math.atan2(s, float(a @ b)) * frac
    k = axis / s
    Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(ang) * Kx + (1 - math.cos(ang)) * (Kx @ Kx)


def recenter(robot, rs, cfg, T_c2g, K, D, target, guard=None):
    """Brings the FIXED target point of the bootstrap (target['centroid'], base
    frame) to the image centre. Where that point really is in the image is
    predicted with the native FK and corrected by the bootstrap markers in view
    (median offset between where they are detected and where the FK puts them:
    the chain error at this pose). Unlike the centroid of the markers in view,
    the aim does not move when markers appear or disappear (e.g. hidden by the
    phantom). The correction rotates the camera about its centre (same
    position, smallest rotation: the roll chosen by the planner is kept), by
    'damping' x the needed angle. Returns True if at least one frame saw the
    markers (meaningful photo), False otherwise. If 'guard' is given, every
    correction is collision-checked along its joint path before moving."""
    sv = cfg.get("servo", {})
    if not sv.get("enable", True):
        return True
    tol = float(sv.get("pixel_tol", 8)); max_iters = int(sv.get("max_iters", 4))
    vel = float(sv.get("vel", 12)); min_m = int(sv.get("min_markers", 2))
    damping = float(sv.get("damping", 0.6))
    max_step = float(sv.get("max_joint_step_deg", 20.0))   # larger = posture flip: skip
    # detection robustness: the FIRST frame right after MoveJ can be
    # spoiled by vibration/exposure not yet settled -> 0 markers even
    # if the markers are perfectly visible in the stream. Settle a few frames
    # BEFORE detecting, and retry detection on fresh frames before
    # concluding "no markers".
    settle_s = float(sv.get("settle_before_detect_s", 0.4))
    detect_retries = int(sv.get("detect_retries", 4))
    n_avg = int(cfg.get("camera", {}).get("n_avg_frames", 5))
    cx, cy = K[0, 2], K[1, 2]
    detector = make_detector(cfg["aruco"]["dictionary"])
    tool = int(cfg["robot"]["tool"]); user = int(cfg["robot"]["user"])
    aim3d = np.asarray(target["centroid"], float)
    known = target.get("centers_by_id") or {}

    def _grab_detect():
        """Grab+detect ArUco, retrying on transient frames without markers.
        Returns (corners, ids, img|None)."""
        last = None
        for _ in range(max(1, detect_retries)):
            img = rs.grab(max(2, n_avg))
            if img is None:
                continue
            last = img
            c, i = detect_markers(detector, cv.cvtColor(img, cv.COLOR_BGR2GRAY))
            if i is not None and len(c) > 0:
                return c, i, img
        return None, None, last

    # one-off settling before the first detection (post-MoveJ)
    if settle_s > 0:
        t0 = time.time()
        while time.time() - t0 < settle_s:
            try: rs.pipe.wait_for_frames(10000)
            except Exception: pass

    for it in range(max_iters):
        corners, ids, img = _grab_detect()
        if img is None:
            print("  [servo] no frame"); return False
        if ids is None or len(corners) == 0:
            print(f"  [servo] it{it}: no marker in view "
                  f"({detect_retries} attempts)"); return False
        T_c2b, _ = camera_T_base(robot, T_c2g)
        if T_c2b is None:
            print("  [servo] FK not readable"); return True
        res = []
        for mc, mid in zip(corners, np.asarray(ids).ravel().tolist()):
            if int(mid) not in known:
                continue
            pred = _project_px(known[int(mid)], T_c2b, K, D)
            if pred is not None:
                res.append(mc.reshape(-1, 2).mean(axis=0) - pred)
        if len(res) < min_m:
            print(f"  [servo] it{it}: {len(res)} bootstrap markers in view (<{min_m}), "
                  "not correcting")
            return True
        aim = _project_px(aim3d, T_c2b, K, D)
        if aim is None:
            print("  [servo] target point behind the camera: not correcting"); return True
        aim = aim + np.median(np.asarray(res), axis=0)
        off = math.hypot(aim[0] - cx, aim[1] - cy)
        if off <= tol:
            print(f"  [servo] centred ({off:.0f}px) in {it} iter"); return True
        nrm = cv.undistortPoints(aim.reshape(1, 1, 2).astype(np.float64), K, D).reshape(2)
        ray = T_c2b[:3, :3] @ np.array([nrm[0], nrm[1], 1.0])
        T_new = T_c2b.copy()
        T_new[:3, :3] = _rot_towards(T_c2b[:3, 2], ray, damping) @ T_c2b[:3, :3]
        flange = gripper_pose_from_camera_T(T_new, T_c2g)
        # The correction is a joint-space move (MoveJ) to the IK solution
        # NEAREST to the current joints: a Cartesian MoveL is refused by the
        # controller when the target needs another joint configuration (e.g.
        # from the "alternative IK" postures). Small correction -> small joint
        # change; a big one means a posture flip: not corrected.
        q_now = robot.get_joints()
        q_new = robot.inverse_kin_near(flange, q_now) if q_now is not None else None
        if q_new is None:
            print("  [servo] no IK for the correction: not correcting"); return True
        jump = max(abs(a - b) for a, b in zip(q_new, q_now))
        if jump > max_step:
            print(f"  [servo] correction needs {jump:.0f} deg on a joint (> {max_step:.0f}): "
                  "not correcting"); return True
        # collision-check the correction BEFORE moving, along the joint path
        # (no contact with the sample, not even during the recentring moves).
        if guard is not None and guard.available and guard.validated:
            ok_c, reasons = guard.path_safe(q_now, q_new)
            if not ok_c:
                print(f"  [servo] correction ABORTED (collision): {'; '.join(reasons)}")
                return True
        print(f"  [servo] it{it}: off {off:.0f}px ({len(res)} markers) -> correcting "
              f"(max joint change {jump:.1f} deg)")
        err = robot.move_j(q_new, tool, user, vel, desc_pose=flange)
        if err != 0:
            print(f"  [servo] MoveJ failed (code {err})"); return True
    print("  [servo] max_iters reached"); return True


# -----------------------------------------------------------------------------
#  Execution
# -----------------------------------------------------------------------------
def _reorder_poses_min_travel(poses, q_start):
    """Reorders the poses WITHIN each layer to minimise joint-space travel,
    WITHOUT changing the poses (only their execution order).

    Each layer is an azimuth ring: adjacent azimuths give (usually) adjacent
    joints, except where the wrist 'flips' (large joint jump = backtrack).
    We keep the ring order but 'cut' the ring at the arc with the LARGEST
    joint jump: the resulting open path covers the N-1 small arcs and NOT the
    large one. Between the two directions (same cost) we pick the one that
    STARTS from the pose closest (in joints) to the entry point -> minimum
    travel on entry too.
    Cost = max per-joint difference (L-inf, ~ MoveJ duration)."""
    from collections import OrderedDict
    if not poses:
        return poses
    groups = OrderedDict()
    for wp in poses:
        groups.setdefault(wp["layer"], []).append(wp)

    def jdist(a, b):
        return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))

    out = []
    prev = list(map(float, q_start)) if q_start is not None else None
    for layer, group in groups.items():
        n = len(group)
        if n <= 1:
            seq = group
        else:
            J = [wp["joints"] for wp in group]
            edges = [jdist(J[i], J[(i + 1) % n]) for i in range(n)]   # cyclic arcs
            cut = int(np.argmax(edges))                 # drop the largest jump
            fwd = [(cut + 1 + k) % n for k in range(n)]  # open path that avoids 'cut'
            rev = list(reversed(fwd))
            if prev is not None and jdist(J[rev[0]], prev) < jdist(J[fwd[0]], prev):
                order = rev
            else:
                order = fwd
            seq = [group[i] for i in order]
            print(f"[traj] {layer}: optimised order (cut at the max jump "
                  f"{edges[cut]:.0f}deg between 2 adjacent azimuths)")
        out.extend(seq)
        prev = seq[-1]["joints"]
    return out


def _jdist(a, b):
    return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))


#: Joint weights of the planning cost: moving shoulder/elbow (J1-J3) swings the
#: whole arm, the wrist (J4-J6) only the camera: 1 deg of J1-J3 = 3 deg of wrist.
_PLAN_W = np.array([3.0, 3.0, 3.0, 1.0, 1.0, 1.0])


def _move_cost(a, b):
    """Weighted joint distance used to choose between IK solutions."""
    return float(np.max(_PLAN_W * np.abs(np.asarray(a, float) - np.asarray(b, float))))


def _config_ok(guard, q):
    if guard is None or not (guard.available and guard.validated):
        return True
    return guard.config_safe(q)[0]


def _path_ok(guard, q0, q1):
    if guard is None or not (guard.available and guard.validated):
        return True
    return guard.path_safe(q0, q1)[0]


def _via_configs(target, cfg, robot, T_c2g, guard, q_home):
    """Safe joint configurations with the camera HIGH above the sample, to change
    arm configuration (e.g. a wrist flip) between two poses that cannot be
    joined directly: the camera looks at the centroid from
    planning.via_heights_mm (straight down and tilted 15 deg on 4 sides), 4
    rolls, every IK branch (config 0..7); plus the start pose."""
    pl = cfg.get("planning", {}) or {}
    heights = [float(h) for h in pl.get("via_heights_mm", [220, 280])]
    gimbal_margin = float(cfg.get("visibility", {}).get("gimbal_lock_margin_deg", 12))
    centroid = np.asarray(target["centroid"], float); up = np.asarray(target["up"], float)
    u, v = _orthonormal_basis(up)
    out = []
    if q_home is not None and _config_ok(guard, q_home):
        out.append(list(map(float, q_home)))
    for h in heights:
        for tilt_deg, azs in ((0.0, (0.0,)), (15.0, (0.0, 90.0, 180.0, 270.0))):
            for az_deg in azs:
                t, a = math.radians(tilt_deg), math.radians(az_deg)
                cam_pos = centroid + h * (math.cos(t) * up
                                          + math.sin(t) * (math.cos(a) * u + math.sin(a) * v))
                for roll in (0.0, 90.0, 180.0, 270.0):
                    T = np.eye(4); T[:3, :3] = look_at_R(cam_pos, centroid, roll); T[:3, 3] = cam_pos
                    flange = gripper_pose_from_camera_T(T, T_c2g)
                    if abs(abs(flange[4]) - 90.0) < gimbal_margin:
                        continue
                    for c in range(8):
                        err, q = _pair(robot.rpc.GetInverseKin(0, list(map(float, flange)), c))
                        if err != 0 or q is None:
                            continue
                        q = list(map(float, q))
                        if any(_jdist(q, s) < 1.0 for s in out):
                            continue
                        if _config_ok(guard, q):
                            out.append(q)
    return out


def _via_route(q_from, arrivals, vias, guard):
    """Cheapest safe route q_from -> via -> arrival. arrivals: list of joint
    configs. Returns (via, index of the arrival) or None."""
    pairs = sorted(((_move_cost(q_from, v) + _move_cost(v, q), iv, ia)
                    for iv, v in enumerate(vias) for ia, q in enumerate(arrivals)))
    first_ok = {}
    for _, iv, ia in pairs:
        if iv not in first_ok:
            first_ok[iv] = _path_ok(guard, q_from, vias[iv])
        if first_ok[iv] and _path_ok(guard, vias[iv], arrivals[ia]):
            return vias[iv], ia
    return None


def plan_sequence(poses, q_start, target, cfg, robot, T_c2g, K, D, guard):
    """Re-plans the (ordered) poses one after the other, so that the robot can
    really go from each to the next:
      - IK: the solution NEAREST to the previous pose (GetInverseKinRef), so
        neighbours stay in the same arm configuration (no wrist flips);
      - roll: the camera may turn about its optical axis within
        +-planning.roll_range_deg of the layer roll (same viewpoint and framing);
        the roll with the smallest joint change + roll_weight x |roll offset| is
        taken. Kept small because MASt3R is not rotation invariant: the images
        stay roughly upright;
      - the joint path from the previous pose is collision-checked HERE
        (camera-vs-link, floor, sample cube, limits);
      - if no roll gives a safe direct path with that solution, the other arm
        configurations are tried directly (nominal solution, every IK branch);
      - only if none works, the arm changes configuration through a 'via' pose
        high above the sample (_via_configs); if that fails too, the pose is
        dropped.
    Each returned pose has 'joints' / 'flange' / 'cam_T_base' / 'roll_deg' of the
    chosen solution, 'via' ([] or [via joints]) and 'planned' = True."""
    pl = cfg.get("planning", {}) or {}
    if not bool(pl.get("enable", True)) or not poses:
        return poses
    rng = float(pl.get("roll_range_deg", 45.0))
    step = max(1.0, float(pl.get("roll_step_deg", 15.0)))
    rw = float(pl.get("roll_weight", 0.5))       # cost of 1 deg of roll, in deg of joint motion
    offsets = [0.0] + [s * k * step for k in range(1, int(rng // step) + 1) for s in (1, -1)]
    ev = _planning_ctx(target, cfg, robot, T_c2g, K, D)
    vias = None
    out, q_prev = [], list(map(float, q_start))
    n_via = n_drop = 0
    for wp in poses:
        az, tilt, dist, roll0 = wp["view"]
        cands = []
        for off in offsets:
            p, _ = ev(az, tilt, dist, roll0 + off, q_ref=q_prev)
            if p is not None and _config_ok(guard, p["joints"]):
                cands.append((_move_cost(p["joints"], q_prev) + rw * abs(off), off, p))
        cands.sort(key=lambda c: c[0])
        chosen, via = None, []
        for _, off, p in cands:
            if _path_ok(guard, q_prev, p["joints"]):
                chosen = (off, p)
                break
        if chosen is None:
            # the other arm configurations, still DIRECTLY: the nominal solution
            # of generate_poses and every IK branch (config 0..7) of each roll.
            more = []
            if _config_ok(guard, wp["joints"]):
                more.append((_move_cost(wp["joints"], q_prev), 0.0,
                             {"flange": wp["flange"], "joints": list(map(float, wp["joints"])),
                              "cam_T_base": wp["cam_T_base"]}))
            for off in offsets:
                for c in range(8):
                    p, _ = ev(az, tilt, dist, roll0 + off, ik_config=c)
                    if p is not None and _config_ok(guard, p["joints"]):
                        more.append((_move_cost(p["joints"], q_prev) + rw * abs(off), off, p))
            uniq = []
            for c_ in sorted(cands + more, key=lambda c: c[0]):
                if not any(_jdist(c_[2]["joints"], u_[2]["joints"]) < 1.0 for u_ in uniq):
                    uniq.append(c_)
            seen_ = cands
            for _, off, p in uniq:
                if any(p is s[2] for s in seen_):
                    continue                              # already path-checked above
                if _path_ok(guard, q_prev, p["joints"]):
                    chosen = (off, p)
                    break
        if chosen is None:
            if vias is None:
                vias = _via_configs(target, cfg, robot, T_c2g, guard, q_start)
                print(f"[plan] {len(vias)} safe via configurations above the sample")
            arr = [(off, p) for _, off, p in uniq]
            route = _via_route(q_prev, [a[1]["joints"] for a in arr], vias, guard) if arr else None
            if route is None:
                n_drop += 1
                print(f"[plan] {wp['layer']} az={wp['azim_deg']:.0f}: no safe path, not even "
                      "through a pose above the sample -> dropped")
                continue
            chosen, via = arr[route[1]], [route[0]]
            n_via += 1
        off, p = chosen
        new = dict(wp)
        new.update(flange=p["flange"], joints=p["joints"], cam_T_base=p["cam_T_base"],
                   roll_deg=round(roll0 + off, 1), via=via, planned=True)
        out.append(new)
        q_prev = p["joints"]
    jumps = [_jdist(a["joints"], b["joints"]) for a, b in zip(out[:-1], out[1:]) if not b["via"]]
    print(f"[plan] {len(out)}/{len(poses)} poses planned in sequence; {n_via} reached through "
          f"a pose above the sample, {n_drop} dropped; largest direct joint change "
          f"{max(jumps) if jumps else 0:.0f} deg; roll offsets used "
          f"{sorted({round(o['roll_deg'] - o['view'][3]) for o in out})} deg")
    return out


def run_acquire(cfg, robot, T_c2g, K, D, rs, out_dir, use_viz=True):
    tool = int(cfg["robot"]["tool"]); user = int(cfg["robot"]["user"])
    vel = float(cfg["robot"].get("vel", 20))
    # 'dwell': wait BEFORE THE SHOT, applied ONLY after the servo has
    # confirmed the centred pose (not before the check) - reduces the residual
    # motion blur from mechanical settling without wasting time when another
    # correction is still needed.
    dwell = float(cfg["robot"].get("dwell_s", 2.5))
    pt_gate = float(cfg["robot"].get("pointing_gate_deg", 15.0))

    target = bootstrap(robot, rs, cfg, T_c2g, K, D)
    poses = generate_poses(target, cfg, robot, T_c2g, K, D)

    # start pose (bootstrap): entry point for the order optimisation
    # AND pose to return to at the end of the acquisition.
    q_home = robot.get_joints()

    # execution order optimised in joint space (poses unchanged, only the
    # order): each ring starts from the ideal pose to minimise
    # backtracking between consecutive azimuths (see _reorder_poses_min_travel).
    poses = _reorder_poses_min_travel(poses, q_home)

    print(f"\nTotal poses to execute: {len(poses)}")
    for i, wp in enumerate(poses, 1):
        print(f"  {i:2d}: {wp['layer']:>14} az={wp['azim_deg']:>5.0f}  "
              f"joints={[round(v,1) for v in wp['joints']]}  "
              f"flange={[round(v,1) for v in wp['flange']]}")

    # -------------------------- collision checking --------------------------
    # Geometric guard (URDF+FCL) VALIDATED against the controller GetForwardKin
    # (see collision_guard.py). q_home (above) is also the detour waypoint.
    guard = None
    try:
        from collision_guard import CollisionGuard
        guard = CollisionGuard(cfg, robot, pose_to_T, robot.forward_kin,
                               T_c2g=T_c2g, euler_order=_EULER)
        if guard.enable and guard.available:
            guard.set_target(target["centroid"])
            guard.validate([wp["joints"] for wp in poses])
        if not guard.enable:
            guard = None
    except Exception as e:
        print(f"[collision] warning: guard not initialised ({e}); no geometric checks.")
        guard = None

    # sequential planning: IK continuity between neighbours, free roll within a
    # range, paths checked now, configuration changes through a pose above the
    # sample (see plan_sequence). Needs the validated guard for the path checks.
    poses = plan_sequence(poses, q_home, target, cfg, robot, T_c2g, K, D, guard)
    for i, wp in enumerate(poses, 1):
        if wp.get("planned"):
            print(f"  {i:2d}: {wp['layer']:>14} az={wp['azim_deg']:>5.0f} roll={wp['roll_deg']:>6.1f}"
                  f"  joints={[round(v, 1) for v in wp['joints']]}"
                  + ("  (through a pose above the sample)" if wp["via"] else ""))

    # per-pose (arrival) safety check, computed ONCE and reused both to
    # colour the Rerun preview (red = not safely reachable) and
    # for the report below. Uses resolve_arrival: a pose whose nominal IK
    # collides but that has a safe ALTERNATIVE IK is NOT marked red.
    pose_checks = None
    if guard is not None and guard.available and guard.validated:
        pose_checks = []
        for wp in poses:
            q_safe, info = guard.resolve_arrival(wp, q_home)
            pose_checks.append((q_safe is not None, info))

    # -------------------------- Rerun visualisation -------------------------
    # Phase 1 (generated poses + target cube): always, ALSO in --dry-run (it is how
    # to inspect the plan visually before moving the robot). Phases 2/3
    # (live robot) start only for a real run, further below. Any
    # error here does NOT block the acquisition/dry-run (viz = None).
    viz = None
    if use_viz:
        try:
            from pipeline_viz import PipelineViz
            viz = PipelineViz(cfg, K, T_c2g=T_c2g, euler_order=_EULER)
            viz.start(target["centroid"])
            if viz.ready:
                safe_mask = [ok for ok, _ in pose_checks] if pose_checks is not None else None
                viz.show_pose_preview(poses, safe_mask=safe_mask)
            else:
                viz = None
        except Exception as e:
            print(f"[viz] warning: viz not started ({e}); continuing without viewer")
            viz = None

    if cfg.get("_dry_run"):
        if pose_checks is not None:
            n_bad = 0
            for i, (wp, (ok, info)) in enumerate(zip(poses, pose_checks), 1):
                if not ok:
                    n_bad += 1
                    print(f"  [collision] pose {i:2d} ({wp['layer']} az={wp['azim_deg']:.0f}) "
                          f"NOT safely reachable: {info}")
            print(f"[collision] unreachable target poses: {n_bad}/{len(poses)} "
                  "(they would be skipped at runtime; the paths between poses are also "
                  "checked during execution, with a detour via home if needed).")
        print("\n[dry-run] no motion."
              + (" Rerun viewer open with the generated poses." if viz is not None else ""))
        return out_dir, 0

    # Safety GATE: if collision checking is ACTIVE but the model has not
    # been validated against the controller FK, we do NOT trust the
    # link geometry -> stop (same philosophy as the pointing gate).
    if guard is not None and guard.available and not guard.validated:
        print("\n[ABORT] Collision checking active but model NOT validated: "
              "stopping for safety. Fix URDF/euler_order/base, or "
              "disable 'collision.enable' in default.yaml[acquisition].")
        return out_dir, 0

    rgb_dir = os.path.join(out_dir, "rgb")
    depth_dir = os.path.join(out_dir, "depth")
    poses_dir = os.path.join(out_dir, "poses")
    camparam_dir = os.path.join(out_dir, "camera_parameters")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(camparam_dir, exist_ok=True)

    # -------- camera parameters, constant for the run (saved once) ----------
    #  intrinsics.npy  : K (3x3)
    #  distortions.npy : distortion coefficients (1D)
    #  camera2ee.npy   : hand-eye camera->flange (4x4). default.yaml[acquisition] is in mm;
    #                    here it is saved in METRES (translation /1000) for consistency
    #                    with the per-shot poses (also in metres).
    np.save(os.path.join(camparam_dir, "intrinsics.npy"), np.asarray(K, float))
    np.save(os.path.join(camparam_dir, "distortions.npy"), np.asarray(D, float).ravel())
    with open(os.path.join(camparam_dir, "intrinsics_source.json"), "w") as f:
        json.dump(cfg.get("_intrinsics_source", {}), f, indent=2)
    cam2ee = np.asarray(T_c2g, float).copy()
    cam2ee[:3, 3] /= 1000.0
    np.save(os.path.join(camparam_dir, "camera2ee.npy"), cam2ee)
    print(f"[out] saved intrinsics.npy / distortions.npy / camera2ee.npy (mm->m) in {camparam_dir}")

    # Phase 2 (live robot): keeps showing only the generated poses for 'viz.prelude_s'
    # seconds, then hands over to the execution loop (the robot appears
    # at the first real move, see reveal_robot() further below).
    if viz is not None:
        viz.wait_prelude()
        viz.begin_robot_phase(cfg.get("robot", {}).get("ip", "192.168.58.2"))

    # native REACTIVE safety net (current/torque): last physical line of defence
    if guard is not None:
        guard.enable_native_backstop()

    manifest = []
    print("\n" + "=" * 60 + "\n  EXECUTION\n" + "=" * 60)
    MAXFAIL = 3; fails = 0; n_saved = 0
    q_prev = q_home
    try:
      for i, wp in enumerate(poses, 1):
        # collision-checked MOTION PLANNING: checks the arrival pose
        # (vs cube/floor/self/camera-link) AND the joint PATH q_prev->target
        # (camera-vs-link during the motion). If the direct path collides, try a
        # detour via 'home'; if that is not free either, skip the pose.
        planned_ok = False
        if wp.get("planned"):
            # the planned route, re-checked from where the robot really is (the
            # servo may have moved it a little); otherwise the usual planner.
            seq = [list(q) for q in wp.get("via", [])] + [list(wp["joints"])]
            pts = [q_prev] + seq
            planned_ok = all(_path_ok(guard, pts[k], pts[k + 1]) for k in range(len(seq)))
            if planned_ok:
                mode, wps_move = ("via" if wp.get("via") else "direct"), seq
                info = "planned" + (" through a pose above the sample" if wp.get("via") else "")
        if planned_ok:
            pass
        elif guard is not None:
            mode, wps_move, info = guard.plan_move(q_prev, wp, q_home)
        else:
            mode, wps_move, info = "direct", [wp["joints"]], ""
        if mode == "skip":
            print(f"[{i}/{len(poses)}] {wp['layer']} az={wp['azim_deg']:.0f} "
                  f"-> SKIPPED (collision): {info}")
            continue
        # configuration actually executed for the pose (may be an IK
        # alternative to wp['joints'] if the nominal one collided).
        q_final = wps_move[-1]
        if mode in ("detour", "via"):
            print(f"[{i}/{len(poses)}] {wp['layer']} az={wp['azim_deg']:.0f}: {info} "
                  f"-> final joints {[round(v,1) for v in q_final]}")
        else:
            print(f"[{i}/{len(poses)}] {wp['layer']} az={wp['azim_deg']:.0f} -> MoveJ "
                  f"joints={[round(v,1) for v in q_final]}"
                  + (f"  [{info}]" if 'alternative' in info else '')
                  + f"  (expected flange {[round(v,1) for v in wp['flange']]})")

        # show the robot in the viz ONLY now that it is about to really move (poses
        # skipped above with 'continue' do not reveal it: no idle robot on screen).
        if viz is not None:
            viz.reveal_robot()

        err = 0
        for k, q_wp in enumerate(wps_move):
            # desc_pose (informative) only for the FINAL waypoint = target pose
            dp = wp["flange"] if k == len(wps_move) - 1 else None
            err = robot.move_j(q_wp, tool, user, vel, desc_pose=dp)
            if err != 0:
                break
        if err != 0:
            fails += 1
            sc = robot.safety_code()
            print(f"  MoveJ failed (code {err}, safety={sc}) [{fails}/{MAXFAIL}]")
            if fails >= MAXFAIL:
                print(f"\n[SAFETY] {fails} consecutive failures: probable "
                      f"error/collision state on the controller. Stopping. Check the alarms "
                      f"and call reset_errors before restarting.")
                break
            continue
        fails = 0
        q_prev = robot.get_joints() or wp["joints"]

        # POINTING GATE: does the executed optical axis (native FK) point at the centroid?
        T_c2b, _ = camera_T_base(robot, T_c2g)
        if T_c2b is not None:
            z_now = T_c2b[:3, 2]
            to_c = target["centroid"] - T_c2b[:3, 3]
            to_c = to_c / (np.linalg.norm(to_c) + 1e-9)
            pe = math.degrees(math.acos(max(-1.0, min(1.0, float(z_now @ to_c)))))
            print(f"  [pointing] optical-axis->centroid error = {pe:.1f} deg")
            if i == 1 and pe > pt_gate:
                print(f"\n[ABORT] First pose: optical axis {pe:.0f} deg from the centroid "
                      f"(>{pt_gate:.0f}). The chain is NOT correct: run --verify and "
                      f"check hand-eye/convention before continuing.")
                break

        # Logic: as soon as the pose is reached, CHECK IMMEDIATELY (Move* is blocking,
        # so the robot is already still). If the servo must correct, it does so
        # immediately (no wait before); only once the pose is CONFIRMED
        # centred do we wait 'dwell' seconds and shoot. This way no time is
        # wasted waiting before knowing whether another move is needed.
        seen = recenter(robot, rs, cfg, T_c2g, K, D, target, guard=guard)
        if not seen:
            print("  sample not in view -> shot SKIPPED"); continue

        t0 = time.time()
        while time.time() - t0 < dwell:
            try: rs.pipe.wait_for_frames(10000)
            except Exception: pass

        img = rs.grab(int(cfg["camera"].get("n_avg_frames", 5)))
        if img is None:
            print("  no frame -> skipping"); continue

        # ACTUAL pose at shot time (native FK after recentring,
        # robot still). The .txt holds the FLANGE pose in base: Vision
        # composes it with camera_parameters/camera2ee.npy (T_world_cam = T_world_ee @
        # camera2ee). Without a pose the shot is DISCARDED: Vision pairs images and
        # poses by order, and an orphan image would invalidate ALL GT poses.
        T_shot, _ = camera_T_base(robot, T_c2g)
        if T_shot is None:
            print("  FK unavailable -> shot DISCARDED (image without pose)"); continue
        T_flange = T_shot @ np.linalg.inv(T_c2g)

        tag = f"{n_saved + 1:04d}"
        img_name = f"img_{tag}.png"
        txt_name = f"pose_{tag}.txt"
        if not _save_image_pose_pair(img, os.path.join(rgb_dir, img_name),
                                     T_flange, os.path.join(poses_dir, txt_name)):
            print(f"  saving pair {img_name}/{txt_name} failed -> shot DISCARDED")
            continue
        n_saved += 1

        # depth channel co-registered with color (same resolution/intrinsics),
        # 16-bit PNG in MILLIMETRES (0 = no measurement). Same instant, robot
        # still. Saved ONLY if the layer of this pose has depth enabled
        # (wp['depth']); if depth is unavailable, skip without blocking.
        depth_rel = None
        want_depth = bool(wp.get("depth", True))
        if want_depth:
            depth = rs.grab_depth()
            if depth is not None:
                depth_name = f"depth{tag}.png"
                cv.imwrite(os.path.join(depth_dir, depth_name), depth)
                depth_rel = f"depth/{depth_name}"

        manifest.append({"index": i, "image": f"rgb/{img_name}",
                         "depth": depth_rel,
                         "pose_txt": f"poses/{txt_name}", "pose_frame": "flange",
                         "layer": wp["layer"],
                         "azim_deg": wp["azim_deg"], "flange_cmd": wp["flange"],
                         "joints_cmd": wp["joints"], "joints_deg": robot.get_joints()})
        if depth_rel:
            depth_msg = f" + {depth_rel}"
        elif want_depth:
            depth_msg = " (depth n/a)"
        else:
            depth_msg = " (depth OFF for this layer)"
        print(f"  saved rgb/{img_name}" + depth_msg + f" (+ poses/{txt_name})")

        # FINAL pose: pins the frustum in the viz at the same camera pose,
        # so it stays visible while the robot moves on to the next ones.
        if viz is not None and T_shot is not None:
            viz.log_photo_pose(T_shot, tag)

      # ---- end of acquisition: return to the start pose (bootstrap) ----
      if q_home is not None:
          safe, reasons = True, []
          if guard is not None:
              safe, reasons = guard.path_safe(q_prev, q_home)
          if safe:
              print(f"\n[home] returning to the start pose: MoveJ "
                    f"{[round(v,1) for v in q_home]}")
              robot.move_j(q_home, tool, user, vel)
          else:
              route = _via_route(q_prev, [q_home],
                                 _via_configs(target, cfg, robot, T_c2g, guard, None), guard)
              if route is not None:
                  print("\n[home] returning to the start pose through a pose above the sample")
                  if robot.move_j(route[0], tool, user, vel) == 0:
                      robot.move_j(q_home, tool, user, vel)
              else:
                  print(f"\n[home] return to the start pose NOT safe "
                        f"({'; '.join(reasons)}), not even through a pose above the "
                        "sample; leaving the robot where it is.")
    finally:
        # Phase 3: robot hidden, all acquired poses + target cube remain.
        if viz is not None:
            viz.finalize()
        # Image<->pose consistency check + checksum: ALWAYS runs, even if
        # the acquisition was interrupted (Ctrl-C / robot error), so the
        # folder stays readable by Vision.
        manifest = finalize_dataset(out_dir, manifest)
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

    n_saved = len(manifest)
    print(f"\nDone. {n_saved} image/pose pairs in {out_dir}")
    return out_dir, n_saved


# -----------------------------------------------------------------------------
#  Output in the layout read by Vision (DATA/input_vision/<sample>/)
# -----------------------------------------------------------------------------
#  Vision (spectra/vision/inputs.py) takes the sorted images in rgb/ and the
#  sorted poses poses/pose_*.txt and PAIRS THEM BY ORDER, only if the two
#  counts match (otherwise all poses are ignored, with a warning). Here
#  we guarantee 1:1 by index: img_NNNN.png <-> pose_NNNN.txt.
VISION_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
_IMG_RE = re.compile(r"^img_(\d+)$")
_POSE_RE = re.compile(r"^pose_(\d+)$")


def _write_pose_txt(T_mm, path):
    x, y, z, rr_, pp, yy = _cam_pose_xyzrpy_m_rad(T_mm)
    vals = [x, y, z, rr_, pp, yy]
    if not all(math.isfinite(v) for v in vals):
        raise ValueError(f"non-finite pose: {vals}")
    with open(path, "w") as f:
        f.write(" ".join(f"{v:.6f}" for v in vals) + "\n")


def _save_image_pose_pair(img, img_path, T_flange_mm, pose_path):
    """Writes image + pose as a UNIT: if either fails, removes the
    other. Returns True only if both are on disk."""
    ok = False
    try:
        ok = bool(cv.imwrite(img_path, img))
        if ok:
            _write_pose_txt(T_flange_mm, pose_path)
            return True
    except Exception as e:
        print(f"  [save] error: {e}")
    for p in (img_path, pose_path):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    return False


def _read_pose6(path):
    try:
        with open(path, "r") as f:
            v = [float(t) for t in f.read().split()]
    except (OSError, ValueError):
        return None
    return v if len(v) == 6 and all(math.isfinite(t) for t in v) else None


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def finalize_dataset(out_dir, manifest):
    """1) for every image in rgb/ checks that the pose with the same index
       exists (and is readable: 6 finite floats); for every pose checks that
       the image exists. Orphans (and unreadable poses with their image)
       are DISCARDED, i.e. moved to _discarded/ (not read by Vision).
    2) final check: same number of images and poses, same indices.
    3) SHA-256 checksum of all images and poses in SHA256SUMS (format
       `sha256sum -c`), also useful to verify the copy on Mufasa.
    Returns the manifest filtered to the kept pairs only."""
    rgb_dir = os.path.join(out_dir, "rgb")
    poses_dir = os.path.join(out_dir, "poses")
    disc_dir = os.path.join(out_dir, "_discarded")

    def _index(folder, rx, exts):
        out, other = {}, []
        if not os.path.isdir(folder):
            return out, other
        for fn in sorted(os.listdir(folder)):
            p = os.path.join(folder, fn)
            stem, ext = os.path.splitext(fn)
            if not os.path.isfile(p) or ext.lower() not in exts:
                continue
            m = rx.match(stem)
            if m:
                out[int(m.group(1))] = p
            else:
                other.append(p)        # files Vision would read but that cannot be paired
        return out, other

    imgs, img_other = _index(rgb_dir, _IMG_RE, VISION_IMAGE_EXTS)
    poses, pose_other = _index(poses_dir, _POSE_RE, {".txt"})

    def _discard(path, why):
        sub = os.path.join(disc_dir, os.path.basename(os.path.dirname(path)))
        os.makedirs(sub, exist_ok=True)
        dst = os.path.join(sub, os.path.basename(path))
        if os.path.exists(dst):
            dst += f".{int(time.time())}"
        shutil.move(path, dst)
        print(f"  [check] DISCARDED {os.path.relpath(path, out_dir)}: {why}")

    for p in img_other + pose_other:
        _discard(p, "non-conforming name (img_NNNN / pose_NNNN)")
    for k in sorted(set(imgs) - set(poses)):
        _discard(imgs.pop(k), "image without pose")
    for k in sorted(set(poses) - set(imgs)):
        _discard(poses.pop(k), "pose without image")
    for k in sorted(poses):
        if _read_pose6(poses[k]) is None:
            _discard(poses.pop(k), "unreadable pose (6 floats expected)")
            _discard(imgs.pop(k), "its pose is unreadable")

    # ---- final check -------------------------------------------------------
    n_img, n_pose = len(imgs), len(poses)
    if n_img != n_pose or sorted(imgs) != sorted(poses):
        raise RuntimeError(f"[check] residual inconsistency: {n_img} images vs "
                           f"{n_pose} poses (indices {sorted(imgs)} vs {sorted(poses)})")

    lines = []
    for k in sorted(imgs):
        for p in (imgs[k], poses[k]):
            rel = os.path.relpath(p, out_dir).replace(os.sep, "/")
            lines.append(f"{_sha256(p)}  {rel}")
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", newline="\n") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))
    print(f"[check] OK: {n_img} images == {n_pose} poses (indices 1:1); "
          f"checksum of {len(lines)} files in SHA256SUMS")

    kept = {os.path.basename(p) for p in imgs.values()}
    return [e for e in manifest if os.path.basename(e.get("image", "")) in kept]


# -----------------------------------------------------------------------------
#  main
# -----------------------------------------------------------------------------
def load_config(path, sample=None):
    """Reads the SINGLE pipeline config and returns the acquisition dict
    (`acquisition:` section), plus:
      - cfg['aruco']   <- vision.aruco (dictionary + marker edge: SINGLE value)
      - cfg['_out_dir']<- <data_root>/input_vision/<sample_name>, i.e.
                          exactly the folder Vision reads."""
    with open(path, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    acq = full.get("acquisition")
    if not isinstance(acq, dict):
        raise SystemExit(f"[config] 'acquisition:' section missing in {path}")
    cfg = copy.deepcopy(acq)

    vision = full.get("vision") or {}
    va = dict(vision.get("aruco") or {})
    # Same rule as the pipeline: the marker edge length is set ONLY in the
    # repo's configs/default.yaml; any other config gets it from there.
    if os.path.abspath(path) != os.path.abspath(DEFAULT_CONFIG):
        if va.get("marker_edge_length_mm") is not None:
            raise SystemExit(f"[config] {path}: vision.aruco.marker_edge_length_mm cannot be "
                             f"set here, only in {DEFAULT_CONFIG}")
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            dflt = yaml.safe_load(f) or {}
        va["marker_edge_length_mm"] = ((dflt.get("vision") or {}).get("aruco") or {}).get(
            "marker_edge_length_mm")
    if va.get("marker_edge_length_mm") is None:
        raise SystemExit(f"[config] vision.aruco.marker_edge_length_mm missing in {DEFAULT_CONFIG}")
    dict_name = str(va.get("dictionary", "4x4_50")).strip()
    if not dict_name.upper().startswith("DICT_"):
        dict_name = "DICT_" + dict_name.upper()
    cfg["aruco"] = {"dictionary": dict_name,
                    "marker_length_mm": float(va["marker_edge_length_mm"])}

    # Vision derives rgb/ poses/ camera_parameters/ from sample_name ONLY if
    # vision.input.* are null: if they are set by hand, we would write elsewhere.
    v_in = vision.get("input") or {}
    fixed = [k for k in ("rgb_dir", "pose_dir", "camera_params_dir") if v_in.get(k)]
    if fixed:
        raise SystemExit(f"[config] vision.input.{fixed} set by hand: the acquisition "
                         "only writes the auto-derived layout. Set them back to null.")
    sample = sample or full.get("sample_name")
    if not sample:
        raise SystemExit("[config] sample_name missing (config or --sample)")
    data_root = str(full.get("data_root", "DATA"))
    cfg["_sample"] = str(sample)
    # relative to the cwd, like Vision: always launch from the repo root
    cfg["_out_dir"] = os.path.abspath(os.path.join(data_root, "input_vision", str(sample)))
    cfg["_calib_dir"] = os.path.abspath(os.path.join(data_root, "camera_calibration"))
    print(f"[config] {path} | sample={sample} | aruco={dict_name} "
          f"L={cfg['aruco']['marker_length_mm']}mm | out={cfg['_out_dir']}")
    return cfg


_OUT_ITEMS = ("rgb", "poses", "depth", "camera_parameters", "_discarded",
              "manifest.json", "SHA256SUMS")


def prepare_out_dir(out_dir, overwrite=False):
    """NEVER silently overwrites an already acquired sample."""
    existing = [n for n in _OUT_ITEMS if os.path.exists(os.path.join(out_dir, n))]
    if not existing:
        return
    if not overwrite:
        raise SystemExit(f"[out] {out_dir} already contains {existing}. Use another "
                         "--sample or --overwrite to replace it.")
    for n in existing:
        p = os.path.join(out_dir, n)
        shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    print(f"[out] --overwrite: removed {existing} from {out_dir}")


def main():
    """Returns the number of saved image/pose pairs (None for --verify / --reset)."""
    global _EULER
    ap = argparse.ArgumentParser(description="Layered acquisition (native Fairino FK)")
    ap.add_argument("--config", default=DEFAULT_CONFIG,
                    help="unified pipeline config (default: configs/default.yaml)")
    ap.add_argument("--sample", default=None,
                    help="override sample_name: output in DATA/input_vision/<sample>/")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace an existing acquisition for the same sample")
    ap.add_argument("--ip", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="check the transformation chain WITHOUT moving the robot")
    ap.add_argument("--dry-run", action="store_true",
                    help="bootstrap + pose generation, but does NOT move the robot")
    ap.add_argument("--reset", action="store_true", help="ResetAllError and exit")
    ap.add_argument("--no-viz", action="store_true",
                    help="disable the Rerun visualisation (poses/cube/live robot)")
    args = ap.parse_args()

    cfg = load_config(args.config, sample=args.sample)
    if not (args.verify or args.dry_run or args.reset):
        prepare_out_dir(cfg["_out_dir"], overwrite=args.overwrite)   # fail-fast, before touching the robot
    ip = args.ip or cfg["robot"].get("ip", "192.168.58.2")
    cfg.setdefault("robot", {})["ip"] = ip   # consistent for the second viz connection

    T_c2g = load_T_cam2gripper(cfg)
    robot = RobotIO(ip)

    try:
        if args.reset:
            robot.reset_errors(); print("[robot] errors reset."); return

        # Euler convention: override from yaml if set, otherwise autodetect
        forced = cfg["robot"].get("euler_order", "AUTO")
        if str(forced).upper() in ("XYZ", "ZYX"):
            _EULER = str(forced).upper()
            print(f"[euler] convention forced by config = {_EULER}")
        else:
            detect_euler_convention(robot)

        rs = RS(cfg)
        try:
            K, D = resolve_intrinsics(cfg, rs)
            if args.verify:
                run_verify(cfg, robot, T_c2g, K, D, rs)
                return
            cfg["_dry_run"] = args.dry_run
            out_dir = cfg["_out_dir"]
            _, n_saved = run_acquire(cfg, robot, T_c2g, K, D, rs, out_dir,
                                     use_viz=not args.no_viz)
        finally:
            rs.stop()
    finally:
        robot.close()

    # Acquisition only: vision / registration (here or on Mufasa, see the
    # `mufasa:` section) are run by `spectra full`, which calls this main().
    return n_saved


if __name__ == "__main__":
    main()