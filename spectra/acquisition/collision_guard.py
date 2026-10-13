# -*- coding: utf-8 -*-
"""
collision_guard.py
==================
PREDICTIVE collision checking for the acquisition pipeline (acquire.py).

WHY URDF (and not "native zones")
    The native Fairino library does NOT expose any function to define, from
    Python, geometric planes/zones/cubes to avoid: safety walls (安全墙) and
    interference zones (干涉区) can only be configured from the teach pendant /
    web UI, and the SDK can only READ their alarm. The SDK only offers REACTIVE
    collision detection based on current/torque (SetAnticollision), which we use
    here as a runtime safety net but which is NOT predictive. Predictive checks
    (above all camera-vs-link during motions) need a geometric model: the FR5
    URDF (via collision_fr5 + FCL).

VALIDATING THE GEOMETRY AGAINST REALITY
    The controller does NOT expose the poses of the individual links
    (GetForwardKin only gives the flange), so each link cannot be compared with
    reality. The equivalent independent test is decisive anyway: the URDF
    flange FK is compared with the controller GetForwardKin over many
    configurations (the planned poses + random configs). If the endpoint of the
    URDF chain matches the controller truth within tolerance, the WHOLE chain —
    and therefore the intermediate link poses derived from it — is reliable. The
    guard REFUSES to authorise motions if this test fails (validate()).

CHECKS
    1) no link/joint (except base), EE or camera below the minimum height above
       the base plane z=0 (threshold 'floor_margin_mm', default 15 mm);
    2) no link/joint/EE/camera/connector inside the sample cube
       (centroid +/- 'sample_tol_mm') — also along the motions;
    3) pre-check of the camera poses vs the cube (at runtime, before every MoveJ);
    4) MOST IMPORTANT: camera-vs-link collision along the PATH between two poses.
       MoveJ interpolates in JOINT space: we sample the joint-space line
       q0->q1 and check every sample (camera-vs-link, self, floor, cube,
       limits). If the path collides -> detour via a safe "home" pose, and if
       that is not free either -> the pose is skipped.

Model distances are in METRES, base_link frame. The pipeline geometry is in mm
in the CONTROLLER base frame: we assume (like the rest of the pipeline) that
the two base frames coincide; we convert mm<->m.
"""

import os
import sys
import math

import numpy as np

try:
    from pipeline_viz import DEFAULT_VIZ_REPO
except Exception:
    DEFAULT_VIZ_REPO = os.path.dirname(os.path.abspath(__file__))   # spectra/acquisition


def _aabb_overlap(min_a, max_a, min_b, max_b):
    return bool(np.all(min_a <= max_b) and np.all(min_b <= max_a))


def _as_bool(x, default, name):
    """Interprets a YAML value as a boolean, flagging typos (e.g. 'ture').
    An unrecognised token is NOT silently treated as truthy: a warning is
    printed and 'default' is used."""
    if isinstance(x, bool):
        return x
    if x is None:
        return default
    s = str(x).strip().lower()
    if s in ("true", "1", "yes", "on"):
        return True
    if s in ("false", "0", "no", "off"):
        return False
    print(f"[collision] warning: '{name}: {x}' is not a valid boolean "
          f"(typo?); using the default {default}.")
    return default


class CollisionGuard:
    """Geometric guard based on collision_fr5 (URDF+FCL), validated against the
    controller's native FK. All geometric methods work on JOINT configurations
    in degrees (order j1..j6)."""

    def __init__(self, cfg, robot, pose_to_T, forward_kin, T_c2g=None, euler_order="ZYX"):
        """cfg: default.yaml[acquisition] dict. robot: RobotIO (for .rpc, forward_kin, get_joints).
        pose_to_T: function [x,y,z,rx,ry,rz]->4x4 (mm) with the Euler convention
        already detected by the pipeline. forward_kin: robot.forward_kin (GetForwardKin).
        T_c2g: the PIPELINE hand-eye (default.yaml[acquisition], mm) — REQUIRED for a
        correct camera-vs-link check: without it the model would use the hand-eye of
        a different project/calibration (rerun-viz/config.yaml) and the camera box
        would be placed in the WRONG spot relative to the real robot."""
        c = (cfg.get("collision") or {}) if isinstance(cfg, dict) else {}
        self.enable = _as_bool(c.get("enable", True), True, "collision.enable")
        self.floor_margin_mm = float(c.get("floor_margin_mm", 15.0))
        # sample cube tolerances: default = viz tolerances if present
        vtol = (cfg.get("viz") or {}).get("target_tol_mm", [45, 45, 30])
        self.sample_tol_mm = np.asarray(c.get("sample_tol_mm", vtol), float).ravel()[:3]
        self.path_samples = int(c.get("path_samples", 12))
        self.val_tol_mm = float(c.get("validate_tol_mm", 5.0))
        self.val_tol_deg = float(c.get("validate_tol_deg", 1.5))
        self.val_n_random = int(c.get("validate_n_random", 8))
        self.native_backstop = bool(c.get("native_backstop", True))
        self.native_level = c.get("native_level", [5, 5, 5, 5, 5, 5])
        self.native_strategy = int(c.get("native_strategy", 2))  # 2=report+stop
        self.floor_per_link_mm = c.get("floor_per_link_mm", {}) or {}
        self.repo_dir = os.path.normpath(c.get("repo_dir", DEFAULT_VIZ_REPO))
        # camera box: SINGLE source = default.yaml[acquisition] -> camera_box: (fallback to
        # the rerun-viz project config.yaml only for backward compatibility).
        self.camera_box_cfg = cfg.get("camera_box") if isinstance(cfg, dict) else None
        self.T_c2g = None if T_c2g is None else np.asarray(T_c2g, float)
        self.euler_order = str(euler_order or "ZYX")

        self.robot = robot
        self.pose_to_T = pose_to_T
        self.forward_kin = forward_kin

        self.col = None
        self.model = None
        self.available = False
        self.validated = False
        self.centroid_m = None
        self.cube_min = None
        self.cube_max = None
        self._cube_obj = None        # FCL box of the sample cube (exact test)
        self._fcl = None
        self._last_q = None          # last evaluated config (for the AABB fallback)

        if self.enable:
            self._build_model()

    # ------------------------------------------------------------ setup model
    def _build_model(self):
        try:
            src = os.path.join(self.repo_dir, "src")
            if os.path.isdir(src) and src not in sys.path:
                sys.path.append(src)
            import collision_fr5 as col

            urdf = os.path.join(self.repo_dir, "fairino_description",
                                "urdf", "fairino5_v6.urdf")

            # camera box: default.yaml[acquisition] (camera_box:) is the ONLY expected source;
            # fallback to the rerun-viz project config.yaml only if missing.
            CB = self.camera_box_cfg
            if CB is None:
                CB = self._fallback_camera_box()
            # hand-eye: the PIPELINE one (default.yaml[acquisition]), NEVER another project's
            # -- otherwise the camera box ends up in the wrong place relative to
            # the real robot and the camera-vs-link check is unreliable.
            T_c2g = self.T_c2g if self.T_c2g is not None else np.array(
                CB.get("T_cam2gripper", self._fallback_camera_box()["T_cam2gripper"]), float)

            # --- uniform floor rule at floor_margin_mm (base excluded) ---
            col.URDF_PATH = urdf
            col.MIN_Z_MM = self.floor_margin_mm
            col.EULER_ORDER = self.euler_order
            col.FLOOR_PER_LINK_MM = {k: float(v) for k, v in self.floor_per_link_mm.items()}
            col.FLOOR_IGNORE_LINKS = {"base_link"}
            # --- camera+adapter geometry (hand-eye referred to wrist3) ------
            col.T_CAM2GRIPPER_MM = np.asarray(T_c2g, float)
            col.CAM_BOX_L_MM = float(CB["length_mm"])
            col.CAM_BOX_W_MM = float(CB["width_mm"])
            col.CAM_BOX_H_MM = float(CB["height_mm"])
            col.SUPPORT_EXTENSION_MM = float(CB["support_extension_mm"])
            col.CAM_BOX_MARGIN_MM = float(CB.get("collision_margin_mm",
                                                 CB.get("margin_mm", 10.0)))

            self.model = col.FR5CollisionModel(
                urdf_path=urdf,
                mesh_root=os.path.join(self.repo_dir, "fairino_description"),
                verbose=False)
            self.col = col
            self.available = True
            print(f"[collision] URDF+FCL model loaded (floor z>={self.floor_margin_mm:.0f}mm, "
                  f"sample cube +/-{self.sample_tol_mm.tolist()}mm)")
        except Exception as exc:
            print(f"[collision] warning: model NOT loaded ({exc}); "
                  "geometric checks disabled (only the native backstop remains)")
            self.available = False

    def _fallback_camera_box(self):
        """Fallback ONLY if default.yaml[acquisition] does not define 'camera_box:' (backward compat):
        reads the box dimensions from the rerun-viz/ project config.yaml."""
        import yaml
        cfg_path = os.path.join(self.repo_dir, "config.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            vc = yaml.safe_load(f) or {}
        print("[collision] warning: 'camera_box:' missing in default.yaml[acquisition], using "
              f"{cfg_path} (add 'camera_box:' to default.yaml[acquisition] to remove "
              "this dependency)")
        return dict(vc.get("camera_box", {}))

    def set_target(self, centroid_mm):
        """Defines the sample cube around the centroid (mm, base) as an FCL box
        (EXACT collision test: no false positives from AABBs on rotated
        boxes, which is why oblique poses showed up 'inside the cube'
        without touching it)."""
        c = np.asarray(centroid_mm, float).ravel()[:3] / 1000.0
        half = self.sample_tol_mm / 1000.0
        self.centroid_m = c
        self.cube_min = c - half
        self.cube_max = c + half
        self._cube_obj = None
        if self.model is not None:
            try:
                import fcl
                box = fcl.Box(float(2 * half[0]), float(2 * half[1]), float(2 * half[2]))
                tf = fcl.Transform(np.eye(3), c.astype(float))
                self._cube_obj = fcl.CollisionObject(box, tf)
                self._fcl = fcl
            except Exception as exc:
                print(f"[collision] warning: FCL cube not created ({exc}); "
                      "using the AABB test (more conservative).")

    # --------------------------------------------------- validation (GATE)
    def _wrist3_mm(self, q_deg):
        T = self.model.fk(np.deg2rad(np.asarray(q_deg, float).ravel()))["wrist3_link"].copy()
        T[:3, 3] *= 1000.0
        return T

    @staticmethod
    def _rot_angle_deg(R):
        c = (np.trace(R) - 1.0) / 2.0
        return math.degrees(math.acos(max(-1.0, min(1.0, c))))

    def validate(self, configs_deg):
        """Independent test: URDF flange FK vs the controller GetForwardKin over
        several configurations. Does NOT move the robot (uses GetForwardKin on joints).
        Returns (ok, max_pos_mm, max_rot_deg). On ok=True sets self.validated."""
        if not self.available:
            return False, float("inf"), float("inf")
        jc = self.robot.get_joints()
        if jc is None:
            print("[collision] validation: cannot read the current joints.")
            return False, float("inf"), float("inf")

        # rigid offset wrist3(URDF) -> flange(controller), calibrated on the
        # current pose (GetForwardKin as ground truth).
        ctrl0 = self.pose_to_T(self.forward_kin(jc))         # 4x4 mm, controller base
        w0 = self._wrist3_mm(jc)                             # 4x4 mm, URDF base
        T_off = np.linalg.inv(w0) @ ctrl0

        # APPLY the offset to the collision model: the default.yaml[acquisition] hand-eye is
        # referred to the controller FLANGE, not to wrist3. Without this, the
        # camera (and its box) would be placed at wrist3 @ hand_eye instead
        # of wrist3 @ offset @ hand_eye = flange @ hand_eye: shifted by the tool
        # offset -> the floor / cube / camera-vs-link checks on the CAMERA would be
        # computed in the wrong place (the real camera could hit the base while
        # "passing" the check). collision_fr5's T_flange2tcp is in METRES.
        T_off_m = T_off.copy(); T_off_m[:3, 3] /= 1000.0
        self.model.T_flange2tcp = T_off_m
        off_mm = T_off[:3, 3]
        print(f"[collision] wrist3->flange offset applied to the camera model: "
              f"[{off_mm[0]:.1f}, {off_mm[1]:.1f}, {off_mm[2]:.1f}] mm")

        # set of configs: current pose + the planned poses + random within limits
        configs = [jc] + [list(map(float, q)) for q in (configs_deg or [])]
        rng_lo = np.array([-170, -120, -150, -170, -120, -170.0])
        rng_hi = np.array([170, 120, 150, 170, 120, 170.0])
        # deterministic 'random' (no system Date/random): samples a
        # pseudo-uniform grid derived from fixed indices.
        for k in range(self.val_n_random):
            frac = ((k + 1) * np.array([0.13, 0.29, 0.47, 0.61, 0.73, 0.89])) % 1.0
            configs.append((rng_lo + frac * (rng_hi - rng_lo)).tolist())

        max_pos = max_rot = 0.0
        for q in configs:
            try:
                ctrl = self.pose_to_T(self.forward_kin(q))
            except Exception:
                continue
            pred = self._wrist3_mm(q) @ T_off
            dpos = float(np.linalg.norm(pred[:3, 3] - ctrl[:3, 3]))
            drot = self._rot_angle_deg(pred[:3, :3].T @ ctrl[:3, :3])
            max_pos = max(max_pos, dpos)
            max_rot = max(max_rot, drot)

        ok = (max_pos <= self.val_tol_mm) and (max_rot <= self.val_tol_deg)
        self.validated = ok
        tag = "OK" if ok else "FAILED"
        print(f"[collision] URDF vs GetForwardKin validation: {tag}  "
              f"(max pos {max_pos:.2f}mm / threshold {self.val_tol_mm:.1f}, "
              f"max rot {max_rot:.2f}deg / threshold {self.val_tol_deg:.1f}, "
              f"{len(configs)} config)")
        if not ok:
            print("[collision] *** The URDF chain does NOT match the controller "
                  "FK: the link geometry is NOT trusted. Check "
                  "URDF/euler_order/base. Collision-gated motions REFUSED. ***")
        return ok, max_pos, max_rot

    # -------------------------------------------------- geometric checks
    def _cube_hit_fcl(self):
        """True if the (oriented) camera box or a link (except base) enters the
        sample cube. Uses the FCL objects ALREADY PLACED by check_config
        (called right before in config_safe): EXACT box-vs-box and
        mesh-vs-box test, without the AABB inflation on rotated geometry.
        Returns (hit, who). AABB fallback (conservative) if FCL is unavailable."""
        if getattr(self, "_cube_obj", None) is None:
            return self._cube_hit_aabb()
        fcl = self._fcl
        req = fcl.CollisionRequest()
        # camera (oriented box) vs cube
        res = fcl.CollisionResult()
        if fcl.collide(self.model.fcl_cam, self._cube_obj, req, res) and res.is_collision:
            return True, "camera"
        # link (mesh) vs cube
        for name, obj in self.model.fcl_link.items():
            if name == "base_link":
                continue
            res = fcl.CollisionResult()
            if fcl.collide(obj, self._cube_obj, req, res) and res.is_collision:
                return True, name
        return False, None

    def _cube_hit_aabb(self):
        """Fallback: AABB of the link vertices + AABB of the camera box vs cube.
        Conservative (may give false positives on strongly rotated geometry)."""
        if self.cube_min is None or self._last_q is None:
            return False, None
        Tl = self.model.fk(np.deg2rad(np.asarray(self._last_q, float).ravel()))
        for name, V in self.model.link_vertices.items():
            if name == "base_link":
                continue
            pv = (Tl[name][:3, :3] @ V.T).T + Tl[name][:3, 3]
            if _aabb_overlap(pv.min(axis=0), pv.max(axis=0), self.cube_min, self.cube_max):
                return True, name
        Tc2b = Tl["wrist3_link"] @ self.model.camera_pose_in_flange()
        cc = (Tc2b @ self.model._cam_corners_local)[:3, :].T
        if _aabb_overlap(cc.min(axis=0), cc.max(axis=0), self.cube_min, self.cube_max):
            return True, "camera"
        return False, None

    def config_safe(self, q_deg):
        """(bool, reasons) for a single joint configuration (degrees)."""
        if not (self.available and self.validated):
            return True, []          # without a validated model we do not judge
        reasons = []
        # check_config PLACES the FCL objects at this config (side effect) and
        # checks self-collision / camera-vs-link / floor / limits.
        res = self.model.check_config(q_deg, check_limits=True)
        self._last_q = q_deg          # for the possible AABB fallback
        if res["self_collisions"]:
            reasons.append("self-collision")
        if res["camera_collisions"]:
            links = ",".join(n for n, _ in res["camera_collisions"])
            reasons.append(f"camera-vs-link ({links})")
        if res["floor_violations"]:
            names = ",".join(n for n, _ in res["floor_violations"])
            reasons.append(f"below floor height ({names})")
        if res["limit_violations"]:
            reasons.append("joint limits exceeded")
        # sample cube: uses the FCL objects just placed by check_config.
        hit, who = self._cube_hit_fcl()
        if hit:
            reasons.append(f"inside sample cube ({who})")
        return (len(reasons) == 0), reasons

    def path_safe(self, q0, q1, n=None):
        """(bool, reasons) on the JOINT path q0->q1 (linear interpolation,
        consistent with MoveJ). Samples n points (endpoints included)."""
        if not (self.available and self.validated):
            return True, []
        q0 = np.asarray(q0, float).ravel()
        q1 = np.asarray(q1, float).ravel()
        n = int(n or self.path_samples)
        for t in np.linspace(0.0, 1.0, max(2, n)):
            q = q0 + t * (q1 - q0)
            ok, reasons = self.config_safe(q)
            if not ok:
                return False, [f"t={t:.2f}: " + "; ".join(reasons)]
        return True, []

    @staticmethod
    def _jdist(a, b):
        return float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float))))

    def resolve_arrival(self, wp, q_ref):
        """Finds a SAFE joint configuration that reaches the CAMERA pose
        of 'wp'. Tries the precomputed IK first (wp['joints']); if it collides, looks
        for ALTERNATIVE IK configurations for the SAME flange pose (exactly the same
        camera viewpoint, but a different arm posture: elbow up/down,
        wrist flip) and picks the collision-free one closest to q_ref (least
        travel). Returns (joints|None, info)."""
        if not (self.available and self.validated):
            return list(wp["joints"]), "guard inactive"
        ok, reasons = self.config_safe(wp["joints"])
        if ok:
            return list(wp["joints"]), "nominal IK"
        # alternative IK configurations for the SAME flange (config 0..7 =
        # solution branch; -1 = auto, already tried as wp['joints']).
        flange = list(map(float, wp["flange"]))
        best, bestcost, seen = None, float("inf"), []
        for cfg_id in range(8):
            try:
                err, js = self.robot.rpc.GetInverseKin(0, flange, cfg_id)
            except Exception:
                continue
            if err != 0 or js is None:
                continue
            js = list(js)
            if any(self._jdist(js, s) < 1.0 for s in seen):
                continue                      # duplicate solution
            seen.append(js)
            ok2, _ = self.config_safe(js)
            if ok2:
                cost = self._jdist(js, q_ref)
                if cost < bestcost:
                    best, bestcost = js, cost
        if best is not None:
            return best, "alternative IK (same view, different posture)"
        return None, "unsafe pose: " + "; ".join(reasons)

    def plan_move(self, q_prev, wp, q_home):
        """Decides how to reach the camera pose of 'wp' from q_prev safely.
        Returns (mode, waypoints, info):
          'direct' [q_target]         : direct path is free
          'detour' [q_home, q_target] : direct path unsafe, but via home it is safe
          'skip'   []                 : target pose unrecoverable or path/detour
                                        not safe.
        'wp' is the pose dict (uses 'joints' and 'flange'). q_target may be
        an ALTERNATIVE IK configuration if the nominal one collides (see
        resolve_arrival)."""
        if not (self.available and self.validated):
            return "direct", [list(wp["joints"])], "guard inactive"
        # (3) safe ARRIVAL configuration (with search for alternative IKs)
        q_target, ainfo = self.resolve_arrival(wp, q_prev)
        if q_target is None:
            return "skip", [], ainfo
        alt = "alternative" in ainfo
        # (4) direct path
        ok, reasons = self.path_safe(q_prev, q_target)
        if ok:
            return "direct", [q_target], ("direct path free" +
                                          (f" + {ainfo}" if alt else ""))
        info_direct = reasons[0] if reasons else "direct path unsafe"
        # detour via the home pose (known safe: the robot started from there)
        okA, _ = self.path_safe(q_prev, q_home)
        okB, _ = self.path_safe(q_home, q_target)
        if okA and okB:
            return "detour", [list(q_home), q_target], ("detour via home" +
                                                        (f" + {ainfo}" if alt else ""))
        return "skip", [], f"unsafe path ({info_direct}) and detour not feasible"

    # ----------------------------------------------- native Fairino backstop
    def enable_native_backstop(self):
        """Enables the native REACTIVE collision detection (current/torque) as
        the last physical line of defence during motion. Not predictive. Not fatal if it fails."""
        if not self.native_backstop:
            return
        rpc = getattr(self.robot, "rpc", None)
        if rpc is None:
            return
        try:
            rpc.SetCollisionDetectionMethod(0, 0)   # 0=current, fixed threshold
        except Exception as e:
            print(f"[collision] SetCollisionDetectionMethod failed: {e}")
        try:
            err = rpc.SetAnticollision(0, list(map(float, self.native_level)), 0)
            print(f"[collision] native anti-collision (level {self.native_level}) -> err={err}")
        except Exception as e:
            print(f"[collision] SetAnticollision failed: {e}")
        try:
            rpc.SetCollisionStrategy(self.native_strategy)
        except Exception as e:
            print(f"[collision] SetCollisionStrategy failed: {e}")
