# -*- coding: utf-8 -*-
"""
pipeline_viz.py
===============
Rerun visualisation orchestration for the acquisition pipeline (acquire.py).

GOAL
    A SINGLE Rerun window that, during a real acquisition, shows in sequence:

      Phase 1 (prelude, VIZ_PRELUDE_S seconds, configurable):
        ONLY the generated camera poses (frustums) + the target cube with its
        tolerances. The robot is NOT visible.

      Phase 2 (execution):
        the prelude poses disappear; the robot appears, animated in real time
        (URDF driven by the joints read live). Every time the robot stops at a
        FINAL pose (the one where it is about to take the photo), that frustum
        is logged and STAYS visible while the robot moves on to the next poses.

      Phase 3 (end):
        the robot disappears; ALL acquired poses + the target cube remain
        visible. The viewer stays open.

DESIGN PRINCIPLES
    - rerun_viz.py only holds the logging helpers used here
      (`_log_joint_transforms`, `_log_camera_frame`, `_log_collision_boxes`,
      `_log_floor`, `_check_urdf_meshes`); this module drives them.
    - The support modules (`collision_fr5`, the URDF + meshes) live in this
      same folder (spectra/acquisition/, see DEFAULT_VIZ_REPO). All numeric
      parameters (hand-eye, camera box) come from default.yaml[acquisition].
    - The pipeline geometry (centroid, poses) is in mm in the CONTROLLER base
      frame; the URDF robot lives in metres in the `base_link` frame. Like the
      rest of the pipeline, we assume the two base frames coincide:
      we convert mm->m and declare `rr.CoordinateFrame("base_link")` (the same
      trick rerun_viz uses for aruco/floor/collision boxes), so everything
      is consistent in the same scene as the robot.
    - Any viz error is NOT fatal for the acquisition: a warning is printed
      and the pipeline continues without the viewer.
"""

import os
import sys
import time
import threading

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))          # .../spectra/acquisition

# Dependencies (collision_fr5.py, fairino_description/ with URDF + meshes) now
# live in this same folder. Can be overridden with
# acquisition.viz.repo_dir in the unified config if they are ever moved.
DEFAULT_VIZ_REPO = HERE

# Defaults (overridden by default.yaml[acquisition] -> 'viz' section)
DEFAULT_PRELUDE_S = 5.0
DEFAULT_TARGET_TOL_MM = [45.0, 45.0, 30.0]   # half-sides of the target cube (tolerances)
DEFAULT_POLL_HZ = 20
DEFAULT_MEMORY_LIMIT = "512MB"
DEFAULT_FRUSTUM_LEN_M = 0.0015                 # length of the drawn frustums (m)

# colours (RGB / RGBA)
COLOR_PREVIEW = [80, 160, 240]               # safe generated poses (phase 1)
COLOR_PREVIEW_UNSAFE = [230, 30, 30]         # UNSAFE generated poses (collision guard)
COLOR_PHOTO = [40, 200, 90]                  # final acquired poses (phase 2/3)
COLOR_CUBE = [255, 60, 60, 60]               # target cube (semi-transparent)
COLOR_CUBE_EDGE = [255, 60, 60]
COLOR_CENTROID = [255, 60, 60]

FRUSTUM_RADIUS_M = 0.0015
POINT_RADIUS_M = 0.004


def _frustum_strips_m(T_cam2base_mm, K, w, h, length_m):
    """Camera frustum lines (apex + far rectangle) in METRES, in the base
    frame. `T_cam2base_mm` is 4x4 with translation in mm; K are the px intrinsics.
    Returns a list of strips (each a list of [x,y,z] in metres)."""
    Kinv = np.linalg.inv(np.asarray(K, float))
    cam_pos = np.asarray(T_cam2base_mm, float)[:3, 3] / 1000.0
    R = np.asarray(T_cam2base_mm, float)[:3, :3]
    corners = []
    for (uu, vv) in ((0, 0), (w, 0), (w, h), (0, h)):
        d = Kinv @ np.array([uu, vv, 1.0])
        d = d / (np.linalg.norm(d) + 1e-12) * length_m
        corners.append((cam_pos + R @ d).tolist())
    strips = [[cam_pos.tolist(), c] for c in corners]        # apex -> 4 corners
    strips.append([corners[0], corners[1], corners[2], corners[3], corners[0]])
    return strips, cam_pos.tolist()


class PipelineViz:
    """Handles the whole Rerun visualisation of the pipeline. Main methods:
    start() -> show_pose_preview() -> wait_prelude() -> begin_robot_phase()
    -> log_photo_pose() [for every shot] -> finalize().

    All methods are defensive: if the viz is not ready they become no-ops."""

    def __init__(self, cfg, K, T_c2g=None, euler_order="ZYX"):
        self.K = np.asarray(K, float)
        # PIPELINE hand-eye (default.yaml[acquisition], referred to the controller
        # flange/tool0) and detected Euler convention: used to draw the
        # live camera with the SAME calibration that drives the robot (otherwise the
        # viz would use the hand-eye of the rerun-viz config.yaml, which is a
        # DIFFERENT calibration -> camera rotated with respect to the real photos).
        self.pipeline_handeye = None if T_c2g is None else np.asarray(T_c2g, float)
        self.euler_order = str(euler_order or "ZYX")
        vcfg = (cfg.get("viz") or {}) if isinstance(cfg, dict) else {}
        self.enabled = bool(vcfg.get("enable", True))
        self.prelude_s = float(vcfg.get("prelude_s", DEFAULT_PRELUDE_S))
        tol = vcfg.get("target_tol_mm", DEFAULT_TARGET_TOL_MM)
        self.target_tol_mm = np.asarray(tol, float).ravel()[:3]
        self.poll_hz = int(vcfg.get("poll_hz", DEFAULT_POLL_HZ))
        self.memory_limit = str(vcfg.get("memory_limit", DEFAULT_MEMORY_LIMIT))
        self.frustum_len_m = float(vcfg.get("frustum_len_m", DEFAULT_FRUSTUM_LEN_M))
        self.viz_repo = os.path.normpath(vcfg.get("repo_dir", DEFAULT_VIZ_REPO))
        self.show_floor = bool(vcfg.get("show_floor", True))
        self.show_camera_frame = bool(vcfg.get("show_camera_frame", True))
        self.show_collision_boxes = bool(vcfg.get("show_collision_boxes", True))
        self.cam_pinhole_distance_m = float(vcfg.get("camera_pinhole_distance_m", 0.06))

        # camera+adapter box (physical dimensions) and floor height: SINGLE
        # source of truth = default.yaml[acquisition] (camera_box: / collision.floor_margin_mm),
        # NOT the rerun-viz project config.yaml (a different calibration/
        # geometry, used only for URDF/mesh/code).
        self.camera_box_cfg = cfg.get("camera_box") if isinstance(cfg, dict) else None
        ccfg = (cfg.get("collision") or {}) if isinstance(cfg, dict) else {}
        self.floor_margin_mm = float(ccfg.get("floor_margin_mm", 15.0))

        cam = cfg.get("camera", {}) if isinstance(cfg, dict) else {}
        self.img_w = int(cam.get("width", 1280))
        self.img_h = int(cam.get("height", 800))

        # runtime state
        self.ready = False
        self.rr = None
        self.viz = None                 # rerun_viz module (local)
        self.col = None                 # collision_fr5 module (local)
        self.urdf_tree = None
        self.col_model = None
        self.robot_name = None          # e.g. "fairino5_v6_robot"
        self.robot_viz = None           # second robot connection (read-only)
        self._stop = threading.Event()
        self._thread = None
        self._n_photo = 0
        self._revealed = False

    # ------------------------------------------------------------------ setup
    def _configure_viz_globals(self, viz, col, urdf_path):
        """Sets the collision_fr5/rerun_viz parameters. ALL parameters
        (hand-eye, camera box dimensions, floor height, display toggles)
        come from default.yaml[acquisition]: the rerun-viz/ project config.yaml provides
        ONLY assets/code (URDF, meshes, collision_fr5.py), never numeric values —
        if default.yaml[acquisition] does not specify 'camera_box:' that config.yaml remains a
        convenience fallback (backward compatibility), but it is no longer the expected source."""
        # hand-eye: prefer the PIPELINE one (default.yaml[acquisition]) so the live camera
        # matches the real photo poses.
        if self.pipeline_handeye is not None:
            T_c2g = self.pipeline_handeye
            print("[viz] live camera: using the default.yaml[acquisition] hand-eye "
                  f"(euler {self.euler_order}); wrist3->flange offset calibrated at runtime")
        else:
            T_c2g = self._fallback_camera_box()["T_cam2gripper"]

        CB = self.camera_box_cfg or self._fallback_camera_box()

        # collision_fr5: camera FK + collision box of the live robot
        col.URDF_PATH = urdf_path
        col.MIN_Z_MM = self.floor_margin_mm
        col.EULER_ORDER = self.euler_order
        col.T_CAM2GRIPPER_MM = T_c2g.copy()
        col.CAM_BOX_L_MM = float(CB["length_mm"])
        col.CAM_BOX_W_MM = float(CB["width_mm"])
        col.CAM_BOX_H_MM = float(CB["height_mm"])
        col.SUPPORT_EXTENSION_MM = float(CB["support_extension_mm"])
        col.CAM_BOX_MARGIN_MM = float(CB.get("collision_margin_mm",
                                              CB.get("margin_mm", 10.0)))

        # rerun_viz: only what _log_camera_frame / _log_floor need
        viz.URDF_PATH = urdf_path
        viz.MIN_Z_MM = self.floor_margin_mm
        viz.T_CAM2GRIPPER = T_c2g.copy()
        viz.POLL_HZ = self.poll_hz
        viz.SHOW_FLOOR = self.show_floor
        viz.SHOW_CAMERA_FRAME = self.show_camera_frame
        viz.SHOW_COLLISION_BOXES = self.show_collision_boxes
        viz.CAM_PINHOLE_DISTANCE_M = self.cam_pinhole_distance_m
        # pinhole panel intrinsics: use the REAL pipeline ones if available
        viz.PINHOLE_K = self.K
        viz.PREF_W = self.img_w
        viz.PREF_H = self.img_h

    def _fallback_camera_box(self):
        """Fallback ONLY if default.yaml[acquisition] does not define 'camera_box:' (backward compat):
        reads the box dimensions (and the hand-eye, if pipeline_handeye is missing) from
        the rerun-viz/ project config.yaml."""
        import yaml
        cfg_path = os.path.join(self.viz_repo, "config.yaml")
        with open(cfg_path, encoding="utf-8") as f:
            vc = yaml.safe_load(f) or {}
        CB = dict(vc.get("camera_box", {}))
        CB["T_cam2gripper"] = np.array(CB["T_cam2gripper"], dtype=float)
        print("[viz] warning: 'camera_box:' missing in default.yaml[acquisition], using "
              f"{cfg_path} (add 'camera_box:' to default.yaml[acquisition] to remove "
              "this dependency)")
        return CB

    def start(self, centroid_mm):
        """Initialises Rerun, opens the viewer, prepares the URDF (robot hidden) and
        logs the target cube. Idempotent-safe: on error -> self.ready=False."""
        if not self.enabled:
            print("[viz] disabled (viz.enable=false)")
            return
        try:
            if self.viz_repo not in ("", None):
                src = os.path.join(self.viz_repo, "src")
                if os.path.isdir(src) and src not in sys.path:
                    sys.path.append(src)          # APPENDED: the local rerun_viz takes priority
            import rerun as rr
            import rerun.urdf as rr_urdf
            import rerun.blueprint as bp
            import rerun_viz as viz             # local (spectra/acquisition/rerun_viz.py)
            import collision_fr5 as col         # local (spectra/acquisition/collision_fr5.py)
            self._bp = bp

            urdf_path = os.path.join(self.viz_repo, "fairino_description",
                                     "urdf", "fairino5_v6.urdf")
            self._configure_viz_globals(viz, col, urdf_path)

            # URDF meshes resolved via ROS_PACKAGE_PATH -> folder containing fairino_description/
            cur = os.environ.get("ROS_PACKAGE_PATH", "")
            parts = [p for p in cur.split(os.pathsep) if p]
            if self.viz_repo not in parts:
                os.environ["ROS_PACKAGE_PATH"] = os.pathsep.join([self.viz_repo] + parts)

            viz._check_urdf_meshes(urdf_path)     # SystemExit if the STL files are missing

            self.rr, self.viz, self.col = rr, viz, col
            # initial blueprint already WITHOUT the robot (phase 1): no robot flash
            # at startup. robot_name deterministic from the URDF (<robot name=...>).
            self.robot_name = "fairino5_v6_robot"
            rr.init("spectra_acquire_viz", spawn=True,
                    default_blueprint=self._blueprint(show_robot=False))
            rr.spawn(memory_limit=self.memory_limit)
            print(f"[viz] viewer live (tetto RAM {self.memory_limit})")

            self.urdf_tree = rr_urdf.UrdfTree.from_file_path(
                urdf_path, entity_path_prefix=viz.WORLD_ROOT)
            self.urdf_tree.log_urdf_to_recording()
            self.robot_name = self.urdf_tree.name

            try:
                self.col_model = col.FR5CollisionModel(
                    urdf_path=urdf_path,
                    mesh_root=os.path.join(self.viz_repo, "fairino_description"),
                    verbose=False)
            except Exception as exc:
                print(f"[viz] warning: collision model not loaded ({exc}); "
                      "robot camera/collision box disabled")
                self.col_model = None

            if viz.SHOW_FLOOR:
                viz._log_floor(viz.MIN_Z_MM)

            self._hide_robot()                    # phase 1: robot hidden (blueprint)
            self._log_target_cube(centroid_mm)    # cube + tolerances (persistent)
            self.ready = True
            print(f"[viz] scene ready: prelude phase {self.prelude_s:.1f}s "
                  "(generated poses + target cube)")
        except SystemExit as exc:
            print(f"[viz] warning: viz disabled ({exc}); acquiring without viewer")
            self.ready = False
        except Exception as exc:
            print(f"[viz] warning: viz init failed ({exc}); acquiring without viewer")
            self.ready = False

    # ---------------------------------------------------------------- helpers
    def _base_frame(self):
        return self.rr.CoordinateFrame("base_link")

    def _blueprint(self, show_robot):
        """3D blueprint. Robot visibility is controlled with the view's content
        FILTER (not with Clear: the URDF meshes are logged 'static'
        and a temporal Clear does not hide them). By excluding its entity paths, the
        robot (meshes + camera + collision box) disappears from the view while staying
        in the data; cube/poses/floor stay visible."""
        bp = self._bp
        w = self.viz.WORLD_ROOT
        name = self.robot_name or "fairino5_v6_robot"
        contents = [f"+ /{w}/**"]
        if not show_robot:
            contents += [f"- /{w}/{name}/**",
                         f"- /{w}/camera/**",
                         f"- /{w}/collision/**"]
        view3d = bp.Spatial3DView(
            origin=f"/{w}", name="Robot", contents=contents,
            eye_controls=bp.EyeControls3D(position=[1.8, 1.8, 1.2],
                                          look_target=[0.0, 0.0, 0.45]),
        )
        return bp.Blueprint(view3d, collapse_panels=True)

    def _hide_robot(self):
        """Phase 1/3: hides robot+camera+collision box from the view (blueprint)."""
        try:
            self.rr.send_blueprint(self._blueprint(show_robot=False), make_active=True)
        except Exception:
            pass

    def _show_robot(self):
        """Phase 2: shows the animated robot."""
        try:
            self.rr.send_blueprint(self._blueprint(show_robot=True), make_active=True)
        except Exception:
            pass

    def _log_target_cube(self, centroid_mm):
        """Target cube at the centroid with half-sides = tolerances (mm->m)."""
        try:
            c = np.asarray(centroid_mm, float).ravel()[:3] / 1000.0
            half = self.target_tol_mm / 1000.0
            base = f"{self.viz.WORLD_ROOT}/target"
            self.rr.log(
                f"{base}/cube",
                self._base_frame(),
                self.rr.Boxes3D(
                    centers=[c.tolist()],
                    half_sizes=[half.tolist()],
                    colors=[COLOR_CUBE],
                    fill_mode=self.rr.components.FillMode.Solid,
                    radii=0.0015,
                    labels=["target +/- tolerances"],
                ),
                static=True,
            )
            self.rr.log(
                f"{base}/centroid",
                self._base_frame(),
                self.rr.Points3D([c.tolist()], colors=[COLOR_CENTROID],
                                 radii=POINT_RADIUS_M),
                static=True,
            )
        except Exception as exc:
            print(f"[viz] warning: target cube not logged ({exc})")

    def _log_frustum(self, entity, T_cam2base_mm, color, static=False, label=None):
        try:
            strips, cam_pos = _frustum_strips_m(
                T_cam2base_mm, self.K, self.img_w, self.img_h, self.frustum_len_m)
            self.rr.log(
                f"{entity}/frustum",
                self._base_frame(),
                self.rr.LineStrips3D(strips, colors=[color] * len(strips),
                                     radii=FRUSTUM_RADIUS_M),
                static=static,
            )
            kwargs = dict(colors=[color], radii=POINT_RADIUS_M)
            if label is not None:
                kwargs["labels"] = [label]
            self.rr.log(f"{entity}/pos", self._base_frame(),
                        self.rr.Points3D([cam_pos], **kwargs), static=static)
        except Exception:
            pass

    # -------------------------------------------------------------- phase 1
    def show_pose_preview(self, poses, safe_mask=None):
        """Logs the frustums of ALL generated poses (phase 1).

        safe_mask: optional list of bools, same order/length as 'poses'
        (e.g. from the collision guard: config_safe() for each pose). False poses
        are drawn in RED instead of the default blue -> at a glance,
        before even moving the robot, you see which poses would be
        discarded/skipped during execution."""
        if not self.ready:
            return
        try:
            base = f"{self.viz.WORLD_ROOT}/preview"
            strips_all, pts_all, strip_colors, pt_colors = [], [], [], []
            n_unsafe = 0
            for i, wp in enumerate(poses):
                T = wp.get("cam_T_base")
                if T is None:
                    continue
                safe = True if safe_mask is None else bool(safe_mask[i])
                color = COLOR_PREVIEW if safe else COLOR_PREVIEW_UNSAFE
                if not safe:
                    n_unsafe += 1
                strips, cam_pos = _frustum_strips_m(
                    T, self.K, self.img_w, self.img_h, self.frustum_len_m)
                strips_all.extend(strips)
                strip_colors.extend([color] * len(strips))
                pts_all.append(cam_pos)
                pt_colors.append(color)
            if strips_all:
                self.rr.log(f"{base}/frustums", self._base_frame(),
                            self.rr.LineStrips3D(strips_all, colors=strip_colors,
                                                 radii=FRUSTUM_RADIUS_M))
                self.rr.log(f"{base}/positions", self._base_frame(),
                            self.rr.Points3D(pts_all, colors=pt_colors,
                                             radii=POINT_RADIUS_M))
            msg = f"[viz] phase 1: showing {len(pts_all)} generated poses"
            if safe_mask is not None:
                msg += f" ({n_unsafe} unsafe, in red)"
            print(msg)
        except Exception as exc:
            print(f"[viz] warning: pose preview not logged ({exc})")

    def wait_prelude(self):
        """Waits VIZ_PRELUDE_S seconds (robot still idle)."""
        if not self.ready:
            return
        t0 = time.time()
        while time.time() - t0 < self.prelude_s:
            time.sleep(0.05)

    # -------------------------------------------------------------- phase 2
    def begin_robot_phase(self, ip):
        """Phase 2 SETUP (without showing the robot yet): opens the second
        read-only robot connection, calibrates the camera offset and starts the
        thread that animates the URDF from the live joints. The robot stays HIDDEN
        (the generated poses stay visible) until reveal_robot() is called -> so the
        robot appears ONLY when it really starts moving."""
        if not self.ready:
            return
        # second robot connection (the SDK XML-RPC ServerProxy objects are not thread-safe:
        # motion and joint reads must use separate sockets).
        try:
            from fairino import Robot
            self.robot_viz = Robot.RPC(ip)
            Robot.RPC.is_connect = True
            ok, joints0 = self._read_joints(self.robot_viz)
            if not ok:
                print("[viz] warning: the second robot connection cannot read the joints; "
                      "URDF not animated (acquisition ok)")
                self.robot_viz = None
        except Exception as exc:
            print(f"[viz] warning: second robot connection failed ({exc}); static URDF")
            self.robot_viz = None

        # Calibrate the wrist3(URDF)->flange(controller) offset ONCE: this way the
        # live camera = wrist3 @ offset @ hand_eye = TCP_controller @ hand_eye,
        # i.e. the SAME camera pose the pipeline uses for the photos (green
        # frustums). Uses collision_fr5's validate_tcp(set_tool=True).
        if (self.robot_viz is not None and self.col_model is not None
                and self.pipeline_handeye is not None):
            try:
                r = self.robot_viz.GetActualTCPPose(1)
                tcp = list(r[1]) if isinstance(r, (list, tuple)) and len(r) >= 2 else None
                if tcp is not None:
                    res = self.col_model.validate_tcp(joints0, tcp, set_tool=True)
                    off = res["T_offset"][:3, 3] * 1000.0
                    print(f"[viz] live camera calibrated: wrist3->flange offset "
                          f"[{off[0]:.1f}, {off[1]:.1f}, {off[2]:.1f}] mm "
                          f"(residual {res['d_trans_mm']:.1f}mm/{res['d_rot_deg']:.1f}deg)")
            except Exception as exc:
                print(f"[viz] warning: camera offset calibration failed ({exc}); "
                      "live camera placed on wrist3 (correct orientation, "
                      "position off by the tool offset)")

        self._stop.clear()
        self._thread = threading.Thread(target=self._viz_loop, daemon=True)
        self._thread.start()
        print("[viz] phase 2 ready (robot still hidden): it will appear at the first motion")

    def reveal_robot(self):
        """Shows the robot and hides the prelude poses. Call it at the instant
        the robot REALLY starts moving (idempotent)."""
        if not self.ready or self._revealed:
            return
        self._revealed = True
        try:
            self.rr.log(f"{self.viz.WORLD_ROOT}/preview", self.rr.Clear(recursive=True))
            self._show_robot()
            print("[viz] robot shown (motion started); prelude poses hidden")
        except Exception:
            pass

    @staticmethod
    def _read_joints(robot):
        try:
            res = robot.GetActualJointPosDegree(1)
        except Exception:
            return False, None
        if isinstance(res, int):
            return False, None
        err, joints = res
        if err == 0 and joints is not None and len(joints) >= 6:
            return True, [float(x) for x in joints[:6]]
        return False, None

    def _viz_loop(self):
        period = 1.0 / max(1, self.poll_hz)
        while not self._stop.is_set():
            t0 = time.time()
            if self.robot_viz is not None:
                ok, joints = self._read_joints(self.robot_viz)
                if ok:
                    try:
                        self.viz._log_joint_transforms(self.urdf_tree, joints)
                        if self.col_model is not None and self.viz.SHOW_CAMERA_FRAME:
                            self.viz._log_camera_frame(self.col_model, joints)
                        if self.col_model is not None and self.viz.SHOW_COLLISION_BOXES:
                            self.viz._log_collision_boxes(self.col_model, joints)
                    except Exception:
                        pass
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)

    def log_photo_pose(self, T_cam2base_mm, tag):
        """Pins the frustum of the FINAL pose (where the photo is about to be taken):
        it stays visible for the rest of the session."""
        if not self.ready:
            return
        self._n_photo += 1
        entity = f"{self.viz.WORLD_ROOT}/photo_poses/p{tag}"
        self._log_frustum(entity, T_cam2base_mm, COLOR_PHOTO,
                          static=True, label=str(tag))

    # -------------------------------------------------------------- phase 3
    def finalize(self):
        """Stops the thread, hides the robot; all acquired poses
        + the target cube remain. The viewer stays open."""
        if not self.ready:
            return
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        #self._hide_robot()
        if self.robot_viz is not None:
            try:
                self.robot_viz.CloseRPC()
            except Exception:
                pass
            self.robot_viz = None
        print(f"[viz] phase 3: robot hidden; {self._n_photo} acquired poses "
              "shown + target cube. Viewer still open.")
