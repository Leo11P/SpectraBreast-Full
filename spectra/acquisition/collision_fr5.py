# -*- coding: utf-8 -*-
"""
collision_fr5.py
================
REAL collision checking for the Fairino FR5 (fairino5_v6) with an Intel
RealSense D456 camera mounted on the flange through the SupportoCIRCOLARE adapter.

Unlike the "light" geometric check of the old HandEyePipeline.py (which only
checked the camera box against reach/height), here the FR5 URDF is loaded, the
forward kinematics is computed from the 6 joint angles, and a
collision-checking library (python-fcl) computes the real intersections between:
  - all robot links with each other (self-collision, non-adjacent pairs);
  - the camera+adapter box against all links;
  - every robot part against the table plane (MIN_Z height constraint).

CONVENTIONS (aligned with HandEyePipeline.py)
  - Input: 6 joint angles in DEGREES (like GetActualJointPosDegree).
  - Internally everything is in METRES and radians (URDF/FCL standard); the
    mm/degree values of the pipeline are converted once, below.
  - The hand-eye calibration "gripper" == bare flange == wrist3_link
    (confirmed: TCP at zero on the controller during calibration). Therefore
    T_CAM2GRIPPER is already "camera with respect to wrist3_link" and is applied
    directly on top of the URDF FK.

Dependencies:  numpy, trimesh, python-fcl
    pip install numpy trimesh python-fcl

Author: (for Leonardo / SpectraBreast)
"""

import os
import math
import itertools
import xml.etree.ElementTree as ET

import numpy as np
import trimesh
import fcl


# ============================================================================
# CONFIGURATION
# ============================================================================

# --- Paths ----------------------------------------------------------------
# FR5 URDF. MESH_ROOT is the folder that CONTAINS 'meshes/' (i.e. the
# fairino_description folder). If MESH_ROOT=None it is derived from the
# URDF path (urdf/ and meshes/ are siblings inside fairino_description).
URDF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fairino_description", "urdf", "fairino5_v6.urdf")
MESH_ROOT = None
PACKAGE_PREFIX = "package://fairino_description/"

# --- Floor / table constraint ---------------------------------------------
# No robot part (link or camera) may go below this height,
# measured in the base frame (base z = 0 = robot mounting plane).
MIN_Z_MM = 120.0

# Links EXCLUDED from the floor constraint because they are below MIN_Z by
# construction (fixed structure around the base: bolted down or
# pivot housing). Measured on the URDF, z_min in EVERY configuration:
#     base_link      z_min = 0   mm   (mounting plane)
#     shoulder_link  z_min = 74  mm   (housing above the base)
#     upperarm_link  z_min = 94  mm   (shoulder pivot housing)
# Applying 120 mm to these three would always give a violation. The elbow
# area is still covered by forearm_link, which IS subject to the constraint.
# Only the base is excluded (it is the mounting plane, z=0 by construction).
FLOOR_IGNORE_LINKS = {"base_link"}

# PER-LINK height threshold (mm). Links with a structurally low housing have
# a threshold equal to their structural height (minus a small margin so the
# always-present housing does not trigger false positives). This way the housing is
# tolerated, but a DIVE of the link below that height (e.g. elbow towards the table)
# is still caught. The other links use MIN_Z_MM.
#   shoulder_link  housing ~74 mm  -> threshold 69
#   upperarm_link  housing ~94 mm  -> threshold 89
FLOOR_PER_LINK_MM = {"shoulder_link": 69.0, "upperarm_link": 89.0}

# --- Camera + adapter (from HandEyePipeline.py) -----------------------------
# Camera (optical centre) -> flange (wrist3_link) transform. Translation
# in mm; converted to metres below.
T_CAM2GRIPPER_MM = np.array([
    [-9.991e-01,  9.000e-04,  4.150e-02, -11.93],
    [ 4.150e-02,  3.000e-03,  9.991e-01,  80.46],
    [ 8.000e-04,  1.000e+00, -3.000e-03,  -9.66],
    [ 0.0,        0.0,        0.0,         1.0 ],
])

# D456 physical size (mm): 124 (X) x 26 (Y) x 29 (optical Z).
CAM_BOX_L_MM = 124.0
CAM_BOX_W_MM = 26.0
CAM_BOX_H_MM = 29.0
# The support extends from the flange down to the camera: the box is
# extended along optical -Z (towards the flange) by this amount (mm).
SUPPORT_EXTENSION_MM = 80.0
# Safety margin added on EVERY side (mm). Reduced from 10 -> 5.
CAM_BOX_MARGIN_MM = 5.0

# Links that do NOT conflict with the camera by construction: the
# camera+adapter box (extended towards the flange) wraps the whole wrist group,
# so it is ALWAYS in contact with wrist3 (mount), wrist2 and wrist1. These are
# the same pairs MoveIt disables as "always in collision".
# Meaningful camera collisions remain with forearm/upperarm/shoulder/base.
CAM_IGNORE_LINKS = {"wrist3_link", "wrist2_link", "wrist1_link"}

# --- Self-collision -------------------------------------------------------
# Link pairs to ignore besides the adjacent ones (adjacent = connected
# by a joint, hence always "in contact"). Filled after an empirical test:
# this URDF shows no false positives between non-adjacent links, so it
# stays empty. Add pairs here if you see fixed spurious contacts.
SELF_PAIR_IGNORE = set()  # e.g. {("wrist1_link", "wrist3_link")}

# Fairino controller Euler convention (for the TCP self-check): ZYX.
EULER_ORDER = "ZYX"


# ============================================================================
# MATH UTILITIES
# ============================================================================

def _rpy_to_R(r, p, y):
    """RPY URDF (fixed-axis): R = Rz(y) @ Ry(p) @ Rx(r)."""
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _euler_deg_to_R(rx, ry, rz):
    """Fairino controller Euler angles (degrees). ZYX -> R = Rz @ Ry @ Rx."""
    rx, ry, rz = map(math.radians, (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]])
    Ry = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]])
    Rz = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]])
    return (Rz @ Ry @ Rx) if EULER_ORDER == "ZYX" else (Rx @ Ry @ Rz)


def _rot_axis(axis, theta):
    """Rotation of theta (rad) about axis (Rodrigues)."""
    a = np.asarray(axis, float)
    n = np.linalg.norm(a)
    if n < 1e-12:
        return np.eye(3)
    x, y, z = a / n
    c, s = math.cos(theta), math.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,   x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C,   y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C],
    ])


def _T(xyz, R):
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = xyz
    return M


def _rotation_angle(R):
    """Angle (deg) of the rotation R."""
    val = (np.trace(R) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, val))))


# ============================================================================
# COLLISION MODEL
# ============================================================================

class FR5CollisionModel:
    """
    Loads the FR5 URDF, builds the FCL models once, and provides
    check_config(joints_deg) to check a joint configuration.
    """

    def __init__(self, urdf_path=URDF_PATH, mesh_root=MESH_ROOT, verbose=True):
        self.urdf_path = urdf_path
        if mesh_root is None:
            # the urdf is in <fairino_description>/urdf/ ; meshes/ is its sibling.
            mesh_root = os.path.dirname(os.path.dirname(os.path.abspath(urdf_path)))
        self.mesh_root = mesh_root
        self.verbose = verbose

        self._parse_urdf()
        self._load_link_geometries()
        self._build_camera_geometry()
        self._build_pair_lists()

        if verbose:
            print(f"[FR5CollisionModel] {len(self.links)} links loaded, "
                  f"{len(self.self_pairs)} self-collision pairs, "
                  f"camera vs {len(self.cam_link_targets)} links.")

    # ---------------------------------------------------------------- URDF
    def _parse_urdf(self):
        root = ET.parse(self.urdf_path).getroot()

        # Link -> collision mesh file (identity origin in this URDF, but
        # read anyway for robustness).
        self.link_mesh = {}       # name -> (filepath, origin_4x4)
        for l in root.findall("link"):
            name = l.get("name")
            col = l.find("collision")
            src = col if col is not None else l.find("visual")
            mesh = src.find("geometry/mesh") if src is not None else None
            if mesh is None:
                continue
            fn = mesh.get("filename").replace(PACKAGE_PREFIX, "")
            o = src.find("origin")
            oxyz = [float(v) for v in (o.get("xyz", "0 0 0")).split()] if o is not None else [0, 0, 0]
            orpy = [float(v) for v in (o.get("rpy", "0 0 0")).split()] if o is not None else [0, 0, 0]
            self.link_mesh[name] = (os.path.join(self.mesh_root, fn),
                                    _T(oxyz, _rpy_to_R(*orpy)))

        # Joints in chain order.
        self.joints = []          # dict: name,type,parent,child,xyz,rpy,axis,lower,upper
        for j in root.findall("joint"):
            o = j.find("origin")
            a = j.find("axis")
            lim = j.find("limit")
            self.joints.append(dict(
                name=j.get("name"),
                type=j.get("type"),
                parent=j.find("parent").get("link"),
                child=j.find("child").get("link"),
                xyz=[float(v) for v in (o.get("xyz", "0 0 0")).split()] if o is not None else [0, 0, 0],
                rpy=[float(v) for v in (o.get("rpy", "0 0 0")).split()] if o is not None else [0, 0, 0],
                axis=[float(v) for v in (a.get("xyz", "0 0 1")).split()] if a is not None else [0, 0, 1],
                lower=float(lim.get("lower")) if lim is not None and lim.get("lower") else None,
                upper=float(lim.get("upper")) if lim is not None and lim.get("upper") else None,
            ))

        self.actuated = [j["name"] for j in self.joints if j["type"] in ("revolute", "prismatic", "continuous")]
        # Order of the links along the chain (base_link + children, in order).
        self.links = ["base_link"] + [j["child"] for j in self.joints]
        # Adjacency (pairs connected by a joint).
        self._adjacent = {frozenset((j["parent"], j["child"])) for j in self.joints}

    # ------------------------------------------------------------ geometry
    def _load_link_geometries(self):
        """For each link: FCL BVH (for collision) + np vertices (for height)."""
        self.fcl_link = {}        # name -> fcl.CollisionObject (BVH)
        self.link_vertices = {}   # name -> (N,3) vertices in the link frame
        for name, (path, origin) in self.link_mesh.items():
            mesh = trimesh.load(path, force="mesh")
            V = np.asarray(mesh.vertices, dtype=np.float64)
            Fc = np.asarray(mesh.faces, dtype=np.int64)
            # apply the <collision> origin (identity here) to the vertices
            V = (origin[:3, :3] @ V.T).T + origin[:3, 3]

            m = fcl.BVHModel()
            m.beginModel(len(V), len(Fc))
            m.addSubModel(V, Fc)
            m.endModel()
            self.fcl_link[name] = fcl.CollisionObject(m, fcl.Transform())
            self.link_vertices[name] = V

    def _build_camera_geometry(self):
        """Camera+adapter box as an fcl.Box, exactly replicating the
        HandEyePipeline.py geometry (extended towards the flange, inflated by the
        margin). Everything converted to metres."""
        self.T_cam2gripper = T_CAM2GRIPPER_MM.copy()
        self.T_cam2gripper[:3, 3] /= 1000.0   # mm -> m
        # flange(wrist3) -> TCP (tool configured on the controller). Identity if
        # TCP == flange; otherwise set by validate_tcp(set_tool=True).
        # Needed because cam2gripper is referred to the TCP, not to the bare flange.
        self.T_flange2tcp = np.eye(4)

        hx = (CAM_BOX_L_MM / 2 + CAM_BOX_MARGIN_MM) / 1000.0
        hy = (CAM_BOX_W_MM / 2 + CAM_BOX_MARGIN_MM) / 1000.0
        hz = (CAM_BOX_H_MM / 2 + CAM_BOX_MARGIN_MM) / 1000.0
        ext = SUPPORT_EXTENSION_MM / 1000.0
        z_min = -hz - ext
        z_max = +hz

        # full box dimensions and centre (in the camera frame)
        size = np.array([2 * hx, 2 * hy, (z_max - z_min)])
        self._cam_box_size = size
        self._cam_center_local = np.array([0.0, 0.0, (z_min + z_max) / 2.0])

        # 8 corners in the camera frame (for the height constraint), as in the pipeline
        corners = []
        for sx in (-hx, hx):
            for sy in (-hy, hy):
                for sz in (z_min, z_max):
                    corners.append([sx, sy, sz, 1.0])
        self._cam_corners_local = np.array(corners).T   # 4x8

        box = fcl.Box(*size)
        self.fcl_cam = fcl.CollisionObject(box, fcl.Transform())

    def _build_pair_lists(self):
        """Link-link pairs (non-adjacent, not ignored) and target links for
        the camera."""
        self.self_pairs = []
        for a, b in itertools.combinations(self.links, 2):
            if frozenset((a, b)) in self._adjacent:
                continue
            if (a, b) in SELF_PAIR_IGNORE or (b, a) in SELF_PAIR_IGNORE:
                continue
            if a not in self.fcl_link or b not in self.fcl_link:
                continue
            self.self_pairs.append((a, b))

        self.cam_link_targets = [n for n in self.links
                                 if n in self.fcl_link and n not in CAM_IGNORE_LINKS]

    def camera_pose_in_flange(self):
        """T_camera_optical in the wrist3_link frame (metres)."""
        return self.T_flange2tcp @ self.T_cam2gripper

    def camera_box_in_flange(self):
        """Camera+adapter box in the wrist3_link frame (for Rerun).

        Returns (T_center_4x4, half_extents_m) with the same geometry used
        by FCL in check_config().
        """
        T_cam = self.camera_pose_in_flange()
        T_center = T_cam @ _T(self._cam_center_local, np.eye(3))
        half = self._cam_box_size * 0.5
        return T_center, half

    def camera_box_in_base(self, joints_deg):
        """Camera+adapter box in the base frame (for visualisation).

        Returns (T_center_4x4, half_extents_m) with the same geometry used
        by FCL in check_config().
        """
        q = np.deg2rad(np.asarray(joints_deg, float).ravel())
        Tl = self.fk(q)
        T_cam2base = Tl["wrist3_link"] @ self.camera_pose_in_flange()
        T_center = T_cam2base @ _T(self._cam_center_local, np.eye(3))
        half = self._cam_box_size * 0.5
        return T_center, half

    # ------------------------------------------------------------------ FK
    def fk(self, q_rad):
        """Forward kinematics. q_rad: iterable of len == number of actuated joints.
        Returns dict link_name -> 4x4 (in the base frame)."""
        q_rad = np.asarray(q_rad, float).ravel()
        if q_rad.shape[0] != len(self.actuated):
            raise ValueError(f"Expected {len(self.actuated)} joints, got {q_rad.shape[0]}")
        Tl = {"base_link": np.eye(4)}
        qi = 0
        for j in self.joints:
            if j["type"] in ("revolute", "continuous"):
                Rj = _rot_axis(j["axis"], q_rad[qi]); qi += 1
                Tj = _T(j["xyz"], _rpy_to_R(*j["rpy"])) @ _T([0, 0, 0], Rj)
            elif j["type"] == "prismatic":
                d = q_rad[qi]; qi += 1
                axis = np.asarray(j["axis"], float)
                Tj = _T(j["xyz"], _rpy_to_R(*j["rpy"])) @ _T(axis / np.linalg.norm(axis) * d, np.eye(3))
            else:  # fixed
                Tj = _T(j["xyz"], _rpy_to_R(*j["rpy"]))
            Tl[j["child"]] = Tl[j["parent"]] @ Tj
        return Tl

    # --------------------------------------------------------------- checks
    def check_config(self, joints_deg, check_limits=True):
        """
        Checks a joint configuration.

        Parameters
        ----------
        joints_deg : sequence of 6 angles in DEGREES (order j1..j6).
        check_limits : if True, flags joints outside the URDF limits.

        Returns a dict with:
          is_safe           : bool  (True if no violation)
          self_collisions   : [(linkA, linkB, penetration_mm), ...]
          camera_collisions : [(link, penetration_mm), ...]
          floor_violations  : [(name, z_min_mm), ...]   ('camera' included
                              if the box goes below MIN_Z)
          limit_violations  : [(joint, value_deg, lower_deg, upper_deg), ...]
        """
        q = np.deg2rad(np.asarray(joints_deg, float).ravel())
        Tl = self.fk(q)

        # placing the FCL objects
        for name, obj in self.fcl_link.items():
            T = Tl[name]
            obj.setTransform(fcl.Transform(T[:3, :3].copy(), T[:3, 3].copy()))
        T_cam2base = Tl["wrist3_link"] @ self.T_flange2tcp @ self.T_cam2gripper
        Tcb = T_cam2base @ _T(self._cam_center_local, np.eye(3))
        self.fcl_cam.setTransform(fcl.Transform(Tcb[:3, :3].copy(), Tcb[:3, 3].copy()))

        req = fcl.CollisionRequest(num_max_contacts=8, enable_contact=True)

        # 1) self-collision
        self_col = []
        for a, b in self.self_pairs:
            res = fcl.CollisionResult()
            if fcl.collide(self.fcl_link[a], self.fcl_link[b], req, res) and res.is_collision:
                depth = max((c.penetration_depth for c in res.contacts), default=0.0)
                self_col.append((a, b, depth * 1000.0))

        # 2) camera vs links
        cam_col = []
        for name in self.cam_link_targets:
            res = fcl.CollisionResult()
            if fcl.collide(self.fcl_cam, self.fcl_link[name], req, res) and res.is_collision:
                depth = max((c.penetration_depth for c in res.contacts), default=0.0)
                cam_col.append((name, depth * 1000.0))

        # 3) floor constraint (analytic, min z of vertices / corners)
        floor = []
        for name in self.links:
            if name in FLOOR_IGNORE_LINKS or name not in self.link_vertices:
                continue
            V = self.link_vertices[name]
            zmin = (Tl[name][:3, :3] @ V.T).T[:, 2] + Tl[name][2, 3]
            zmin = zmin.min() * 1000.0
            thr = FLOOR_PER_LINK_MM.get(name, MIN_Z_MM)
            if zmin < thr:
                floor.append((name, zmin))
        cam_corners_base = (T_cam2base @ self._cam_corners_local)[:3, :].T
        cam_zmin = cam_corners_base[:, 2].min() * 1000.0
        if cam_zmin < MIN_Z_MM:
            floor.append(("camera", cam_zmin))

        # 4) joint limits
        lim = []
        if check_limits:
            for j, val in zip(self.joints, q):
                if j["type"] not in ("revolute", "prismatic"):
                    continue
                if j["lower"] is not None and val < j["lower"] - 1e-6:
                    lim.append((j["name"], math.degrees(val), math.degrees(j["lower"]), math.degrees(j["upper"])))
                elif j["upper"] is not None and val > j["upper"] + 1e-6:
                    lim.append((j["name"], math.degrees(val), math.degrees(j["lower"]), math.degrees(j["upper"])))

        is_safe = not (self_col or cam_col or floor or lim)
        return dict(is_safe=is_safe, self_collisions=self_col,
                    camera_collisions=cam_col, floor_violations=floor,
                    limit_violations=lim)

    # ----------------------------------------------------- self-check TCP
    def validate_tcp(self, joints_deg, tcp_pose_mm_deg, tol_mm=3.0, tol_deg=1.0,
                     set_tool=False):
        """
        One-off check of the consistency between the URDF FK and the TCP pose
        read from the controller FOR THE SAME joint configuration.

        - joints_deg      : 6 joints in degrees (GetActualJointPosDegree).
        - tcp_pose_mm_deg : [x,y,z,rx,ry,rz] from GetActualTCPPose (mm, degrees).

        If the residual is ~0 -> bare flange confirmed and URDF FK ~ controller FK.
        If there is a fixed offset -> it is the tool; T_offset is printed
        (in the flange frame) so it can be compensated.
        Returns a dict with residuals and T_offset.
        """
        Tl = self.fk(np.deg2rad(joints_deg))
        T_fk = Tl["wrist3_link"]

        p = np.asarray(tcp_pose_mm_deg, float)
        T_ctrl = np.eye(4)
        T_ctrl[:3, :3] = _euler_deg_to_R(p[3], p[4], p[5])
        T_ctrl[:3, 3] = p[:3] / 1000.0

        d_trans = np.linalg.norm(T_fk[:3, 3] - T_ctrl[:3, 3]) * 1000.0
        R_res = T_fk[:3, :3].T @ T_ctrl[:3, :3]
        d_rot = _rotation_angle(R_res)
        T_offset = np.linalg.inv(T_fk) @ T_ctrl   # tool in the flange frame

        ok = (d_trans <= tol_mm) and (d_rot <= tol_deg)
        if set_tool:
            # apply the measured tool to the model: from now on the camera is
            # placed on wrist3 -> tool -> cam2gripper (correct physical position)
            self.T_flange2tcp = T_offset.copy()
            if self.verbose:
                off = T_offset[:3, 3] * 1000.0
                print(f'[validate_tcp] tool applied to the collision model: '
                      f'[{off[0]:.1f}, {off[1]:.1f}, {off[2]:.1f}] mm along the flange axis')
        if self.verbose:
            print(f"[validate_tcp] translation residual = {d_trans:.2f} mm, "
                  f"rotation = {d_rot:.2f} deg  -> "
                  f"{'OK (bare flange, consistent FK)' if ok else 'OFFSET detected'}")
            if not ok:
                off_mm = T_offset[:3, 3] * 1000.0
                print(f"               tool offset (flange frame) = "
                      f"[{off_mm[0]:.1f}, {off_mm[1]:.1f}, {off_mm[2]:.1f}] mm, "
                      f"rot {_rotation_angle(T_offset[:3, :3]):.2f} deg")
        return dict(ok=ok, d_trans_mm=d_trans, d_rot_deg=d_rot, T_offset=T_offset)


# ============================================================================
# PRINT UTILITIES
# ============================================================================

def format_report(res):
    """Makes the check_config dict readable."""
    if res["is_safe"]:
        return "SAFE — no collision, height OK, joints within limits."
    lines = ["NOT SAFE:"]
    for a, b, d in res["self_collisions"]:
        lines.append(f"  self-collision: {a} <-> {b}  (penetration {d:.1f} mm)")
    for name, d in res["camera_collisions"]:
        lines.append(f"  camera <-> {name}  (penetration {d:.1f} mm)")
    for name, z in res["floor_violations"]:
        lines.append(f"  height: {name} at z={z:.1f} mm (< {MIN_Z_MM:.0f} mm)")
    for jn, val, lo, hi in res["limit_violations"]:
        lines.append(f"  joint limit: {jn}={val:.1f} deg (range [{lo:.1f}, {hi:.1f}])")
    return "\n".join(lines)


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    import sys
    urdf = sys.argv[1] if len(sys.argv) > 1 else URDF_PATH
    if not os.path.isfile(urdf):
        print(f"URDF not found: {urdf}\n"
              f"Pass the path as an argument:  python collision_fr5.py <path/fairino5_v6.urdf>\n"
              f"or set URDF_PATH at the top of the file.")
        sys.exit(1)
    model = FR5CollisionModel(urdf_path=urdf)

    tests = {
        "home (arm up)":            [0, -90, 0, -90, 0, 0],
        "q=0 (arm horizontal)":      [0, 0, 0, 0, 0, 0],
        "elbow towards the table":   [0, -30, 120, 0, 0, 0],
        "strongly folded":          [0, -120, 140, 0, 0, 0],
        "j2 out of limits":         [0, -300, 0, 0, 0, 0],
    }
    for name, q in tests.items():
        res = model.check_config(q)
        print(f"\n### {name}  q={q}")
        print(format_report(res))

    # Self-check usage example (with a fake TCP pose = exact FK):
    # with a real reading you would use robot.GetActualTCPPose(1)[1].
    print("\n--- TCP self-check (demo with TCP = exact FK) ---")
    q_demo = [10, -80, 40, -50, 30, 15]
    T = model.fk(np.deg2rad(q_demo))["wrist3_link"]
    # rebuild a fake ZYX TCP reading from the FK values
    def R_to_zyx_deg(R):
        sy = math.hypot(R[0, 0], R[1, 0])
        rz = math.atan2(R[1, 0], R[0, 0])
        ry = math.atan2(-R[2, 0], sy)
        rx = math.atan2(R[2, 1], R[2, 2])
        return math.degrees(rx), math.degrees(ry), math.degrees(rz)
    rx, ry, rz = R_to_zyx_deg(T[:3, :3])
    tcp = [T[0, 3]*1000, T[1, 3]*1000, T[2, 3]*1000, rx, ry, rz]
    model.validate_tcp(q_demo, tcp)