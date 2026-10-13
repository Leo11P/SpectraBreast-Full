# -*- coding: utf-8 -*-
"""
rerun_viz.py
=============
Rerun logging helpers for the FR5 robot, used by pipeline_viz.py during an
acquisition: URDF joint transforms, flange camera frame + pinhole, camera
collision box, floor grid, URDF mesh check.

Module-level values below are defaults; pipeline_viz.py overwrites them from
default.yaml[acquisition] before logging. Does NOT import rerun at module
level (lazy imports inside the functions).
"""

import math
import os
import xml.etree.ElementTree as ET

import numpy as np

# ============ defaults (overwritten by pipeline_viz from default.yaml[acquisition]) ============
URDF_PATH = None
POLL_HZ = 15
SHOW_FLOOR = True
SHOW_CAMERA_FRAME = True
SHOW_COLLISION_BOXES = True
MIN_Z_MM = 120.0
T_CAM2GRIPPER = np.eye(4)
PINHOLE_K = None                  # 3x3 intrinsics for the live pinhole (None = generic guess)
CAM_PINHOLE_DISTANCE_M = 0.12
COLLISION_EDGE_RADIUS_M = 0.0015  # collision box border (Boxes3D)
FLOOR_LINE_RADIUS_M = -1.5        # floor grid lines (negative = screen px)
PREF_W, PREF_H, PREF_FPS = 1280, 800, 15

PACKAGE_PREFIX = "package://fairino_description/"
JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
WORLD_ROOT = "world"  # root of the 3D entities in the Rerun viewer


def _resolve_mesh_paths(urdf_path):
    """Returns a unique list of absolute paths to the meshes referenced in the URDF."""
    root = ET.parse(urdf_path).getroot()
    mesh_root = os.path.dirname(os.path.dirname(os.path.abspath(urdf_path)))
    seen = set()
    paths = []
    for mesh in root.iter("mesh"):
        fn = mesh.get("filename")
        if not fn:
            continue
        rel = fn.replace(PACKAGE_PREFIX, "")
        full = os.path.normpath(os.path.join(mesh_root, rel))
        if full not in seen:
            seen.add(full)
            paths.append(full)
    return paths


def _check_urdf_meshes(urdf_path):
    """Checks that the STL meshes exist; otherwise exits with a clear message."""
    missing = [p for p in _resolve_mesh_paths(urdf_path) if not os.path.isfile(p)]
    if missing:
        readme = os.path.join(os.path.dirname(urdf_path), "..", "README.md")
        raise SystemExit(
            "Missing URDF meshes (needed by the Rerun visualisation):\n"
            + "\n".join(f"  - {p}" for p in missing[:5])
            + (f"\n  ... and {len(missing) - 5} more" if len(missing) > 5 else "")
            + f"\n\nCopy the STL files into fairino_description/meshes/ as described in:\n"
            f"  {os.path.normpath(readme)}"
        )


def _log_floor(min_z_mm, extent_m=1.5):
    import rerun as rr

    z = float(min_z_mm) / 1000.0
    s = extent_m
    grid = []
    n = 10
    step = 2 * s / n
    for i in range(n + 1):
        t = -s + i * step
        grid.append([[t, -s, z], [t, s, z]])
        grid.append([[-s, t, z], [s, t, z]])
    rr.log(
        f"{WORLD_ROOT}/floor",
        # Grid is defined in base_link coordinates; declare the frame so it
        # resolves to the view target (same fix as arucos/collision box).
        rr.CoordinateFrame("base_link"),
        rr.LineStrips3D(grid, colors=[[180, 180, 180, 120]], radii=FLOOR_LINE_RADIUS_M),
        static=True,
    )


def _log_joint_transforms(urdf_tree, joints_deg):
    import rerun as rr

    joint_map = {
        j.name: j
        for j in urdf_tree.joints()
        if j.joint_type in ("revolute", "continuous", "prismatic")
    }
    for name, angle_deg in zip(JOINT_NAMES, joints_deg):
        joint = joint_map.get(name)
        if joint is None:
            continue
        if joint.joint_type == "prismatic":
            value = float(angle_deg) / 1000.0
        else:
            value = math.radians(float(angle_deg))
        # Temporal (non static): /tf_static from log_urdf_to_recording() already
        # holds the zero-pose tree; logging static joint updates conflicts with it.
        rr.log(f"{WORLD_ROOT}/transforms", joint.compute_transform(value), static=False)


def _load_pinhole_intrinsics():
    """fx, fy, cx, cy, width, height (px) for rr.Pinhole."""
    w, h = float(PREF_W), float(PREF_H)
    fx = fy = 0.9 * w
    cx, cy = w / 2.0, h / 2.0
    if PINHOLE_K is not None:
        K = np.asarray(PINHOLE_K, dtype=float)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
    return fx, fy, cx, cy, w, h


def _log_collision_boxes(col_model, joints_deg):
    """Camera+adapter box used by the collision checker (FCL)."""
    import rerun as rr

    del joints_deg  # pose follows wrist3_link via the URDF TF tree
    T_center, half = col_model.camera_box_in_flange()
    entity = f"{WORLD_ROOT}/collision/camera_box"

    rr.log(
        entity,
        rr.Transform3D(
            translation=T_center[:3, 3],
            mat3x3=T_center[:3, :3],
            # T_center is the box pose expressed IN wrist3_link (child-in-parent),
            # i.e. the default ParentFromChild relation. NOT from_parent=True,
            # which would invert it (same convention as the URDF joint transforms).
            parent_frame="wrist3_link",
            child_frame="camera_collision_box",
        ),
    )
    rr.log(
        entity,
        # The box lives in the camera_collision_box frame defined by the
        # Transform3D above (wrist3_link -> camera_collision_box). Declare it so
        # Rerun resolves the box to base_link; without CoordinateFrame the
        # Boxes3D falls into the implicit tf#/world/collision/camera_box frame,
        # which is disconnected -> "No transform path ... to base_link".
        rr.CoordinateFrame("camera_collision_box"),
        rr.Boxes3D(
            half_sizes=half.tolist(),
            colors=[255, 165, 0, 55],
            fill_mode=rr.components.FillMode.Solid,
            radii=COLLISION_EDGE_RADIUS_M,
            labels=["camera collision"],
        ),
    )


def _log_camera_frame(col_model, joints_deg):
    """Flange-mounted camera: optical frame + red pinhole frustum."""
    import rerun as rr

    del joints_deg  # pose follows wrist3_link via the URDF TF tree
    T_cam = col_model.camera_pose_in_flange()
    fx, fy, cx, cy, w, h = _load_pinhole_intrinsics()
    entity = f"{WORLD_ROOT}/camera"

    rr.log(
        entity,
        rr.Transform3D(
            translation=T_cam[:3, 3],
            mat3x3=T_cam[:3, :3],
            # T_cam is the camera pose expressed IN wrist3_link (child-in-parent),
            # which is exactly the default ParentFromChild relation used by the
            # URDF joint transforms. from_parent=True would invert it and place
            # the frame ~10 cm off (and inconsistent with the point cloud).
            parent_frame="wrist3_link",
            child_frame="camera_optical",
        ),
    )
    rr.log(
        entity,
        rr.Pinhole(
            focal_length=[fx, fy],
            principal_point=[cx, cy],
            width=w,
            height=h,
            camera_xyz=rr.ViewCoordinates.RDF,  # OpenCV / RealSense: +X right, +Y down, +Z fwd
            image_plane_distance=CAM_PINHOLE_DISTANCE_M,
            color=[255, 0, 0, 255],
            parent_frame="wrist3_link",
            child_frame="camera_optical",
        ),
    )
