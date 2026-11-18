"""Move the Fairino FR5 so that the CAMERA reaches a list of target camera poses.

The targets (poses.zip or a folder of pose_XXXX.txt) are camera poses in the
robot base frame, one per file: "x y z rx ry rz" in metres / radians, with
rx, ry, rz fixed-axis XYZ angles (R = Rz @ Ry @ Rx, the Fairino convention).
They are converted to flange poses with the hand-eye matrix camera2ee.npy
(T_ee_cam, metres):

    T_base_ee = T_base_cam @ inv(T_ee_cam)

then solved with the controller IK (closest solution to the previous pose,
so the arm does not flip between poses) and executed with MoveJ.

Run from the repo root:
    python goto_camera_poses.py --poses poses.zip --camera2ee camera2ee.npy
    python goto_camera_poses.py --dry-run          # IK only, no motion
    python goto_camera_poses.py --vel 10           # runs by itself (default)
    python goto_camera_poses.py --confirm          # Enter before each move
    python goto_camera_poses.py --only 1 5 26      # subset (1-based)

Rerun viewer: FR5 URDF following the live joints (second read-only
connection) plus the target camera frustums (intrinsics.npy).
"""
import os
import sys
import time
import zipfile
import argparse
import threading

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as Rot

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ACQ_DIR = os.path.join(REPO_ROOT, "spectra", "acquisition")
sys.path.insert(0, ACQ_DIR)
from acquire import RobotIO  # noqa: E402

EULER = "xyz"  # extrinsic X-Y-Z = Fairino rx, ry, rz


# ---------------------------------------------------------------- pose utils
def load_camera_poses(path):
    """Return [(name, T_base_cam 4x4, metres)] sorted by file name."""
    entries = []
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for n in sorted(zf.namelist()):
                if n.lower().endswith(".txt"):
                    entries.append((os.path.basename(n), zf.read(n).decode("utf-8")))
    else:
        for n in sorted(os.listdir(path)):
            if n.lower().endswith(".txt"):
                with open(os.path.join(path, n), "r", encoding="utf-8") as f:
                    entries.append((n, f.read()))
    poses = []
    for name, txt in entries:
        v = np.array(txt.split(), dtype=float)
        if v.size != 6:
            raise ValueError(f"{name}: expected 6 values, got {v.size}")
        poses.append((name, vec_to_T(v)))
    if not poses:
        raise SystemExit(f"no pose_*.txt found in {path}")
    return poses


def vec_to_T(v):
    """[x y z (m), rx ry rz (rad)] -> 4x4."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler(EULER, v[3:6]).as_matrix()
    T[:3, 3] = v[:3]
    return T


def T_to_desc(T):
    """4x4 (m) -> Fairino desc_pos [x y z (mm), rx ry rz (deg)]."""
    rpy = Rot.from_matrix(T[:3, :3]).as_euler(EULER, degrees=True)
    return [float(x) for x in np.r_[T[:3, 3] * 1000.0, rpy]]


def desc_to_T(d):
    """Fairino desc_pos [mm, deg] -> 4x4 (m)."""
    T = np.eye(4)
    T[:3, :3] = Rot.from_euler(EULER, d[3:6], degrees=True).as_matrix()
    T[:3, 3] = np.asarray(d[:3], dtype=float) / 1000.0
    return T


def pose_error(Ta, Tb):
    """(position error mm, rotation error deg)."""
    dp = np.linalg.norm(Ta[:3, 3] - Tb[:3, 3]) * 1000.0
    dr = np.degrees(Rot.from_matrix(Ta[:3, :3].T @ Tb[:3, :3]).magnitude())
    return dp, dr


def _unpack(res):
    """fairino calls return either an int error or (err, data)."""
    if isinstance(res, int):
        return res, None
    return res[0], res[1]


def solve_ik(rpc, desc, q_ref):
    """Controller IK, solution closest to q_ref. Returns (err, joints)."""
    try:
        err, q = _unpack(rpc.GetInverseKinRef(0, desc, q_ref))
    except AttributeError:  # older SDK without GetInverseKinRef
        err, q = _unpack(rpc.GetInverseKin(0, desc, -1))
    return err, (None if q is None else [float(x) for x in q[:6]])


# -------------------------------------------------------------------- viewer
class LiveRobotViz:
    """Rerun viewer with the FR5 URDF following the real joints."""

    def __init__(self, ip, floor_z_mm, poll_hz=20):
        import rerun as rr
        import rerun.urdf as rr_urdf
        import rerun_viz as viz
        from fairino import Robot

        urdf = os.path.join(ACQ_DIR, "fairino_description", "urdf", "fairino5_v6.urdf")
        os.environ["ROS_PACKAGE_PATH"] = os.pathsep.join(
            [ACQ_DIR] + [p for p in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep) if p])
        viz._check_urdf_meshes(urdf)

        rr.init("goto_camera_poses", spawn=False)
        rr.spawn(memory_limit="512MB")
        self.tree = rr_urdf.UrdfTree.from_file_path(urdf, entity_path_prefix=viz.WORLD_ROOT)
        self.tree.log_urdf_to_recording()
        viz._log_floor(floor_z_mm)
        self.rr, self.viz = rr, viz

        self.rpc = Robot.RPC(ip)              # read-only connection for the viewer
        Robot.RPC.is_connect = True
        self.period = 1.0 / max(1, poll_hz)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def log_targets(self, poses, K, img_wh):
        """Target camera frustums (static), base frame in metres."""
        rr = self.rr
        for name, T in poses:
            path = f"{self.viz.WORLD_ROOT}/targets/{os.path.splitext(name)[0]}"
            rr.log(path, rr.Transform3D(translation=T[:3, 3], mat3x3=T[:3, :3]), static=True)
            rr.log(path, rr.Pinhole(image_from_camera=K, resolution=list(img_wh),
                                    camera_xyz=rr.ViewCoordinates.RDF,
                                    image_plane_distance=0.03), static=True)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.time()
            try:
                res = self.rpc.GetActualJointPosDegree(1)
                if not isinstance(res, int) and res[0] == 0:
                    self.viz._log_joint_transforms(self.tree, [float(x) for x in res[1][:6]])
            except Exception:
                pass
            time.sleep(max(0.0, self.period - (time.time() - t0)))

    def close(self):
        self._stop.set()
        self._thread.join(timeout=1.0)
        try:
            self.rpc.CloseRPC()
        except Exception:
            pass


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "default.yaml"))
    ap.add_argument("--poses", default=os.path.join(REPO_ROOT, "poses.zip"),
                    help="zip or folder with pose_XXXX.txt (camera poses, m / rad)")
    ap.add_argument("--camera2ee", default=os.path.join(REPO_ROOT, "camera2ee.npy"),
                    help="4x4 hand-eye T_ee_cam (m), ee = flange")
    ap.add_argument("--intrinsics", default=os.path.join(REPO_ROOT, "intrinsics.npy"),
                    help="3x3 K, only for the viewer frustums")
    ap.add_argument("--img-size", type=int, nargs=2, default=[1024, 768], metavar=("W", "H"))
    ap.add_argument("--tool", type=int, default=0,
                    help="tool frame for MoveJ (0 = flange, matches camera2ee)")
    ap.add_argument("--user", type=int, default=0, help="user frame (0 = base)")
    ap.add_argument("--vel", type=float, default=None, help="speed %% (default: config)")
    ap.add_argument("--only", type=int, nargs="+", default=None, help="1-based indices to run")
    ap.add_argument("--max-jump", type=float, default=120.0,
                    help="skip a pose if a joint would move more than this [deg] (with --confirm: ask instead)")
    ap.add_argument("--dry-run", action="store_true", help="IK only, no motion")
    ap.add_argument("--confirm", action="store_true", help="wait for Enter before each move")
    ap.add_argument("--no-confirm", action="store_true", help=argparse.SUPPRESS)  # legacy, now default
    ap.add_argument("--dwell", type=float, default=1.0, help="pause after each move [s]")
    ap.add_argument("--no-viz", action="store_true", help="do not open the Rerun viewer")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        acfg = (yaml.safe_load(f) or {})["acquisition"]
    rcfg = acfg["robot"]
    ip = rcfg.get("ip", "192.168.58.2")
    vel = args.vel if args.vel is not None else float(rcfg.get("vel", 20))
    if int(rcfg.get("tool", 0)) != args.tool:
        print(f"[goto] note: config tool={rcfg.get('tool')} ignored, using tool={args.tool} "
              f"(camera2ee is relative to the flange)")

    # --camera2ee may be a folder (e.g. .../camera_parameters) holding the .npy files
    if os.path.isdir(args.camera2ee):
        cam_dir = args.camera2ee
        args.camera2ee = os.path.join(cam_dir, "camera2ee.npy")
        if not os.path.exists(args.intrinsics):
            args.intrinsics = os.path.join(cam_dir, "intrinsics.npy")
    if not os.path.isfile(args.camera2ee):
        raise SystemExit(f"camera2ee not found: {args.camera2ee}")

    # camera targets -> flange targets
    T_ee_cam = np.load(args.camera2ee)
    assert T_ee_cam.shape == (4, 4), "camera2ee must be 4x4"
    T_cam_ee = np.linalg.inv(T_ee_cam)
    cam_poses = load_camera_poses(args.poses)
    if args.only:
        cam_poses = [cam_poses[i - 1] for i in args.only]
    targets = [(name, T_bc, T_bc @ T_cam_ee) for name, T_bc in cam_poses]
    print(f"[goto] {len(targets)} camera poses from {args.poses}")

    robot = RobotIO(ip)
    viewer = None
    if not args.no_viz:
        try:
            floor_z = float((acfg.get("collision") or {}).get("floor_margin_mm", 15.0))
            poll_hz = int((acfg.get("viz") or {}).get("poll_hz", 20))
            viewer = LiveRobotViz(ip, floor_z, poll_hz)
            K = np.load(args.intrinsics) if os.path.exists(args.intrinsics) else None
            if K is not None:
                viewer.log_targets(cam_poses, K, args.img_size)
        except (Exception, SystemExit) as exc:
            print(f"[viz] disabled ({exc}); moving without viewer")
            viewer = None

    try:
        robot.reset_errors()
        q_start = [float(x) for x in robot.get_joints()]
        print(f"[goto] start joints: {np.round(q_start, 3).tolist()}  vel={vel}%")

        # 1) IK for every target before moving anything
        plan, q_ref = [], q_start
        for i, (name, _, T_be) in enumerate(targets, 1):
            desc = T_to_desc(T_be)
            err, q = solve_ik(robot.rpc, desc, q_ref)
            if err != 0 or q is None:
                print(f"  [{i:2d}] {name}: IK FAILED (code {err})  flange={np.round(desc, 1).tolist()}")
                plan.append((name, desc, None))
                continue
            jump = float(np.max(np.abs(np.subtract(q, q_ref))))
            print(f"  [{i:2d}] {name}: flange={np.round(desc, 1).tolist()}  "
                  f"q={np.round(q, 2).tolist()}  max dq={jump:.1f}deg")
            plan.append((name, desc, q))
            q_ref = q
        n_bad = sum(q is None for _, _, q in plan)
        if n_bad:
            print(f"[goto] {n_bad} unreachable pose(s) will be skipped")
        if args.dry_run:
            print("[goto] dry run, no motion")
            return

        # 2) execute
        q_prev = q_start
        for i, (name, desc, q) in enumerate(plan, 1):
            if q is None:
                continue
            jump = float(np.max(np.abs(np.subtract(q, q_prev))))
            tag = f"[{i}/{len(plan)}] {name}"
            if jump > args.max_jump and not args.confirm:
                print(f"{tag} SKIPPED: joint jump {jump:.0f}deg > --max-jump {args.max_jump:.0f}")
                continue
            if args.confirm:
                warn = f"  !! joint jump {jump:.0f}deg" if jump > args.max_jump else ""
                input(f"{tag} Enter -> MoveJ{warn}  (Ctrl-C to abort) ")
            else:
                print(f"{tag} MoveJ")
            err = robot.move_j(q, args.tool, args.user, vel)
            if err != 0:
                print(f"  MoveJ failed: code {err}, safety={robot.safety_code()} -> stop")
                break
            time.sleep(args.dwell)
            q_prev = [float(x) for x in robot.get_joints()]
            # check reached flange pose vs target
            e, fl = _unpack(robot.rpc.GetActualToolFlangePose(1))
            if e == 0 and fl is not None:
                dp, dr = pose_error(desc_to_T(fl), desc_to_T(desc))
                print(f"  reached: flange err {dp:.2f} mm / {dr:.2f} deg")
            else:
                print(f"  reached: joints {np.round(q_prev, 3).tolist()}")
        print("[goto] done")
    except KeyboardInterrupt:
        print("\n[goto] aborted by user")
        try:
            robot.rpc.StopMotion()
        except Exception:
            pass
    finally:
        if viewer is not None:
            viewer.close()
        robot.close()


if __name__ == "__main__":
    main()