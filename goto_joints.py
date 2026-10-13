"""Move the Fairino FR5 through a fixed list of joint configurations (MoveJ).

Uses the RobotIO wrapper from spectra/acquisition/acquire.py (official fairino
library) and the robot settings (ip, tool, user, vel) from configs/default.yaml.

Run from the repo root:
    python goto_joints.py              # asks Enter before each move
    python goto_joints.py --no-confirm # runs all moves back-to-back
    python goto_joints.py --vel 10     # override speed (%)
    python goto_joints.py --no-viz     # without the Rerun viewer

Rerun viewer: FR5 URDF (spectra/acquisition/fairino_description) animated from
the live joints, read on a SECOND read-only connection (the fairino XML-RPC
proxy is not thread-safe), using the helpers in spectra/acquisition/rerun_viz.py.
"""
import os
import sys
import time
import argparse
import threading

import yaml

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
ACQ_DIR = os.path.join(REPO_ROOT, "spectra", "acquisition")
sys.path.insert(0, ACQ_DIR)
from acquire import RobotIO  # noqa: E402

# Joint targets [J1..J6] in degrees
POSES = [
    [66.546,  -74.044, 117.687, -172.105,  -88.877, -82.596],
    [67.327,  -90.308, 127.099,  -88.724,  -33.249, 124.739],
    [73.675, -109.578, 138.704,  -79.536,  -61.214,  95.196],
    [41.587,  -74.039, 123.442, -172.105,  -88.926, -74.788],
    [76.424,  -77.979, 143.171, -181.996, -121.568, -41.438],
]


class LiveRobotViz:
    """Rerun viewer with the FR5 URDF following the real joints."""

    def __init__(self, ip, floor_z_mm, poll_hz=20):
        import rerun as rr
        import rerun.urdf as rr_urdf
        import rerun_viz as viz
        from fairino import Robot

        urdf = os.path.join(ACQ_DIR, "fairino_description", "urdf", "fairino5_v6.urdf")
        # package://fairino_description/... resolved via ROS_PACKAGE_PATH
        os.environ["ROS_PACKAGE_PATH"] = os.pathsep.join(
            [ACQ_DIR] + [p for p in os.environ.get("ROS_PACKAGE_PATH", "").split(os.pathsep) if p])
        viz._check_urdf_meshes(urdf)

        # same pattern as pipeline_viz.py (viewer used by acquire.py)
        rr.init("goto_joints", spawn=False)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(REPO_ROOT, "configs", "default.yaml"))
    ap.add_argument("--vel", type=float, default=None, help="speed %% (default: config)")
    ap.add_argument("--no-confirm", action="store_true", help="do not wait for Enter")
    ap.add_argument("--dwell", type=float, default=1.0, help="pause after each move [s]")
    ap.add_argument("--no-viz", action="store_true", help="do not open the Rerun viewer")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        acfg = (yaml.safe_load(f) or {})["acquisition"]
    rcfg = acfg["robot"]
    ip = rcfg.get("ip", "192.168.58.2")
    tool, user = int(rcfg.get("tool", 0)), int(rcfg.get("user", 0))
    vel = args.vel if args.vel is not None else float(rcfg.get("vel", 20))

    robot = RobotIO(ip)
    viewer = None
    if not args.no_viz:
        try:
            floor_z = float((acfg.get("collision") or {}).get("floor_margin_mm", 15.0))
            poll_hz = int((acfg.get("viz") or {}).get("poll_hz", 20))
            viewer = LiveRobotViz(ip, floor_z, poll_hz)
        except (Exception, SystemExit) as exc:
            print(f"[viz] disabled ({exc}); moving without viewer")
            viewer = None
    try:
        robot.reset_errors()
        print(f"[goto] start joints: {robot.get_joints()}  vel={vel}%")
        for i, q in enumerate(POSES, 1):
            if not args.no_confirm:
                input(f"[{i}/{len(POSES)}] Enter -> MoveJ {q}  (Ctrl-C to abort) ")
            else:
                print(f"[{i}/{len(POSES)}] MoveJ {q}")
            err = robot.move_j(q, tool, user, vel)
            if err != 0:
                print(f"  MoveJ failed: code {err}, safety={robot.safety_code()} -> stop")
                break
            time.sleep(args.dwell)
            print(f"  reached: {robot.get_joints()}")
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
