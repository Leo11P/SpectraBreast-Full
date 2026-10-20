#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py - hand-eye pipeline, Fairino FR5 + RealSense D456 (drag-mode acquisition only)
======================================================================================

Commands:

  CALIBRATION
    python main.py calibration acquire     drag mode: move the robot, ENTER = save view
    python main.py calibration calibrate   hand-eye from data/input/calibration
    python main.py calibration full        acquire, then calibrate

  INTRINSICS (optional, OpenCV)
    python main.py intrinsics              K, D from the calibration images
                                           -> data/output/intrinsics/intrinsics_opencv.json

  DIAGNOSIS
    python main.py diagnose                rotational observability of the calibration poses

  VALIDATION
    python main.py validation acquire      drag mode on NEW views (no intrinsics written)
    python main.py validation validate     metrics X / Y
    python main.py validation full         acquire, then validate

Options:
  --config PATH    alternative YAML (default: config.yaml next to this file)
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import config as cfgmod  # noqa: E402


def banner(txt):
    print("\n" + "=" * 64)
    print(f"  {txt}")
    print("=" * 64)


def do_calibration(cfg, mode):
    if mode in ("acquire", "full"):
        import acquisition
        banner("CALIBRATION - drag-mode acquisition")
        acquisition.run(cfg, "calibration")
    if mode in ("calibrate", "full"):
        import hand_eye
        banner("CALIBRATION - hand-eye")
        hand_eye.run(cfg)


def do_intrinsics(cfg):
    import intrinsics
    banner("INTRINSIC CALIBRATION (OpenCV)")
    intrinsics.run(cfg)


def do_diagnose(cfg):
    import pose_diagnostics as pdg
    banner("ROTATIONAL OBSERVABILITY (calibration poses)")
    poses = pdg.load_poses(cfg["_paths"]["cal_in"])
    if len(poses) < 3:
        raise SystemExit(f"Found {len(poses)} poses (need >= 3).")
    print(pdg.format_report(pdg.analyze(poses)))


def do_validation(cfg, mode):
    if mode in ("acquire", "full"):
        import acquisition
        banner("VALIDATION - drag-mode acquisition")
        acquisition.run(cfg, "validation")
    if mode in ("validate", "full"):
        import validation
        banner("VALIDATION - metrics")
        validation.run(cfg)


def main():
    ap = argparse.ArgumentParser(
        prog="main.py", description="Hand-eye pipeline (Fairino FR5 + RealSense D456)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Commands:", 1)[1])
    ap.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")
    sub.add_parser("calibration").add_argument("mode", choices=["acquire", "calibrate", "full"])
    sub.add_parser("intrinsics")
    sub.add_parser("diagnose")
    sub.add_parser("validation").add_argument("mode", choices=["acquire", "validate", "full"])
    args = ap.parse_args()

    cfg = cfgmod.load(args.config)
    P = cfg["_paths"]
    print(f"[config] {args.config}")
    print(f"[config] intrinsics source: {cfg['intrinsics']['source']}")
    print(f"[config] calibration in/out: {P['cal_in']} -> {P['cal_out']}")
    print(f"[config] validation  in/out: {P['val_in']} -> {P['val_out']}")

    if args.cmd == "calibration":
        do_calibration(cfg, args.mode)
    elif args.cmd == "intrinsics":
        do_intrinsics(cfg)
    elif args.cmd == "diagnose":
        do_diagnose(cfg)
    else:
        do_validation(cfg, args.mode)


if __name__ == "__main__":
    main()
