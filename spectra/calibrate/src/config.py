# -*- coding: utf-8 -*-
"""
config.py
=========
Loads config.yaml and resolves every path against the project root (the folder
that contains config.yaml), so the folder can be moved freely.
"""

import os

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve(path):
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(ROOT, path))


def load(path=None):
    path = path or os.path.join(ROOT, "config.yaml")
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    P = cfg["paths"]
    paths = dict(
        cal_in=resolve(P["calib_input_dir"]),
        val_in=resolve(P["val_input_dir"]),
        cal_out=resolve(P["calib_output_dir"]),
        intr_out=resolve(P["intrinsics_output_dir"]),
        val_out=resolve(P["val_output_dir"]),
    )
    for d in paths.values():
        os.makedirs(d, exist_ok=True)

    paths["intr_realsense"] = os.path.join(paths["cal_in"], "intrinsics_realsense.json")
    paths["intr_opencv"] = os.path.join(paths["intr_out"], "intrinsics_opencv.json")
    paths["intr_file"] = resolve(cfg["intrinsics"].get("file"))
    paths["cam2gripper"] = os.path.join(paths["cal_out"], "cam2gripper.txt")

    cfg["_config_path"] = path
    cfg["_paths"] = paths
    return cfg
