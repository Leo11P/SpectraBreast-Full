# -*- coding: utf-8 -*-
"""Camera calibration for the robot acquisition (Fairino FR5 + RealSense):
hand-eye (camera -> flange) and intrinsics from chessboard views acquired in
drag mode, plus out-of-sample validation. CLI: ``spectra calibrate``.

Settings: ``calibration:`` in the unified config (robot and camera stream from
``acquisition.robot`` / ``acquisition.camera``). Data, fixed layout:

    <data_root>/camera_calibration/
        input/calibration/    imgXXXX.png + posaXXXX.txt (+ intrinsics_realsense.json)
        input/validation/     imgXXXX.png + posaXXXX.txt
        output/calibration/   cam2gripper.txt, matrici_4x4.txt, gripper2base.txt,
                              kept_images.txt, intrinsics_used.json
        output/intrinsics/    intrinsics_opencv.json
        output/validation/    validation_report.csv, overlay_X/, overlay_Y/
"""

from __future__ import annotations

#: `spectra calibrate` without --modality: the whole procedure, in this order.
#: Intrinsics come before the hand-eye, which uses them when
#: calibration.intrinsics.source is `opencv`.
SEQUENCE = (
    ("calibration", "acquire"),
    ("intrinsics", None),
    ("calibration", "calibrate"),
    ("validation", "acquire"),
    ("validation", "validate"),
)
#: Steps accepted after --modality (``full`` = its steps of SEQUENCE).
STEPS = {
    "calibration": ("acquire", "calibrate", "full"),
    "validation": ("acquire", "validate", "full"),
    "intrinsics": (),
}


def plan(modality=None, step=None):
    """[(modality, step)] to run. Raises ValueError on a wrong combination."""
    if modality is None:
        if step:
            raise ValueError(f"step {step!r} needs --modality (calibration | validation)")
        return list(SEQUENCE)
    if modality not in STEPS:
        raise ValueError(f"--modality must be one of {list(STEPS)}, got {modality!r}")
    if modality == "intrinsics":
        if step:
            raise ValueError("--modality intrinsics takes no step")
        return [("intrinsics", None)]
    step = step or "full"
    if step not in STEPS[modality]:
        raise ValueError(f"--modality {modality} takes one of {list(STEPS[modality])}, "
                         f"got {step!r}")
    return [s for s in SEQUENCE if s[0] == modality] if step == "full" else [(modality, step)]


def run(cfg, steps):
    """Run the planned steps on the settings dict of common.settings()."""
    for modality, step in steps:
        title = f"{modality.upper()}" + (f" - {step}" if step else "")
        print("\n" + "=" * 64 + f"\n  {title}\n" + "=" * 64)
        if step == "acquire":
            from . import acquisition
            acquisition.run(cfg, modality)
        elif modality == "intrinsics":
            from . import intrinsics
            intrinsics.run(cfg)
        elif step == "calibrate":
            from . import hand_eye
            hand_eye.run(cfg)
        else:
            from . import validation
            validation.run(cfg)
