# calibration_before_HSI — hand-eye calibration, Fairino FR5 + RealSense D456

Estimates `cam2gripper` (camera → flange) with **manual drag-mode acquisition only**.
No starting `cam2gripper` is needed. All parameters are in `config.yaml`.

## Structure

```
calibration_before_HSI/
├── main.py                 # single entry point
├── config.yaml             # all parameters
├── requirements.txt
├── src/
│   ├── config.py           # loads config.yaml, resolves paths
│   ├── common.py           # poses, chessboard detection, intrinsics loading
│   ├── acquisition.py      # drag-mode acquisition (calibration + validation)
│   ├── intrinsics.py       # OpenCV intrinsic calibration
│   ├── hand_eye.py         # hand-eye calibration
│   ├── validation.py       # metrics X / Y
│   ├── pose_diagnostics.py # rotational observability of the poses
│   ├── rs_setup.py         # RealSense exposure/gain + frame averaging
│   └── fairino/Robot.py    # Fairino Python SDK
└── data/
    ├── input/calibration/  # imgXXXX.png + posaXXXX.txt + intrinsics_realsense.json
    ├── input/validation/   # imgXXXX.png + posaXXXX.txt (only)
    └── output/
        ├── calibration/    # cam2gripper.txt, matrici_4x4.txt, gripper2base.txt, kept_images.txt, intrinsics_used.json
        ├── intrinsics/     # intrinsics_opencv.json
        └── validation/     # validation_report.csv, overlay_X/, overlay_Y/
```

## Install

```bash
pip install -r requirements.txt
```

## Procedure

**Before starting:** the active tool on the controller must be **tool 0** (bare flange).
The acquisition checks it and refuses to start otherwise.

1. **Acquisition** (~30 views): `python main.py calibration acquire`
   Move the robot by hand, frame the chessboard (keep it still), press **ENTER** to save; **q/ESC** to quit.
   Views where the chessboard is not detected are not saved. Re-running continues the numbering.
   Vary the orientation a lot (tilt in all directions + roll around the optical axis).
   The factory intrinsics are saved once in `input/calibration/intrinsics_realsense.json`.
2. **Intrinsics** — choose `intrinsics.source` in `config.yaml`:
   - `realsense`: factory values (default, nothing to do);
   - `opencv`: run `python main.py intrinsics` first (target: mean reprojection < ~0.3 px);
   - `file`: a JSON `{"K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "D": [[k1,k2,p1,p2,k3]]}` in `intrinsics.file`.
     From MATLAB: `K = cameraParams.Intrinsics.K` (not the transposed `IntrinsicMatrix`),
     subtract 1 from cx and cy (MATLAB is 1-based),
     `D = [RadialDistortion(1) RadialDistortion(2) TangentialDistortion(1) TangentialDistortion(2) RadialDistortion(3)]`
     (k3 = 0 if only 2 radial coefficients were estimated).
3. **(optional) Diagnosis:** `python main.py diagnose` — checks the rotational diversity of the poses.
4. **Hand-eye:** `python main.py calibration calibrate` → `data/output/calibration/cam2gripper.txt`.
   Views with PnP reprojection error > `calibration.max_reproj_err_px` are excluded (tune it in `config.yaml`).
5. **Validation:** `python main.py validation full` (or `acquire` then `validate`).
   ~20 NEW drag-mode views. **Nothing but images and poses is written in `input/validation`**: the
   intrinsics come from the same `intrinsics.source` used by the calibration (same K, D for both),
   and a warning is printed if they differ from `output/calibration/intrinsics_used.json`.

## Metrics to look at

| Step | Metric | Good if |
|---|---|---|
| Intrinsics (opencv) | mean reprojection error | < 1 px (sub-pixel) |
| Hand-eye | std chessboard→base (self-check) | a few mm |
| Validation | 3D error mm, metric **Y** | compatible with the task requirements |

- **X** (self-consistency): board pose in base estimated from the validation images themselves.
- **Y** (out-of-sample, stricter): board pose in base estimated from the calibration images.
  The chessboard must **not move** between calibration and validation.

## Conventions

Fairino poses in mm and degrees, Euler **ZYX** (`R = Rz@Ry@Rx`), TCP = bare flange (tool 0).
Intrinsics are tied to the resolution (1280×800): images at a different resolution are rejected.
