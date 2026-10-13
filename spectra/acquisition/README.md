# spectra/acquisition/acquire.py — Layered Image Acquisition with Visual Servoing

## Overview

`acquire.py` acquires multi-view images of an ArUco-marked object using a Fairino FR5 robot equipped with a wrist-mounted Intel RealSense camera. The resulting dataset is intended for downstream 3D reconstruction pipelines such as MASt3R-SfM or VGGT.
The acquisition strategy combines **offline viewpoint planning** with **online visual servoing**. Planned viewpoints provide complete geometric coverage of the object, while visual servoing compensates for calibration inaccuracies, robot repeatability errors and small perception errors before each image is captured.

## Design Principle

The **Fairino controller is the only source of truth for robot kinematics**.
Whenever the current flange pose is required, the software queries the controller through `GetForwardKin`. No URDF-based forward kinematics or external robot model is used to estimate the robot state.

---

# Pipeline Overview

The acquisition pipeline consists of five stages.

## 1. Bootstrap
The robot remains stationary while several independent camera frames are acquired.
For every frame, the pose of each visible ArUco marker is estimated using `solvePnP`. Marker observations are aggregated **per marker ID using the median**, making the estimate robust against frame-to-frame noise and occasional outliers.
From these observations the pipeline estimates:

- the object centroid;
- the average surface normal;
- the object reference expressed in the robot base frame.

This reference remains fixed throughout the acquisition.

---

## 2. Viewpoint Planning
Camera poses are generated around the estimated centroid according to the acquisition layers defined in `configs/default.yaml` → `acquisition:`.
Each layer specifies only the acquisition geometry (distance, tilt, azimuth sampling and optional roll). Every generated viewpoint is oriented with a look-at strategy so that the optical axis always points towards the estimated centroid.
Each candidate pose is accepted only if:

- the centroid remains inside the image boundaries;
- the resulting flange orientation is sufficiently far from wrist gimbal lock;
- the native Fairino inverse kinematics finds a valid solution.

The inverse kinematics solution is stored and later executed directly through `MoveJ`.

---

## 3. Robot Motion
The robot reaches each planned viewpoint using **MoveJ**.
Joint-space interpolation is preferred over Cartesian interpolation because image acquisition consists of independent stop-and-capture poses rather than continuous scanning trajectories.
Using MoveJ avoids trajectory failures caused by wrist singularities that may appear during Cartesian interpolation even when both endpoint poses are individually reachable.

NB: 6-DOF robot, non-redundant, map joint values -> EE pose its deterministic, which means given 6 joint values, it exists one and only EE pose. NO AMBIGUITY! FK is deterministic.
On the contrary, if I do the opposite, going from cartesian space to joint space, we can get more solution: what in italian we call "gomito giù/su".

---

# Closed-loop Visual Servoing -- discrete PBVS: discrete position-based visual servoing 

Once a planned viewpoint has been reached, image acquisition does **not** start immediately.
Instead, the system enters a closed-loop visual servoing stage whose objective is to remove residual positioning errors.
Unlike the global planner, which relies entirely on geometry, the visual servo relies exclusively on the current camera image.
Each iteration performs the following steps:

1. Acquire a new image.
2. Detect all visible ArUco markers.
3. Estimate their 3D poses.
4. Compute the current centroid.
5. Measure the centroid offset from the image centre.
6. Decide whether a correction is required.

If the projection error is already below the configured pixel tolerance, the viewpoint is accepted.
Otherwise, the desired camera position is recomputed while preserving approximately the current viewing distance from the object.
Rather than executing the complete correction, only a fraction of the estimated displacement is applied. This damping factor prevents overshooting and oscillations caused by noisy pose estimates.
The correction is executed through a short local `MoveL`, after which the process repeats using a newly acquired image.

The loop terminates when either:

- the centroid falls below the pixel tolerance;
- the maximum number of iterations is reached.

The servo therefore implements a genuine closed-loop controller where every correction is computed from fresh visual feedback.

---

# Correction Criteria

Visual corrections are intentionally conservative. A correction is executed only when:

- at least `min_markers` markers are visible;
- the centroid estimate is considered reliable.

Empirically, attempting to servo from a single marker—especially near the image borders—produced unstable `solvePnP` estimates, causing the controller to chase measurement noise instead of reducing the actual error.
If fewer than the required markers are visible, no correction is applied. If no markers are detected at all, image acquisition for that viewpoint is skipped.

---

# Image Acquisition

Only after the visual servo confirms the viewpoint does the pipeline wait for the configured settling time (`dwell_s`).
The waiting time is intentionally applied **after** convergence rather than before it, avoiding unnecessary delays whenever another correction is still required.
The final image is then obtained by averaging several camera frames to reduce sensor noise.

---

# Error Handling and Recovery
Several independent criteria prevent collecting unreliable data.

## Missing Target
If no markers are detected after reaching a viewpoint, the image is discarded and the acquisition proceeds to the next pose.

## Servo Non-convergence
If the maximum number of iterations is reached, the acquisition proceeds using the best corrected pose instead of endlessly attempting convergence.
The objective is robust acquisition rather than mathematically perfect centering.

## Pointing Validation
Immediately after the first executed viewpoint, the actual optical axis—computed from the controller's native forward kinematics—is compared with the direction towards the estimated centroid.
If the angular error exceeds the configured threshold, the acquisition is aborted immediately.
A large initial pointing error almost always indicates an incorrect transformation chain (hand-eye calibration, Euler convention or reference frame definition), making all subsequent viewpoints unreliable.

## Motion Failures
Three consecutive `MoveJ` failures trigger a circuit breaker that stops the acquisition.
Persistent motion failures generally indicate controller errors or collisions rather than isolated unreachable poses.

---

# Configuration
The acquisition behaviour is controlled through `configs/default.yaml` → `acquisition:`.

The most relevant parameters are:

- `robot.euler_order`: Euler convention used by the controller.
- `robot.vel`: MoveJ execution speed.
- `robot.dwell_s`: settling time after servo convergence.
- `robot.pointing_gate_deg`: maximum allowed initial pointing error.
- `hand_eye.T_cam2gripper`: camera-to-flange transformation.
- `servo.*`: visual servo parameters.
- `layers`: acquisition geometry.

---

# Commands

Run from the **repo root** (paths are resolved from the cwd, exactly like `spectra vision`).
The script is standalone: it does not import the `spectra` package (no torch/MASt3R needed on the robot PC).

```bash
python spectra/acquisition/acquire.py --verify
python spectra/acquisition/acquire.py --dry-run
python spectra/acquisition/acquire.py                      # sample_name from the config
python spectra/acquisition/acquire.py --sample SB030       # override sample_name
python spectra/acquisition/acquire.py --sample SB030 --overwrite
python spectra/acquisition/acquire.py --reset
```

Recommended workflow:

1. Run `--verify` after changing camera intrinsics, hand-eye calibration, Euler convention or marker size.
2. Run `--dry-run` to validate viewpoint generation.
3. Execute the complete acquisition.

---

# Configuration & output (integration with Vision)

- **One config**: `configs/default.yaml`. The `acquisition:` section holds robot/camera/servo/layers;
  the ArUco dictionary and the marker edge length are read from `vision.aruco` (single value for
  acquisition, vision and registration).
- **Output** goes straight into the folder Vision reads, `DATA/input_vision/<sample_name>/`:

```
rgb/img_0001.png ...          RGB images
poses/pose_0001.txt ...       FLANGE pose in robot base: x y z roll pitch yaw
                              (metres, radians, R = Rz(yaw) Ry(pitch) Rx(roll))
camera_parameters/            intrinsics.npy, distortions.npy, camera2ee.npy (hand-eye, metres),
                              intrinsics_source.json (where K/D came from)
depth/  manifest.json  SHA256SUMS  _discarded/     (not read by Vision)
```

  Vision composes `T_world_cam = T_world_flange @ camera2ee`, so the hand-eye is applied once.
- **Intrinsics**: by default the RealSense factory intrinsics of the running color
  stream (always matching `camera.width/height`). Set `acquisition.camera.intrinsics`
  to override them with a calibration: the folder written by
  `spectra calibrate-intrinsics`, an `intrinsics.npy`, or a legacy JSON. The override
  is rejected if it is more than 5% off the factory values (calibrated at another
  resolution). The source is recorded in `camera_parameters/intrinsics_source.json`.
- **Image/pose pairing**: each shot is saved as a pair (image + pose) or not at all. At the end
  (also after Ctrl-C) every `img_NNNN` is checked against `pose_NNNN` and vice versa; orphans and
  unreadable poses are moved to `_discarded/`, then the counts/indices are verified and SHA-256
  checksums are written to `SHA256SUMS` (`sha256sum -c SHA256SUMS` after copying to Mufasa).
- An existing sample is never overwritten silently: use another `--sample` or `--overwrite`.

---

# Verification Mode
`--verify` performs a complete consistency check without moving the robot.
It verifies:

- consistency between `GetForwardKin` and `GetActualTCPPose`;
- pixel → 3D → pixel reprojection accuracy;
- correctness of the estimated centroid in the robot base frame.

This diagnostic should always succeed before executing a real acquisition.

---

# Motion Strategy
Large robot motions are executed with `MoveJ`, while local visual-servo corrections use `MoveL`.
This separation intentionally decouples:

- **global motion planning**, responsible for safely reaching each viewpoint;
- **local image-based refinement**, responsible for correcting residual positioning errors immediately before image capture.

This combination provides the robustness of joint-space motion together with the accuracy of closed-loop visual feedback, without requiring perfect calibration or robot repeatability.
