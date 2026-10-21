# SpectraBreast — Vision & Registration Pipeline

Contributors: **Alberto Rota**, **Leonardo Passoni**, **Anna Bicchi**

A two-stage pipeline for intraoperative margin assessment on breast specimens:

1. **Vision** — multi-view RGB → metric 3D surface, via MASt3R-SfM, scaled and
   aligned to a world frame using ArUco markers placed around the specimen.
2. **Registration** — an ENVI hyperspectral cube → registered onto that 3D
   surface, producing a spectral point cloud plus error/coverage diagnostics.

The two stages talk to each other through exactly two files, both written by
Vision and read by Registration:

| File | Meaning |
| --- | --- |
| `surface_mesh.ply` | reconstructed surface, metres, in the ArUco-aligned output frame |
| `aruco_markers_3d.json` | 3D marker corners in the same frame |

Everything is driven by one YAML config and one CLI (`spectra`). On the robot
PC the same CLI also calibrates the camera (`spectra calibrate`, §8) and
acquires the RGB views (`spectra acquire`).

---

## Table of contents

- [1. Local Setup](#1-local-setup)
- [2. Mufasa Setup](#2-mufasa-setup)
- [3. Dataset layout](#3-dataset-layout)
- [4. CLI reference](#4-cli-reference)
- [5. Running locally](#5-running-locally)
- [6. Running on Mufasa (SLURM)](#6-running-on-mufasa-slurm)
- [7. Outputs](#7-outputs)
- [8. Camera calibration (robot PC)](#8-camera-calibration-robot-pc)


---

## 1. Local Setup

Dependency management is `uv`. Install it first if needed:
<https://docs.astral.sh/uv/#installation>.

MASt3R and ASMK are **not** pip dependencies — they are cloned *inside* the repo
at fixed paths. `spectra/vision/backends/mast3r_backend.py` appends
`<repo_root>/mast3r` to `sys.path` at runtime, so the location is not
configurable.

### 1.1 Clone the repo

```bash
git clone https://github.com/Leo11P/SpectraBreast-Full.git
cd SpectraBreast-Full
```

### 1.2 Clone MASt3R and ASMK into the repo

```bash
# MASt3R — --recursive is required (pulls dust3r and croco as submodules)
git clone --recursive https://github.com/naver/mast3r.git mast3r

# ASMK — must live at mast3r/asmk (Setup_env.sh and the config expect it there)
git clone https://github.com/jenicek/asmk.git mast3r/asmk
```

Resulting layout:

```
SpectraBreast-Full/
  mast3r/
    dust3r/
      croco/
    asmk/
  spectra/
  configs/
```

Both `mast3r/` and `asmk/` are in `.gitignore` and are never committed.

### 1.3 Create the environment

```bash
uv sync --extra processing                         # core + vision/registration deps + torch (cu130 index)
uv run pip install -r mast3r/requirements.txt
uv run pip install -r mast3r/dust3r/requirements.txt
```

The `processing` extra (torch, open3d, trimesh, ...) is what vision and
registration need; the core alone is the light robot-PC install (§1.7). A plain
`uv sync` removes the extra again: always pass `--extra processing` here.

### 1.4 Build and install ASMK

ASMK ships a Cython extension that must be compiled before install:

```bash
uv run pip install cython
cd mast3r/asmk/cython && uv run cythonize *.pyx && cd ../../..
uv run pip install -e mast3r/asmk
```

### 1.5 Make `mast3r` importable outside the pipeline

The pipeline injects `mast3r/` into `sys.path` itself, so this step is not
needed to *run* the pipeline. It **is** needed for any interactive
`import mast3r` (e.g. from a Python shell or a notebook):

```bash
uv run pip install -e mast3r/ --no-deps
```

Alternatively, put it on `PYTHONPATH` permanently:

```bash
echo 'export PYTHONPATH="$HOME/SpectraBreast-Full/mast3r:$HOME/SpectraBreast-Full/mast3r/dust3r:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

### 1.6 MASt3R weights

`vision.mast3r.model_name` defaults to a Hugging Face repo ID:

```yaml
model_name: naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric
```

The weights (~2.5 GB) are loaded via `AsymmetricMASt3R.from_pretrained(...)` and
are **downloaded automatically on first run** into `$HF_HOME`. Nothing to do
manually.

To use a local checkpoint instead — e.g. on a compute node with no internet —
download it once:

```bash
mkdir -p checkpoints && cd checkpoints
wget https://download.europe.naverlabs.com/ComputerVision/MASt3R/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
cd ..
```

and point the config at the file:

```yaml
vision:
  mast3r:
    model_name: checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth
```

**Retrieval checkpoint** — only needed when `vision.mast3r.scene_graph` contains
`retrieval` (e.g. `retrieval-20-1`). With the default `scene_graph: auto` it is
never used, and `vision.mast3r.retrieval_model` can stay `null`. If you do
enable retrieval, set it to the ASMK codebook checkpoint path or the pipeline
raises at pair-construction time.

### 1.7 Robotic acquisition assets (robot PC only)

Only needed on the PC that drives the robot (`acquire.py`). Vision and
Registration never use them, so skip this on Mufasa.

The robot PC does **not** need the full environment of §1.3 (no torch, no
MASt3R): `acquire.py` never imports the `spectra` package. A plain virtual
environment with these packages is enough:

```bash
python -m venv .venv-acq
.venv-acq\Scripts\activate            # Windows  (Linux: source .venv-acq/bin/activate)
pip install "numpy>=1.24" "opencv-python>=4.8" "PyYAML>=6.0" "pyrealsense2>=2.54" \
            "trimesh>=4.0" "python-fcl>=0.7" "rerun-sdk>=0.20" "paramiko>=3.0"
```

`pyrealsense2` drives the RealSense, `trimesh` + `python-fcl` the collision
checks, `rerun-sdk` the live viewer, `paramiko` the hand-off to Mufasa. The Fairino SDK is not on PyPI: it is the
`Robot.py` copied below (pure Python, standard library only).

For the `spectra` commands the robot PC uses the repo's `uv` environment, and
what it installs follows `mufasa.enabled`:

| `mufasa.enabled` | Robot PC install | Runs here |
| --- | --- | --- |
| `true` | `uv sync --extra acquisition` (no torch, no GPU) | acquisition; vision + registration go to Mufasa |
| `false` | `uv sync --extra acquisition --extra processing` + MASt3R / ASMK (§1.3–§1.4), CUDA GPU | everything |

The heavy modules are imported only when vision / registration run on this
PC, so the light install is enough with `mufasa.enabled: true`. With `false`,
the CLI checks before anything starts (and before the robot moves) that the
`processing` packages, MASt3R and a CUDA GPU are there, and prints the command
to run if they are not.

The Fairino Python SDK and the FR5 URDF + meshes are **not** committed (they
are in `.gitignore`), so copy them into `spectra/acquisition/` by hand, e.g.
from the lab copy in `Fairino/VisuoServoing/`:

```bash
# Fairino Python SDK — only Robot.py is needed (no __init__.py, no build/)
mkdir -p spectra/acquisition/fairino
cp <FAIRINO>/fairino/Robot.py spectra/acquisition/fairino/

# FR5 URDF + meshes — keep the fairino_description/ folder name, the URDF
# references its meshes as package://fairino_description/meshes/...
mkdir -p spectra/acquisition/fairino_description/urdf spectra/acquisition/fairino_description/meshes
cp <FAIRINO>/fairino_description/urdf/fairino5_v6.urdf spectra/acquisition/fairino_description/urdf/
cp -r <FAIRINO>/fairino_description/meshes/fairino5_v6 spectra/acquisition/fairino_description/meshes/
```

Resulting layout:

```
spectra/acquisition/
  acquire.py
  ...
  fairino/
    Robot.py
  fairino_description/
    urdf/
      fairino5_v6.urdf
    meshes/
      fairino5_v6/
        base_link.STL  shoulder_link.STL  upperarm_link.STL  forearm_link.STL
        wrist1_link.STL  wrist2_link.STL  wrist3_link.STL
```

The paths are not configurable by default: `acquire.py` imports `fairino` from
its own folder, and collision checks + Rerun viz look for
`fairino_description/` there. To keep URDF + meshes elsewhere, set
`acquisition.collision.repo_dir` and `acquisition.viz.repo_dir` to the folder
that **contains** `fairino_description/`.

---

## 2. Mufasa Setup

Same dependencies, different constraints: no `sudo`, and no system-wide `pip`
on the cluster. Everything installs into your own home directory instead.
`uv` itself needs no `sudo` — it installs to `~/.local/bin` via a user-space
script — and every dependency after that goes through `uv sync` / `uv run pip`
into a project-local `.venv`, exactly as in [§1](#1-local-setup). `Setup_env.sh`
runs that whole chain for you: it is the equivalent of steps
[1.3](#13-create-the-environment)–[1.5](#15-make-mast3r-importable-outside-the-pipeline)
above, packaged into one script so nothing needs to be typed by hand on the
cluster.

### 2.1 Clone everything on the login node

The script expects `mast3r/` and `mast3r/asmk/` to already be in place, so
clone them first — same commands as [1.1](#11-clone-the-repo) and
[1.2](#12-clone-mast3r-and-asmk-into-the-repo), run on the cluster:

```bash
cd ~
git clone https://github.com/Leo11P/SpectraBreast-Full.git
cd SpectraBreast-Full
git clone --recursive https://github.com/naver/mast3r.git mast3r
git clone https://github.com/jenicek/asmk.git mast3r/asmk
```

### 2.2 Run the setup script

```bash
bash Setup_env.sh        # run on the LOGIN node, NOT via sbatch
```

`Setup_env.sh` assumes the repo is exactly at `~/SpectraBreast-Full`. It
installs `uv` if absent (no `sudo` needed), checks that `mast3r/` and
`mast3r/asmk/` exist, runs `uv sync --extra processing`, installs the MASt3R and dust3r
requirements, cythonizes and installs ASMK, then smoke-tests `torch`, CUDA
visibility and `import spectra`. This one script replaces the manual pip
commands from §1.3–§1.5.


### 2.3 MASt3R weights

Same behaviour as [1.6](#16-mast3r-weights) — the weights download
automatically into `$HF_HOME` on first run. If the node you run on has no
internet access, download the checkpoint once as described there and point
`vision.mast3r.model_name` at the local file.

---

## 3. Dataset layout

Everything lives under two roots, set in the YAML as `data_root` (default
`DATA`) and `results_root` (default `RESULTS`). Both are gitignored.

Paths are resolved **relative to the current working directory**, so always
invoke `spectra` from the repo root. Absolute paths are accepted anywhere.

### 3.1 Layout for `single`, `roi` and `sweep` modes

This is the layout the auto-derivation in `spectra/orchestrator.py` expects:

```
DATA/
├── input_vision/
│   └── <sample>/
│       ├── rgb/                                              ← RGB images
│       │   └── *.jpg | *.png | *.bmp | *.tif | *.tiff | *.webp
│       ├── poses/                                            ← optional
│       │   ├── pose_0000.txt
│       │   ├── pose_0001.txt
│       │   └── ...
│       └── camera_parameters/                                ← optional
│           ├── intrinsics.npy      (3×3)
│           ├── distortions.npy     (1×N, optional)
│           └── camera2ee.npy       (4×4 or 3×4, optional)
│
└── input_registration/
    └── <sample>/
        ├── <sample>_raw.hdr        ← ENVI header (plus its .img / .raw / .bin data file)
        └── <sample>_raw.png        ← LiveView RGB, ROI mode only
```

- If `<sample>/` has no `rgb/` subfolder, the images are read directly from
  `<sample>/`.
- The vision folder may carry a device suffix, `<sample>_<device>/` (e.g.
  `SB021_iphone`, `SB021_realsense`). An exact `<sample>/` wins; otherwise the
  suffix containing `iphone`, otherwise the first alphabetically.

### 3.2 Layout for `batch` mode

Batch discovery (`spectra/registration/batch.py`) uses the **same** layout as
§3.1: it scans every subdirectory of `DATA/input_registration/`:

```
DATA/
└── input_registration/
    └── <sample>/
        ├── <sample>_raw.hdr
        └── <sample>_raw.png        ← only when registration.batch.roi_mode: true
```

Samples are optionally filtered by `registration.batch.sample_regex`, and any
sample missing its `.hdr`, `surface_mesh.ply` or `aruco_markers_3d.json` is
skipped (with a printed reason). Mesh and ArUco JSON are read from
`RESULTS/<dd-mm-yyyy>/<sample>/vision/` for `results_date` (today by default),
falling back to the most recent dated run that has both files.

### 3.3 Pose file format

One file per image, named `pose_*.txt`, sorted lexicographically and zipped
against the sorted image list. If the number of pose files differs from the
number of images, **all** poses are ignored (Vision prints a `[WARNING]` and
runs RGB-only SfM). Each file holds **6 whitespace-separated numbers** on one
line — the **end-effector (flange)** pose in the world/robot-base frame, which
Vision composes with `camera_parameters/camera2ee.npy` (identity if absent):

```
X Y Z ROLL PITCH YAW
```

- Translation in **metres** (`translation_scale = 1.0`).
- Angles in **radians** (`angles_in_degrees = False`).
- Convention `ROLLPITCHYAW`: `R = Rz(yaw) · Ry(pitch) · Rx(roll)`.

Both `poses/` and `camera_parameters/` are fully optional — with neither, Vision
runs pure SfM and takes its metric scale from the ArUco markers alone.

### 3.4 Camera calibration layout

Fixed, under `data_root`; written and read by `spectra calibrate` (§8):

```
DATA/
└── camera_calibration/
    ├── input/
    │   ├── calibration/     imgXXXX.png + posaXXXX.txt (+ intrinsics_realsense.json)
    │   └── validation/      imgXXXX.png + posaXXXX.txt
    └── output/
        ├── calibration/     cam2gripper.txt, matrici_4x4.txt, gripper2base.txt,
        │                    kept_images.txt, intrinsics_used.json
        ├── intrinsics/      intrinsics_opencv.json
        └── validation/      validation_report.csv, overlay_X/, overlay_Y/
```

`posaXXXX.txt` holds the flange pose as read from the controller: `x,y,z,rx,ry,rz`
in mm and degrees (convention `acquisition.robot.euler_order`).



## 4. CLI reference

Installed as the `spectra` console script. All three forms are equivalent:

```bash
uv run spectra ...
uv run python -m spectra ...
uv run python -m spectra.cli ...
```

**Where it runs** is set once in the YAML, by `mufasa.enabled`:

| Command | `mufasa.enabled: true` | `mufasa.enabled: false` |
| --- | --- | --- |
| `spectra full` | acquisition here → vision + registration on Mufasa | acquisition → vision → registration here |
| `spectra vision` | on Mufasa | here (CUDA GPU required) |
| `spectra registration` | on Mufasa | here |
| `spectra acquire` | acquisition only, here | acquisition only, here |
| `spectra calibrate` | here (robot PC) | here (robot PC) |

"On Mufasa" means: upload what the stage needs (unless already there), `sbatch`
a job that runs it on a GPU node, wait, download the selected results here (see
[`remote.py`](#remotepy-mufasa-hand-off)). Every job is launched from this PC
with this PC's config.
Inside a SLURM job `mufasa.enabled` is ignored and everything runs on the node.

### `spectra vision`

MASt3R-SfM + ArUco reconstruction, here or on Mufasa (`mufasa.enabled`).

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c`, `--config` PATH | required | YAML config (unified or legacy vision-only) |
| `--sample` TEXT | from YAML | Override `sample_name` for this run |
| `-s`, `--set` KEY=VAL | — | Dotted override, repeatable |
| `--dry-run` | off | Resolve paths, print the config, run nothing |

### `spectra registration`

HSI → mesh registration, here or on Mufasa (`mufasa.enabled`). Requires Vision
outputs to exist already: on Mufasa it uses the vision run already there, or
this PC's if newer. `--mode batch` runs only here.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c`, `--config` PATH | required | Unified YAML (needs a `registration:` section) |
| `--sample` TEXT | from YAML | Override `sample_name`; also the output filename prefix |
| `--mode` TEXT | from YAML | `single` \| `roi` \| `batch` \| `sweep` |
| `-s`, `--set` KEY=VAL | — | Dotted override, repeatable |
| `--cpu` | off | Force CPU raycasting even when a GPU is available |
| `--dry-run` | off | Validate paths, print the config, run nothing |

### `spectra full`

The stages listed in `stages:` (default `["acquisition", "vision", "registration"]`)
on one sample: acquisition on this PC, then vision → registration here or on
Mufasa (`mufasa.enabled`). Drop `"acquisition"` from `stages` to re-run
vision/registration on a sample already acquired. Before the robot moves it
checks that the rest can run: SSH login + repo on Mufasa, or a CUDA GPU here.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c`, `--config` PATH | required | Unified YAML with both stage sections |
| `--sample` TEXT | from YAML | Override `sample_name` |
| `-s`, `--set` KEY=VAL | — | Dotted override, repeatable (not for `acquisition.*`) |
| `--cpu` | off | Force CPU raycasting in Registration |
| `--force-vision` | off | Re-run Vision even if its outputs exist (here; on Mufasa it always runs) |
| `--overwrite` | off | Acquisition: replace an existing acquisition for the same sample |
| `--no-viz` | off | Acquisition: disable the Rerun viewer |
| `--dry-run` | off | Print the plan (and the Mufasa run config), run nothing |

**Vision is skipped automatically** when both `RESULTS/<dd-mm-yyyy>/<sample>/vision/surface_mesh.ply`
and `RESULTS/<dd-mm-yyyy>/<sample>/vision/aruco_markers_3d.json` already exist
for `results_date` (today by default), unless the acquisition just ran. Use
`--force-vision` to override. Which stages run at all is also gated by the
`stages:` list in the YAML — `spectra vision` and `spectra registration` ignore it.

### `spectra detect`

2D ArUco detection only (no reconstruction) — quick check that markers are
visible in a folder of images.

| Flag | Default | Meaning |
| --- | --- | --- |
| `INPUT_FOLDER` | required | Folder of RGB images |
| `OUTPUT_FOLDER` | required | Receives `json/` and `annotated/` |
| `--dict` TEXT | `4x4_50` | ArUco dictionary |
| `--draw-scale` FLOAT | `2.0` | Scale of the drawn markers in `annotated/` |

### `spectra viewer`

Local 3D viewer (Gradio) for the runs under a results folder.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-r`, `--results-dir` PATH | `RESULTS` | Folder to browse |
| `--host` TEXT | `127.0.0.1` | Bind address |
| `--port` INT | `7860` | Port |
| `--share` / `--no-share` | off | Public Gradio link |
| `--no-browser` | off | Do not open the browser |

### `spectra calibrate-intrinsics`

Generic checkerboard intrinsic calibration of any image folder. Writes
`intrinsics.npy` and `distortions.npy`, the files expected in
`camera_parameters/` (§3.1). The robot camera is calibrated by
`spectra calibrate` (§8).

| Flag | Default | Meaning |
| --- | --- | --- |
| `--image-dir` PATH | `checkerboard` | Checkerboard images |
| `--output-dir` PATH | `intrinsics` | Output folder |
| `--checkerboard-cols` INT | `10` | Inner corners per row |
| `--checkerboard-rows` INT | `7` | Inner corners per column |
| `--square-size-m` FLOAT | `0.024` | Square edge in metres |

### `spectra calibrate`

Hand-eye (camera → flange) and intrinsics of the robot camera, with validation.
Robot PC only (§8). Without `--modality` it runs the whole procedure:
calibration acquisition → intrinsics → hand-eye → validation acquisition →
validation.

```bash
uv run spectra calibrate                                   # everything
uv run spectra calibrate --modality calibration acquire    # drag-mode views only
uv run spectra calibrate --modality calibration calibrate  # hand-eye on the saved views
uv run spectra calibrate --modality calibration            # = full: acquire + calibrate
uv run spectra calibrate --modality intrinsics             # OpenCV intrinsics
uv run spectra calibrate --modality validation validate    # metrics X / Y
uv run spectra calibrate --modality validation             # = full: acquire + validate
```

| Flag / argument | Default | Meaning |
| --- | --- | --- |
| `-m`, `--modality` | — (everything) | `calibration` \| `validation` \| `intrinsics` |
| `STEP` | `full` | `calibration`: `acquire` \| `calibrate` \| `full`; `validation`: `acquire` \| `validate` \| `full`; none for `intrinsics` |
| `-c`, `--config` PATH | `configs/default.yaml` | Unified YAML (needs a `calibration:` section) |

The acquisition steps need `uv sync --extra acquisition` and the Fairino SDK
(§1.7); intrinsics, hand-eye and validation are offline and run with the core
install.

### `spectra acquire`

Fairino FR5 + RealSense acquisition with visual servoing, through the `spectra`
console script. Acquisition only: it never submits anything, `spectra full`
chains vision + registration after it. Same behaviour as `acquire.py` below;
it needs `uv sync --extra acquisition` (RealSense, collision checks), not the
`processing` extra (§1.7).

```bash
uv run spectra acquire --mode verify
uv run spectra acquire --mode dry-run
uv run spectra acquire --sample SB021
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c`, `--config` PATH | `configs/default.yaml` | Unified YAML (needs an `acquisition:` section) |
| `-m`, `--mode` TEXT | `run` | `run` \| `verify` \| `dry-run` \| `reset` (see `--verify` / `--dry-run` / `--reset` below) |
| `--sample` TEXT | from YAML | Override `sample_name`; output goes to `DATA/input_vision/<sample>/` |
| `--ip` TEXT | `acquisition.robot.ip` | Robot controller IP |
| `--no-viz` | off | Disable the Rerun viewer |
| `--overwrite` | off | Replace an existing acquisition for the same sample |

### `acquire.py` (robotic acquisition)

The same acquisition as a standalone script. Runs on the robot PC without
importing the `spectra` package (no torch / MASt3R needed there, see §1.7).
Invoke it from the repo root:

```bash
python spectra/acquisition/acquire.py --config configs/default.yaml --verify
python spectra/acquisition/acquire.py --config configs/default.yaml --dry-run
python spectra/acquisition/acquire.py --config configs/default.yaml --sample SB021
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` PATH | `configs/default.yaml` | Unified YAML (needs an `acquisition:` section) |
| `--sample` TEXT | from YAML | Override `sample_name`; output goes to `DATA/input_vision/<sample>/` |
| `--ip` TEXT | `acquisition.robot.ip` | Robot controller IP |
| `--verify` | off | Check the transformation chain, robot does not move |
| `--dry-run` | off | Bootstrap + pose generation, robot does not move |
| `--no-viz` | off | Disable the Rerun viewer |
| `--overwrite` | off | Replace an existing acquisition for the same sample |
| `--reset` | off | Reset robot errors and exit |

ArUco dictionary and marker edge length are read from `vision.aruco`. The
Fairino SDK (`spectra/acquisition/fairino/`) and the FR5 URDF + meshes
(`spectra/acquisition/fairino_description/`) are not versioned: copy them there
locally before running.

### `remote.py` (Mufasa hand-off)

Runs on this PC and is what "on Mufasa" means above: `spectra full`,
`spectra vision` and `spectra registration` call it when `mufasa.enabled` is
true. On Mufasa everything lives in the repo clone `mufasa.repo_dir`, with the
same layout as here:

```
<repo_dir>/DATA/input_vision/<sample>/          inputs, kept between jobs
<repo_dir>/DATA/input_registration/<sample>/
<repo_dir>/RESULTS/<dd-mm-yyyy>/<sample>/
    vision/  registration/                      results (all of them stay here)
    _job/    run_config.yaml, job.slurm, slurm-<jobid>.out
```

Inputs go by sample name and only what is missing or changed is uploaded (same
size and modification time = already there):

| Stage | Input on Mufasa |
| --- | --- |
| vision | `DATA/input_vision/<sample>/` mirrored from this PC (stale files removed), checked against `SHA256SUMS` |
| registration | `DATA/input_registration/<sample>/` from this PC; if it is not here, the copy already on Mufasa is used |
| registration without vision | the vision run already on Mufasa, or this PC's when it is the one the pipeline would pick (same date, or newer) |

Then it submits `spectra full` for those stages with `sbatch` (vision always
re-runs: `RESULTS/` on Mufasa is shared between jobs), waits for the job and
downloads only `mufasa.pull_patterns` of the stages that ran — by default the
registration Excel reports and the spectral Rerun recording
(`registration/*.xlsx`, `*.rrd`, `*.rbl`) — into the local
`RESULTS/<dd-mm-yyyy>/<sample>/`. In `spectra full`, without the HSI cube
(here or on Mufasa) only Vision runs. CLI overrides (`--sample`, `--date`,
`--mode`, `--set`) travel with the job. Paths set by hand in the YAML
(`vision.input.*`, `registration.paths.*`) are refused: on Mufasa every path is
derived from `sample_name`. Use it by hand to (re)submit a sample:

```bash
python spectra/acquisition/remote.py --sample SB021
python spectra/acquisition/remote.py --sample SB021 --stages registration
python spectra/acquisition/remote.py --sample SB021 --no-wait   # submit and return
python spectra/acquisition/remote.py --sample SB021 --dry-run   # what would be uploaded
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` PATH | `configs/default.yaml` | Unified YAML (needs `mufasa.enabled: true`) |
| `--sample` TEXT | from YAML | Sample name |
| `--stages` ... | `stages` in the YAML | `vision` and/or `registration` |
| `--no-wait` | off | Submit and return, do not wait for the job |
| `--no-pull` | off | Do not download the results |
| `--dry-run` | off | Print what would be uploaded and the run config, send nothing |

Requirements: this PC on the Polimi LAN or VPN, an SSH key authorised on
Mufasa (`mufasa.ssh_key_path`; `null` asks for the password), the repo cloned
at `mufasa.repo_dir` with `Setup_env.sh` done. The QOS allow one job at a time
(`MaxSubmit=1`).

The marker edge length is set only in `configs/default.yaml`
(`vision.aruco.marker_edge_length_mm`; any other config or `--set` that sets it
is rejected). Each submitted run carries its own copy of the local config,
`RESULTS/<dd-mm-yyyy>/<sample>/_job/run_config.yaml` on Mufasa (without the
`acquisition:`, `calibration:` and `mufasa:` sections, so the job never submits
again, and with `results_date` pinned to the submission day), and the job points
`SPECTRA_DEFAULT_CONFIG` at it: for that run it is the default config, so the
`configs/default.yaml` of the repo on Mufasa is neither read nor modified and
the two never need to be kept in sync by hand.

---

## 5. Running locally

With `mufasa.enabled: false` and the `processing` extra installed (§1.3).
Requires a CUDA GPU for MASt3R-SfM (checked before anything runs). Registration falls back to CPU raycasting (`--cpu`),
Vision does not.

Either prefix everything with `uv run`:

```bash
cd ~/SpectraBreast-Full
uv run spectra full --config configs/default.yaml
```

or activate the venv once:

```bash
source .venv/bin/activate
spectra full --config configs/default.yaml
```

---

## 6. Running on Mufasa (SLURM)

The maintained route on Mufasa is the native `uv` environment. One-time setup
is covered in [§2](#2-mufasa-setup) — this section assumes that's already done.

### 6.1 Submitting a job

```bash
mkdir -p logs
sbatch run_spectra.sh
```

`run_spectra.sh` requests:

| Resource | Value |
| --- | --- |
| `--partition` | `jobs` |
| `--qos` | `gpuwide` |
| `--gres` | `gpu:4g.20gb:1` (one 20 GB MIG slice) |
| `--cpus-per-task` | `8` |
| `--mem` | `64G` |
| `--time` | `12:00:00` |
| `--output` / `--error` | `./logs/run_spectra-%j.{out,err}` |


### 6.2 Choosing what the job runs

`run_spectra.sh` is a thin wrapper. The stage is a single line near the bottom
of the file — edit it:

```bash
uv run --extra processing spectra vision --config configs/default.yaml
#uv run --extra processing spectra registration --config configs/default.yaml
#uv run --extra processing spectra full --config configs/default.yaml
```

Inside the job `mufasa.enabled` is ignored and the `acquisition` stage is
skipped (no robot on the cluster).

### 6.3 Monitoring

```bash
squeue -u $USER
tail -f logs/run_spectra-<jobid>.out
tail -f logs/run_spectra-<jobid>.err
scancel <jobid>
```

### 6.4 Getting results back

`RESULTS/` stays in the home directory on the cluster:

```bash
# from your local machine
rsync -avz mufasa:~/SpectraBreast-Full/RESULTS/<dd-mm-yyyy>/SB021/ ./RESULTS/<dd-mm-yyyy>/SB021/
```

Pull just the small artifacts (Excel reports, PNGs, JSON) when the point clouds
aren't needed — a full run's `RESULTS/` can be several GB.

---

## 7. Outputs

Everything goes under `RESULTS/<dd-mm-yyyy>/<sample>/`, one subfolder per
stage. The date folder is the run day (or `results_date` when set); re-running
the same stage on the same day overwrites it.

```
RESULTS/
└── <dd-mm-yyyy>/
    ├── _batch/batch_summary.xlsx               aggregate Excel (batch mode only)
    └── SB021/
        ├── vision/
        │   ├── surface_mesh.ply                    ← consumed by Registration
        │   ├── aruco_markers_3d.json               ← consumed by Registration
        │   ├── cloud.ply                           fused RGB point cloud (CloudCompare)
        │   ├── cloud.npz                            points / colors / confidence
        │   ├── surface.ply                         surface point cloud, with normals
        │   ├── surface.npz                         surface arrays
        │   ├── camera_poses_output_frame.npy       [V,4,4] camera poses, output frame
        │   ├── intrinsics_original_frame.npy       [V,3,3] K at original resolution
        │   ├── intrinsics_network_frame.npy        [V,3,3] K at network resolution
        │   ├── sim3_to_output_frame.npy            4×4 Sim(3), backend → output frame
        │   ├── aruco_bundle_adjustment.json        per-marker/per-view reproj RMSE (if BA on)
        │   ├── reconstruction_metadata.json        run summary — read this first
        │   ├── timings.json                        per-phase wall-clock
        │   ├── run.yaml / run.json                 fully resolved config for this run
        │   ├── aruco_detections/
        │   │   ├── json/                           per-image 2D detections
        │   │   └── annotated/                      images with markers drawn
        │   └── rerun/
        │       └── spectra.rrd                     Rerun recording (if rerun.save_rrd)
        │
        ├── registration/
        │   ├── spectral_pointcloud_SB021.ply             spectral cloud (export.ply)
        │   ├── spectral_pointcloud_SB021_xyz.bin         float32 memmap, (N,3)
        │   ├── spectral_pointcloud_SB021_spectra.bin     float32 memmap, (N,bands)
        │   ├── spectral_pointcloud_SB021_wavelengths.npy wavelengths in nm
        │   ├── spectral_pointcloud_SB021_meta.npy        int64 [N,3,N,bands] — memmap shapes
        │   ├── spectral.rrd                              Rerun: band + PCA views (if rerun.enabled)
        │   ├── spectral.rbl                              Rerun layout (if rerun.save_rbl)
        │   ├── SB021_registration_errors.xlsx            2D/3D errors + Timings sheet
        │   ├── SB021_registration.png                    HSI + render overlay with ArUco
        │   ├── SB021_coverage.png                        HSI → mesh coverage map
        │   ├── SB021_render_pc_topview.png               orthographic top-view render
        │   ├── SB021_render_pc_depth.npy                 depth map for that render
        │   └── SB021_render_pc_turbo.png                 TURBO-colormapped depth
        │
        └── .mast3r_sfm_cache/                      intermediate SfM cache, safe to delete
                                                    (removed automatically when vision.output.clean_sfm_cache: true)
```

---

## 8. Camera calibration (robot PC)

`spectra calibrate` estimates `cam2gripper` (camera → flange) from chessboard
views taken in **drag mode** (you move the robot by hand), and optionally the
camera intrinsics, then validates the result on new views. No starting
`cam2gripper` is needed. Code in `spectra/calibration/`, settings in the
`calibration:` section of `configs/default.yaml`; robot (`ip`, `tool`,
`euler_order`) and camera stream (`width`, `height`, `fps`, `n_avg_frames`)
come from `acquisition.robot` / `acquisition.camera`. Data in
`DATA/camera_calibration/` (§3.4).

The result feeds the acquisition directly: with
`acquisition.hand_eye.T_cam2gripper: null` (default) `spectra acquire` reads
`DATA/camera_calibration/output/calibration/cam2gripper.txt`; a 4x4 matrix
there pins a specific calibration instead. To acquire with the OpenCV
intrinsics, set `acquisition.camera.intrinsics` to
`DATA/camera_calibration/output/intrinsics/intrinsics_opencv.json`.

### 8.1 Procedure

**Before starting:** the active tool on the controller must be
`acquisition.robot.tool` (0 = bare flange); the acquisition checks it and
refuses to start otherwise. The chessboard must **not move** from the first
calibration view to the last validation view.

1. **Calibration views** (~30, `calibration.acquisition.n_calib_target`):
   `--modality calibration acquire`. Move the robot by hand, frame the
   chessboard, keep still, **ENTER** saves, **q/ESC** quits. Views without the
   chessboard are not saved; re-running continues the numbering. Vary the
   orientation a lot (tilt in all directions + roll around the optical axis).
   The factory intrinsics are saved once in `input/calibration/intrinsics_realsense.json`.
2. **Intrinsics** — `calibration.intrinsics.source`, used by both hand-eye and
   validation:
   - `realsense`: factory values, nothing to do;
   - `opencv`: `--modality intrinsics` (target: mean reprojection < ~0.3 px);
   - `file`: a JSON `{"K": [[fx,0,cx],[0,fy,cy],[0,0,1]], "D": [[k1,k2,p1,p2,k3]]}`
     in `calibration.intrinsics.file`. From MATLAB: `K = cameraParams.Intrinsics.K`
     (not the transposed `IntrinsicMatrix`), subtract 1 from cx and cy (MATLAB
     is 1-based), `D = [Radial(1) Radial(2) Tangential(1) Tangential(2) Radial(3)]`
     (k3 = 0 with only 2 radial coefficients).
3. **Hand-eye:** `--modality calibration calibrate` → `output/calibration/cam2gripper.txt`.
   Views with PnP reprojection error above `calibration.hand_eye.max_reproj_err_px`
   are excluded; the per-view residuals flag rotational outliers.
4. **Validation:** `--modality validation` (acquire ~20 NEW views, then
   validate). Only images and poses are written in `input/validation`; the
   intrinsics are the same as the calibration's, with a warning if they differ
   from `output/calibration/intrinsics_used.json`.

`spectra calibrate` alone runs 1 → 2 (`opencv`) → 3 → 4 in one go.

### 8.2 Metrics

| Step | Metric | Good if |
| --- | --- | --- |
| Intrinsics (`opencv`) | mean reprojection error | < 1 px (sub-pixel) |
| Hand-eye | std chessboard → base (self-check) | a few mm |
| Validation | 3D error mm, metric **Y** | compatible with the task requirements |

- **X** (self-consistency): board pose in base estimated from the validation images.
- **Y** (out-of-sample, stricter): board pose in base estimated from the calibration images.

Intrinsics are tied to the resolution (`acquisition.camera.width/height`):
images at another resolution are rejected.
