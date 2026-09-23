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

Everything is driven by one YAML config and one CLI (`spectra`).

---

## Table of contents

- [1. Local Setup](#1-local-setup)
- [2. Mufasa Setup](#2-mufasa-setup)
- [3. Dataset layout](#3-dataset-layout)
- [4. CLI reference](#4-cli-reference)
- [5. Running locally](#5-running-locally)
- [6. Running on Mufasa (SLURM)](#6-running-on-mufasa-slurm)
- [7. Outputs](#7-outputs)


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
git clone https://github.com/Leo11P/SpectraBreast-Vision.git
cd SpectraBreast-Vision
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
SpectraBreast-Vision/
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
uv sync                                            # pyproject deps + torch (cu130 index)
uv run pip install -r mast3r/requirements.txt
uv run pip install -r mast3r/dust3r/requirements.txt
```

### 1.4 Build and install ASMK

ASMK ships a Cython extension that must be compiled before install:

```bash
uv run pip install cython
cd mast3r/asmk/cython && uv run cythonize *.pyx && cd ../../..
uv run pip install -e mast3r/asmk
```

### 1.5 Make `mast3r` importable outside the pipeline

The pipeline injects `mast3r/` into `sys.path` itself, so this step is not
needed to *run* the pipeline. It **is** needed for `preflight_check.sh` and for
any interactive `import mast3r`:

```bash
uv run pip install -e mast3r/ --no-deps
```

Alternatively, put it on `PYTHONPATH` permanently:

```bash
echo 'export PYTHONPATH="$HOME/SpectraBreast-Vision/mast3r:$HOME/SpectraBreast-Vision/mast3r/dust3r:$PYTHONPATH"' >> ~/.bashrc
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
git clone https://github.com/Leo11P/SpectraBreast-Vision.git
cd SpectraBreast-Vision
git clone --recursive https://github.com/naver/mast3r.git mast3r
git clone https://github.com/jenicek/asmk.git mast3r/asmk
```

### 2.2 Run the setup script

```bash
bash Setup_env.sh        # run on the LOGIN node, NOT via sbatch
```

`Setup_env.sh` assumes the repo is exactly at `~/SpectraBreast-Vision`. It
installs `uv` if absent (no `sudo` needed), checks that `mast3r/` and
`mast3r/asmk/` exist, runs `uv sync`, installs the MASt3R and dust3r
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
│       ├── *.jpg | *.png | *.bmp | *.tif | *.tiff | *.webp   ← RGB images, directly here
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



### 3.2 Layout for `batch` mode

Batch discovery (`spectra/registration/batch.py`) uses a **different, flipped
layout** — sample first, stage second:

```
DATA/
└── <sample>/
    └── input_registration/
        ├── <sample>_raw.hdr
        └── <sample>_raw.png        ← only when registration.batch.roi_mode: true
```

Batch scans every immediate subdirectory of `data_root`, optionally filtered by
`registration.batch.sample_regex`, and skips any sample missing its `.hdr`,
`surface_mesh.ply` or `aruco_markers_3d.json` (with a printed reason). Mesh and
ArUco JSON are always read from `RESULTS/<sample>/vision/`.

### 3.3 Pose file format

One file per image, named `pose_*.txt`, sorted lexicographically and zipped
against the sorted image list. Each file holds **6 whitespace-separated
numbers** on one line:

```
X Y Z ROLL PITCH YAW
```

- Translation in **metres** (`translation_scale = 1.0`).
- Angles in **radians** (`angles_in_degrees = False`).
- Convention `ROLLPITCHYAW`: `R = Rz(yaw) · Ry(pitch) · Rx(roll)`.

Both `poses/` and `camera_parameters/` are fully optional — with neither, Vision
runs pure SfM and takes its metric scale from the ArUco markers alone.



## 4. CLI reference

Installed as the `spectra` console script. All three forms are equivalent:

```bash
uv run spectra ...
uv run python -m spectra ...
uv run python -m spectra.cli ...
```

### `spectra vision`

MASt3R-SfM + ArUco reconstruction.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c`, `--config` PATH | required | YAML config (unified or legacy vision-only) |
| `--sample` TEXT | from YAML | Override `sample_name` for this run |
| `-s`, `--set` KEY=VAL | — | Dotted override, repeatable |
| `--dry-run` | off | Resolve paths, print the config, run nothing |

### `spectra registration`

HSI → mesh registration. Requires Vision outputs to exist already.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c`, `--config` PATH | required | Unified YAML (needs a `registration:` section) |
| `--sample` TEXT | from YAML | Override `sample_name`; also the output filename prefix |
| `--mode` TEXT | from YAML | `single` \| `roi` \| `batch` \| `sweep` |
| `-s`, `--set` KEY=VAL | — | Dotted override, repeatable |
| `--cpu` | off | Force CPU raycasting even when a GPU is available |
| `--dry-run` | off | Validate paths, print the config, run nothing |

### `spectra full`

Vision → Registration on one sample.

| Flag | Default | Meaning |
| --- | --- | --- |
| `-c`, `--config` PATH | required | Unified YAML with both stage sections |
| `--sample` TEXT | from YAML | Override `sample_name` |
| `-s`, `--set` KEY=VAL | — | Dotted override, repeatable |
| `--cpu` | off | Force CPU raycasting in Registration |
| `--force-vision` | off | Re-run Vision even if its outputs exist |
| `--dry-run` | off | Print the plan, run nothing |

**Vision is skipped automatically** when both `RESULTS/<sample>/vision/surface_mesh.ply`
and `RESULTS/<sample>/vision/aruco_markers_3d.json` already exist. Use
`--force-vision` to override. Which stages run at all is also gated by the
`stages:` list in the YAML — `spectra vision` and `spectra registration` ignore it.

---

## 5. Running locally

Requires a CUDA GPU for MASt3R-SfM. Registration falls back to CPU raycasting
(`--cpu`), Vision does not.

Either prefix everything with `uv run`:

```bash
cd ~/SpectraBreast-Vision
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
uv run spectra vision --config configs/default.yaml
#uv run spectra registration --config configs/default.yaml
#uv run spectra full --config configs/default.yaml
```

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
rsync -avz mufasa:~/SpectraBreast-Vision/RESULTS/SB021/ ./RESULTS/SB021/
```

Pull just the small artifacts (Excel reports, PNGs, JSON) when the point clouds
aren't needed — a full run's `RESULTS/` can be several GB.

---

## 7. Outputs

Everything goes under `RESULTS/<sample>/`, one subfolder per stage.

```
RESULTS/
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
    │   ├── SB021_registration_errors.xlsx            2D/3D errors + Timings sheet
    │   ├── SB021_registration.png                    HSI + render overlay with ArUco
    │   ├── SB021_coverage.png                        HSI → mesh coverage map
    │   ├── SB021_render_pc_topview.png               orthographic top-view render
    │   ├── SB021_render_pc_depth.npy                 depth map for that render
    │   └── SB021_render_pc_turbo.png                 TURBO-colormapped depth
    │
    └── .mast3r_sfm_cache/                      intermediate SfM cache, safe to delete
```
