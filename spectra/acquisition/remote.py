#!/usr/bin/env python3
# =============================================================================
#  remote.py  —  send an acquired sample to Mufasa and run the pipeline there
# -----------------------------------------------------------------------------
#  Runs on the robot PC (the cluster cannot open connections towards it, so the
#  flow is always PC -> Mufasa over SSH/SFTP). Called automatically by
#  acquire.py at the end of an acquisition (acquisition.remote.submit_after_
#  acquisition), or by hand to (re)submit a sample already on disk:
#
#      python spectra/acquisition/remote.py --sample SB030
#
#  Steps:
#    1. one SSH connection to the login node (key from the config, or password)
#    2. upload DATA/input_vision/<sample>/ (+ DATA/input_registration/<sample>/
#       if the HSI cube is there) into <runs_dir>/<sample>_<timestamp>/DATA/
#    3. verify the upload with SHA256SUMS (written by acquire.py)
#    4. upload run_config.yaml = the LOCAL unified config with only
#       sample_name / data_root / results_root / stages changed, so Mufasa uses
#       exactly the values used for the acquisition (marker size, ...). The job
#       sets SPECTRA_DEFAULT_CONFIG to it: for that run it IS the default config
#       (single source of the marker edge), the repo's configs/default.yaml on
#       Mufasa is neither read nor modified.
#    5. sbatch a generated job script that runs `spectra full` on a GPU node
#    6. optionally wait for the job and download RESULTS/ into the local
#       results_root (same RESULTS/<dd-mm-yyyy>/<sample>/ layout)
#
#  Requires: paramiko, pyyaml   (pip install paramiko pyyaml)
# =============================================================================
from __future__ import annotations

import argparse
import copy
import getpass
import os
import posixpath
import shlex
import stat
import sys
import time

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(os.path.dirname(ROOT))
DEFAULT_CONFIG = os.path.join(REPO_ROOT, "configs", "default.yaml")

# SLURM states meaning the job has left the queue.
TERMINAL_STATES = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
                   "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE"}
# Never uploaded / never downloaded (scratch, caches, model weights).
SKIP_UPLOAD = {"_discarded"}
SKIP_PULL_NAMES = {".mast3r_sfm_cache", "inputs_exif_canonical", "__pycache__"}
SKIP_PULL_SUFFIXES = (".pth", ".ckpt", ".safetensors")

DEFAULTS = {
    "submit_after_acquisition": False,
    "host": None, "port": 22, "user": None, "ssh_key_path": None,
    "repo_dir": None, "runs_dir": None,
    "slurm_bin_dirs": [],
    "partition": "jobs", "qos": "gpuwide", "gres": "gpu:4g.20gb:1",
    "cpus": 8, "mem": "64G", "time": "12:00:00", "job_name": "spectra",
    "wait": True, "poll_s": 20, "pull_results": True,
}


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
def load_settings(config_path, sample=None):
    """Returns (full_cfg_dict, remote_settings, sample, local_paths)."""
    with open(config_path, "r", encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    r = dict(DEFAULTS)
    r.update((full.get("acquisition") or {}).get("remote") or {})
    missing = [k for k in ("host", "user", "repo_dir", "runs_dir") if not r.get(k)]
    if missing:
        raise SystemExit(f"[remote] acquisition.remote.{missing} missing in {config_path}")
    sample = sample or full.get("sample_name")
    if not sample:
        raise SystemExit("[remote] sample_name missing (config or --sample)")
    data_root = os.path.abspath(str(full.get("data_root", "DATA")))
    paths = {
        "vision": os.path.join(data_root, "input_vision", sample),
        "registration": os.path.join(data_root, "input_registration", sample),
        "results_root": os.path.abspath(str(full.get("results_root", "RESULTS"))),
    }
    if not os.path.isdir(paths["vision"]):
        raise SystemExit(f"[remote] {paths['vision']} not found: nothing to send")
    return full, r, sample, paths


def marker_edge_mm(config_path, full):
    """Marker edge length (mm) used locally. Like the pipeline, it is read ONLY
    from the repo's configs/default.yaml; another config must not set it."""
    if os.path.abspath(config_path) != os.path.abspath(DEFAULT_CONFIG):
        if ((full.get("vision") or {}).get("aruco") or {}).get("marker_edge_length_mm") is not None:
            raise SystemExit(f"[remote] {config_path}: vision.aruco.marker_edge_length_mm "
                             f"cannot be set here, only in {DEFAULT_CONFIG}")
        with open(DEFAULT_CONFIG, "r", encoding="utf-8") as f:
            full = yaml.safe_load(f) or {}
    value = ((full.get("vision") or {}).get("aruco") or {}).get("marker_edge_length_mm")
    if value is None:
        raise SystemExit(f"[remote] vision.aruco.marker_edge_length_mm missing in {DEFAULT_CONFIG}")
    return float(value)


def check_remote_repo(sftp, r):
    """Fail early (before uploading) if acquisition.remote.repo_dir is wrong."""
    path = posixpath.join(r["repo_dir"], "pyproject.toml")
    try:
        sftp.stat(path)
    except FileNotFoundError:
        raise SystemExit(f"[remote] {path} not found on Mufasa: check acquisition.remote.repo_dir")


def has_hsi(paths, sample):
    return os.path.isfile(os.path.join(paths["registration"], f"{sample}_raw.hdr"))


def render_run_config(full, sample, remote_run, run_registration, marker_mm):
    """Local unified config with only the run-specific fields changed."""
    cfg = copy.deepcopy(full)
    cfg.pop("acquisition", None)          # robot-PC settings: not needed on the cluster
    # This file is the run's default config on Mufasa (SPECTRA_DEFAULT_CONFIG),
    # so it carries the marker edge used locally (from configs/default.yaml).
    cfg.setdefault("vision", {}).setdefault("aruco", {})["marker_edge_length_mm"] = marker_mm
    ((cfg.get("registration") or {}).get("detection") or {}).pop("marker_edge_length_mm", None)
    cfg["sample_name"] = sample
    cfg["data_root"] = posixpath.join(remote_run, "DATA")
    cfg["results_root"] = posixpath.join(remote_run, "RESULTS")
    cfg["results_date"] = None
    cfg["stages"] = ["vision", "registration"] if run_registration else ["vision"]
    return yaml.safe_dump(cfg, sort_keys=False)


def build_job_script(r, remote_run, run_config):
    return f"""#!/bin/bash
#SBATCH --job-name={r['job_name']}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition={r['partition']}
#SBATCH --qos={r['qos']}
#SBATCH --gres={r['gres']}
#SBATCH --cpus-per-task={r['cpus']}
#SBATCH --mem={r['mem']}
#SBATCH --time={r['time']}
#SBATCH --output={remote_run}/slurm-%j.out
#SBATCH --error={remote_run}/slurm-%j.out

set -euo pipefail
echo "Host: $(hostname)   SLURM_JOB_ID: $SLURM_JOB_ID   Start: $(date)"
cd {shlex.quote(r['repo_dir'])}

export UV_CACHE_DIR="${{HOME}}/.cache/uv"
export PIP_CACHE_DIR="${{HOME}}/.cache/pip"
export HF_HOME="${{HOME}}/.cache/huggingface"      # model weights: persistent cache
export TORCH_HOME="${{HOME}}/.cache/torch"
export TMPDIR="${{HOME}}/.cache/tmp/spectra_${{SLURM_JOB_ID}}"   # per-job scratch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${{UV_CACHE_DIR}}" "${{PIP_CACHE_DIR}}" "${{HF_HOME}}" "${{TORCH_HOME}}" "${{TMPDIR}}"
trap 'rm -rf "${{TMPDIR}}"' EXIT
export PATH="${{HOME}}/.local/bin:${{PATH}}"
# This run's config is its default config (marker edge included): the repo's
# configs/default.yaml on Mufasa is not used.
export SPECTRA_DEFAULT_CONFIG={shlex.quote(run_config)}

uv run spectra full --config {shlex.quote(run_config)}
echo "Finished: $(date)"
"""


# --------------------------------------------------------------------------- #
# SSH / SFTP helpers
# --------------------------------------------------------------------------- #
def connect(r):
    try:
        import paramiko
    except ImportError:
        raise SystemExit("[remote] missing dependency: pip install paramiko")
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    kw = dict(hostname=r["host"], port=int(r["port"]), username=r["user"],
              timeout=30, look_for_keys=False, allow_agent=False)
    if r.get("ssh_key_path"):
        kw["key_filename"] = os.path.expanduser(str(r["ssh_key_path"]))
    else:
        kw["password"] = getpass.getpass(f"Password for {r['user']}@{r['host']}: ")
    print(f"[remote] connecting to {r['user']}@{r['host']} ...")
    client.connect(**kw)
    return client


def run_cmd(client, r, command):
    """Runs `command` on the login node in an interactive login shell
    (`bash -ilc`), with the SLURM binaries prepended to PATH. Non-interactive
    shells skip most of ~/.bashrc, and sbatch copies the caller's environment
    into the job: without -i the job missed env vars set there (this caused
    `ModuleNotFoundError: mast3r` in the past)."""
    dirs = r.get("slurm_bin_dirs") or []
    prefix = f"export PATH={':'.join(dirs)}:$PATH && " if dirs else ""
    _, stdout, stderr = client.exec_command(f"bash -ilc {shlex.quote(prefix + command)}")
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    return stdout.channel.recv_exit_status(), out.strip(), err.strip()


def sftp_makedirs(sftp, remote_dir):
    cur = "/"
    for part in remote_dir.strip("/").split("/"):
        cur = posixpath.join(cur, part)
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def sftp_push_dir(sftp, local_dir, remote_dir):
    """Recursive upload (skips SKIP_UPLOAD). Returns the number of files."""
    n = 0
    for base, dirs, files in os.walk(local_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_UPLOAD)
        rel = os.path.relpath(base, local_dir)
        rdir = remote_dir if rel == "." else posixpath.join(remote_dir, *rel.split(os.sep))
        sftp_makedirs(sftp, rdir)
        for fn in sorted(files):
            sftp.put(os.path.join(base, fn), posixpath.join(rdir, fn))
            n += 1
    return n


def sftp_pull_dir(sftp, remote_dir, local_dir):
    """Recursive download (skips caches / weights). Returns the number of files."""
    os.makedirs(local_dir, exist_ok=True)
    n = 0
    try:
        entries = sftp.listdir_attr(remote_dir)
    except FileNotFoundError:
        return 0
    for e in entries:
        low = e.filename.lower()
        if e.filename in SKIP_PULL_NAMES or low.endswith(SKIP_PULL_SUFFIXES):
            continue
        rpath = posixpath.join(remote_dir, e.filename)
        lpath = os.path.join(local_dir, e.filename)
        if stat.S_ISDIR(e.st_mode):
            n += sftp_pull_dir(sftp, rpath, lpath)
        else:
            sftp.get(rpath, lpath)
            n += 1
    return n


def job_state(client, r, jobid):
    _, out, _ = run_cmd(client, r, f"squeue -j {jobid} -h -o %T")
    if out:
        return out.splitlines()[0].strip()
    _, out, _ = run_cmd(client, r, f"sacct -j {jobid} -n -X -P --format=State")
    return out.splitlines()[0].strip().split()[0] if out else "UNKNOWN"


# --------------------------------------------------------------------------- #
# Main entry point
# --------------------------------------------------------------------------- #
def submit_sample(config_path=DEFAULT_CONFIG, sample=None, wait=None, pull=None):
    """Uploads the sample, submits the SLURM job and (optionally) waits for it
    and downloads the results. Returns (jobid, final_state)."""
    full, r, sample, paths = load_settings(config_path, sample)
    local_mm = marker_edge_mm(config_path, full)
    wait = r["wait"] if wait is None else wait
    pull = r["pull_results"] if pull is None else pull
    run_registration = has_hsi(paths, sample)
    remote_run = posixpath.join(r["runs_dir"], f"{sample}_{time.strftime('%Y%m%d_%H%M%S')}")
    remote_sample = posixpath.join(remote_run, "DATA", "input_vision", sample)
    print(f"[remote] sample {sample} -> {remote_run}")
    if not run_registration:
        print(f"[remote] no {sample}_raw.hdr in {paths['registration']}: "
              "running VISION only (registration needs the HSI cube)")

    client = connect(r)
    try:
        sftp = client.open_sftp()
        check_remote_repo(sftp, r)                        # before uploading anything
        n = sftp_push_dir(sftp, paths["vision"], remote_sample)
        print(f"[remote] uploaded {n} files (input_vision)")
        if run_registration:
            n = sftp_push_dir(sftp, paths["registration"],
                              posixpath.join(remote_run, "DATA", "input_registration", sample))
            print(f"[remote] uploaded {n} files (input_registration)")

        if os.path.isfile(os.path.join(paths["vision"], "SHA256SUMS")):
            code, out, err = run_cmd(client, r, f"cd {shlex.quote(remote_sample)} && "
                                                "sha256sum -c --quiet SHA256SUMS")
            if code != 0:
                raise SystemExit(f"[remote] checksum mismatch after upload:\n{out or err}")
            print("[remote] upload verified (SHA256SUMS)")
        else:
            print("[remote] warning: no SHA256SUMS, upload not verified")

        run_config = posixpath.join(remote_run, "run_config.yaml")
        with sftp.open(run_config, "w") as fh:
            fh.write(render_run_config(full, sample, remote_run, run_registration, local_mm))
        job = posixpath.join(remote_run, "job.slurm")
        with sftp.open(job, "w") as fh:
            fh.write(build_job_script(r, remote_run, run_config))

        code, out, err = run_cmd(client, r, f"cd {shlex.quote(remote_run)} && sbatch job.slurm")
        jobid = out.split()[-1] if code == 0 and out else ""
        if not jobid.isdigit():
            raise SystemExit(f"[remote] sbatch failed:\n{err or out}")
        print(f"[remote] submitted job {jobid} (qos {r['qos']}). "
              f"Log on Mufasa: {remote_run}/slurm-{jobid}.out")
        if not wait:
            return jobid, "SUBMITTED"

        print(f"[remote] waiting for job {jobid} (poll {r['poll_s']}s; Ctrl-C stops "
              "waiting, the job keeps running on Mufasa)")
        last, state, unknown = None, "UNKNOWN", 0
        while True:
            state = job_state(client, r, jobid)
            if state != last:
                print(f"  [{time.strftime('%H:%M:%S')}] {state}", flush=True)
                last = state
            # UNKNOWN right after sbatch can be a short accounting delay
            unknown = unknown + 1 if state == "UNKNOWN" else 0
            if state in TERMINAL_STATES or unknown >= 3:
                break
            time.sleep(int(r["poll_s"]))
        if state == "UNKNOWN":
            print(f"[remote] warning: cannot read the state of job {jobid} "
                  "(check acquisition.remote.slurm_bin_dirs); it may still be running")
        _, tail, _ = run_cmd(client, r, f"tail -n 30 {remote_run}/slurm-{jobid}.out 2>/dev/null")
        if tail:
            print("--- last lines of the job log ---\n" + tail + "\n---")
        if pull:
            n = sftp_pull_dir(sftp, posixpath.join(remote_run, "RESULTS"), paths["results_root"])
            print(f"[remote] downloaded {n} files into {paths['results_root']}")
        return jobid, state
    finally:
        client.close()


def main():
    ap = argparse.ArgumentParser(description="Send an acquired sample to Mufasa and run "
                                             "`spectra full` there (SLURM)")
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="unified pipeline config")
    ap.add_argument("--sample", default=None, help="override sample_name")
    ap.add_argument("--no-wait", action="store_true", help="submit and return, do not wait")
    ap.add_argument("--no-pull", action="store_true", help="do not download the results")
    args = ap.parse_args()
    submit_sample(args.config, args.sample,
                  wait=False if args.no_wait else None,
                  pull=False if args.no_pull else None)


if __name__ == "__main__":
    main()
