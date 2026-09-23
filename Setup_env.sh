#!/bin/bash
# setup_env.sh — da eseguire UNA VOLTA sul nodo di login di Mufasa (NON via sbatch)
# Crea il virtual environment uv usando mast3r e asmk gia' clonati DENTRO la repo.
#
# Uso:
#   cd ~/SpectraBreast-Vision
#   bash setup_env.sh

set -euo pipefail

echo "=============================================="
echo "  Setup ambiente Spectra (uv, no Singularity)"
echo "  User    : ${USER}"
echo "  Start   : $(date)"
echo "=============================================="

REPO_DIR="${HOME}/SpectraBreast-Vision"
cd "${REPO_DIR}"

# --- Cache nella home (mai in /tmp) ------------------------------------------
export UV_CACHE_DIR="${HOME}/.cache/uv"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TMPDIR="${HOME}/.cache/tmp"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

# --- Check spazio disponibile nella home -------------------------------------
MIN_GB=20
AVAIL_GB=$(df -BG --output=avail "${HOME}" | tail -1 | tr -dc '0-9')
echo "  Spazio libero in ${HOME}: ${AVAIL_GB} GB (minimo richiesto: ${MIN_GB} GB)"
if [ "${AVAIL_GB}" -lt "${MIN_GB}" ]; then
    echo "ERRORE: spazio insufficiente nella home. Libera spazio prima di procedere."
    exit 1
fi

# --- 1. Installa uv (se non c'e' gia') ---------------------------------------
export PATH="${HOME}/.local/bin:${PATH}"
if ! command -v uv &> /dev/null; then
    echo "[1/5] Installazione uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
    echo "  uv installato: $(uv --version)"
else
    echo "[1/5] uv gia' presente: $(uv --version)"
fi

# --- 2. Verifica mast3r e asmk -----------------------------------------------
MAST3R_DIR="${REPO_DIR}/mast3r"
ASMK_DIR="${REPO_DIR}/mast3r/asmk"

if [ ! -d "${MAST3R_DIR}" ]; then
    echo "ERRORE: ${MAST3R_DIR} non trovato."
    exit 1
fi
if [ ! -d "${ASMK_DIR}" ]; then
    echo "ERRORE: ${ASMK_DIR} non trovato."
    exit 1
fi
echo "[2/5] mast3r e asmk trovati."

# --- 3. uv sync --------------------------------------------------------------
echo "[3/5] uv sync..."
uv sync

# Installa dipendenze mast3r e dust3r (usa 'uv run pip', non 'pip' direttamente)
uv run pip install --no-cache-dir -r "${MAST3R_DIR}/requirements.txt"
if [ -f "${MAST3R_DIR}/dust3r/requirements.txt" ]; then
    uv run pip install --no-cache-dir -r "${MAST3R_DIR}/dust3r/requirements.txt"
fi

# --- 3b. Fix conflitto opencv-python / opencv-contrib-python -----------------
# dust3r/requirements.txt elenca 'opencv-python' non pinnato. Il progetto usa
# 'opencv-contrib-python' (da pyproject.toml, via uv sync) per avere il modulo
# cv2.aruco richiesto dalla pipeline. I due pacchetti condividono lo stesso
# namespace 'cv2' e sono reciprocamente incompatibili: installarli entrambi
# lascia site-packages/cv2/ in uno stato ambiguo, e qualsiasi build vinca sul
# filesystem puo' avere una firma diversa per cv2.aruco.detectMarkers()
# (es. 'ids' 1D invece di 2D), che rompe la registration con:
#   IndexError: invalid index to scalar variable.
# Si rimuove sempre opencv-python e si reinstalla il contrib pinnato, cosi'
# l'ordine di installazione dei requirements non conta piu'.
echo "[3b/5] Risoluzione conflitto opencv-python / opencv-contrib-python..."
uv run pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
uv run pip install --no-cache-dir --force-reinstall --no-deps "opencv-contrib-python>=4.8"

# Compila cython per asmk e installa
uv run pip install --no-cache-dir cython
if [ -d "${ASMK_DIR}/cython" ]; then
    cd "${ASMK_DIR}/cython"
    uv run cythonize *.pyx
    cd "${REPO_DIR}"
fi
uv run pip install --no-cache-dir -e "${ASMK_DIR}"

# --- 4. Smoke test -----------------------------------------------------------
echo "[4/5] Smoke test..."
uv run python -c "
import torch
print(f'  torch     : {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
import spectra
print(f'  spectra   : {spectra.__version__}')
print('  OK')
"

# --- 5. Verifica cv2.aruco (rileva subito un eventuale conflitto opencv) -----
echo "[5/5] Verifica cv2.aruco..."
uv run python -c "
import cv2
import numpy as np
print(f'  cv2       : {cv2.__version__}')
d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
img = np.zeros((200, 200), dtype=np.uint8)
corners, ids, _ = det.detectMarkers(img)
assert ids is None or ids.ndim == 2, f'ids ha shape inattesa: {None if ids is None else ids.shape}'
print('  cv2.aruco OK (nessun conflitto opencv-python / opencv-contrib-python)')
"

echo "=============================================="
echo "  Setup completato: $(date)"
echo "  Ora puoi lanciare: sbatch run_spectra.sh"
echo "=============================================="