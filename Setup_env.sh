#!/bin/bash
# Setup_env.sh — da eseguire sul nodo di login di Mufasa (NON via sbatch),
# UNA VOLTA per ogni modello di ricostruzione (vision.backend). Ogni modello ha
# il SUO virtual environment, con `processing` + l'extra del modello:
#
#   bash Setup_env.sh               # = mast3r      -> .venv
#   bash Setup_env.sh mast3r        #   (usa mast3r e asmk gia' clonati DENTRO la repo)
#   bash Setup_env.sh vggt_omega    #               -> .venv-vggt_omega
#   bash Setup_env.sh pi3           #               -> .venv-pi3
#   bash Setup_env.sh da3           #               -> .venv-da3
#   bash Setup_env.sh mapanything   #               -> .venv-mapanything
#
# Rilanciarlo e' sicuro (aggiorna l'ambiente). Dopo una modifica di
# pyproject.toml rilanciarlo per i modelli che si usano.

set -euo pipefail

BACKEND="${1:-mast3r}"
case "${BACKEND}" in
    mast3r)     VENV=".venv";            EXTRA="mast3r" ;;
    vggt_omega)  VENV=".venv-vggt_omega";  EXTRA="vggt-omega" ;;
    pi3)         VENV=".venv-pi3";         EXTRA="pi3" ;;
    da3)         VENV=".venv-da3";         EXTRA="da3" ;;
    mapanything) VENV=".venv-mapanything"; EXTRA="mapanything" ;;
    *) echo "ERRORE: modello '${BACKEND}' sconosciuto (mast3r | vggt_omega | pi3 | da3 | mapanything)"; exit 1 ;;
esac

echo "=============================================="
echo "  Setup ambiente Spectra (uv, no Singularity)"
echo "  Modello : ${BACKEND}  ->  ${VENV}  (--extra processing --extra ${EXTRA})"
echo "  User    : ${USER}"
echo "  Start   : $(date)"
echo "=============================================="

REPO_DIR="${HOME}/SpectraBreast-Full"
cd "${REPO_DIR}"

# --- Cache nella home (mai in /tmp) ------------------------------------------
export UV_CACHE_DIR="${HOME}/.cache/uv"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TMPDIR="${HOME}/.cache/tmp"
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

# L'ambiente del modello: uv sync / uv run lavorano su questo.
export UV_PROJECT_ENVIRONMENT="${VENV}"
UV_EXTRAS=(--extra processing --extra "${EXTRA}")

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

# --- 2. Codice del modello ----------------------------------------------------
if [ "${BACKEND}" = "mast3r" ]; then
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
else
    echo "[2/5] ${BACKEND}: il codice arriva con l'extra ${EXTRA} (pyproject.toml)."
fi

# --- 3. uv sync --------------------------------------------------------------
echo "[3/5] uv sync ${UV_EXTRAS[*]}  (in ${VENV})..."
# processing = vision + registration (torch, open3d, ...), + l'extra del modello.
# NB: un `uv sync` senza questi --extra li RIMUOVEREBBE.
uv sync "${UV_EXTRAS[@]}"

if [ "${BACKEND}" = "mast3r" ]; then
    # Installa dipendenze mast3r e dust3r (usa 'uv run pip', non 'pip' direttamente)
    uv run "${UV_EXTRAS[@]}" pip install --no-cache-dir -r "${MAST3R_DIR}/requirements.txt"
    if [ -f "${MAST3R_DIR}/dust3r/requirements.txt" ]; then
        uv run "${UV_EXTRAS[@]}" pip install --no-cache-dir -r "${MAST3R_DIR}/dust3r/requirements.txt"
    fi

    # --- 3b. Fix conflitto opencv-python / opencv-contrib-python -------------
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
    uv run "${UV_EXTRAS[@]}" pip uninstall -y opencv-python opencv-python-headless 2>/dev/null || true
    uv run "${UV_EXTRAS[@]}" pip install --no-cache-dir --force-reinstall --no-deps "opencv-contrib-python>=4.8"

    # Compila cython per asmk e installa
    uv run "${UV_EXTRAS[@]}" pip install --no-cache-dir cython
    if [ -d "${ASMK_DIR}/cython" ]; then
        cd "${ASMK_DIR}/cython"
        uv run "${UV_EXTRAS[@]}" cythonize *.pyx
        cd "${REPO_DIR}"
    fi
    uv run "${UV_EXTRAS[@]}" pip install --no-cache-dir -e "${ASMK_DIR}"
fi
# Gli altri modelli: opencv-python / opencv-python-headless (loro dipendenze)
# sono esclusi in pyproject.toml ([tool.uv] override-dependencies), quindi
# niente conflitto cv2.

# --- 4. Smoke test -----------------------------------------------------------
echo "[4/5] Smoke test..."
uv run "${UV_EXTRAS[@]}" python -c "
import torch
print(f'  torch     : {torch.__version__}')
print(f'  CUDA avail: {torch.cuda.is_available()}')
import spectra
print(f'  spectra   : {spectra.__version__}')
from spectra.vision.backends.registry import missing_requirements
m = missing_requirements('${BACKEND}')
print(f'  ${BACKEND}: ' + ('OK' if not m else f'MANCA {m}'))
"
if [ "${BACKEND}" = "vggt_omega" ]; then
    # Pesi (~4-5 GB ciascuno) nella cache HF_HOME, la stessa dei job. Repo HF ad
    # accesso riservato: serve una volta l'accesso approvato su
    # https://huggingface.co/facebook/VGGT-Omega e `uv run hf auth login`.
    # Varianti da scaricare: VGGT_VARIANTS="512 416" bash Setup_env.sh vggt_omega
    VGGT_VARIANTS="${VGGT_VARIANTS:-512}" uv run "${UV_EXTRAS[@]}" python -c "
import os
from spectra.vision.backends.vggt_omega_backend import CHECKPOINTS
from huggingface_hub import hf_hub_download
for v in os.environ['VGGT_VARIANTS'].split():
    repo, f, _ = CHECKPOINTS[v]
    try:
        print(f'  pesi {v}: {hf_hub_download(repo, f)}')
    except Exception as e:
        print(f'  pesi {v}: NON scaricati ({type(e).__name__}). Accesso su https://huggingface.co/{repo} + uv run hf auth login, poi rilancia.')
" || true
fi
if [ "${BACKEND}" = "pi3" ] || [ "${BACKEND}" = "da3" ] || [ "${BACKEND}" = "mapanything" ]; then
    # Pesi dal Hugging Face Hub nella cache HF_HOME (la stessa dei job), cosi'
    # i job non li scaricano. Solo la variante di default; altre con
    # MODEL_VARIANTS="x base" bash Setup_env.sh pi3 (nomi = vision.<modello>.variant).
    MODEL_VARIANTS="${MODEL_VARIANTS:-}" uv run "${UV_EXTRAS[@]}" python -c "
import importlib, os
from huggingface_hub import snapshot_download
from spectra.vision.config import ReconstructionConfig
mod = importlib.import_module('spectra.vision.backends.${BACKEND}_backend')
default = ReconstructionConfig.model_fields['${BACKEND}'].default_factory().variant
for v in (os.environ['MODEL_VARIANTS'].split() or [default]):
    repo = mod.REPOS[v]
    try:
        print(f'  pesi {v}: {snapshot_download(repo)}')
    except Exception as e:
        print(f'  pesi {v}: NON scaricati ({type(e).__name__}: {e})')
" || true
fi

# --- 5. Verifica cv2.aruco (rileva subito un eventuale conflitto opencv) -----
echo "[5/5] Verifica cv2.aruco..."
uv run "${UV_EXTRAS[@]}" python -c "
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
echo "  Setup ${BACKEND} completato: $(date)"
echo "  Ambiente: ${VENV}. Nel config: vision.backend: ${BACKEND}"
echo "=============================================="
