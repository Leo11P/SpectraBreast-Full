#!/bin/bash
#SBATCH --job-name=spectra_run
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=jobs
#SBATCH --qos=gpuwide 
#SBATCH --gres=gpu:4g.20gb:1
#SBATCH --time=12:00:00
#SBATCH --output=./logs/run_spectra-%j.out
#SBATCH --error=./logs/run_spectra-%j.err

set -euo pipefail
mkdir -p logs

echo "=============================================="
echo "  Spectra run (uv, no Singularity)"
echo "  Job ID  : ${SLURM_JOB_ID}"
echo "  Node    : $(hostname)"
echo "  Start   : $(date)"
echo "=============================================="

cd ~/SpectraBreast-Full

export UV_CACHE_DIR="${HOME}/.cache/uv"
export PIP_CACHE_DIR="${HOME}/.cache/pip"
export HF_HOME="${HOME}/.cache/huggingface"
export TORCH_HOME="${HOME}/.cache/torch"
export TMPDIR="${HOME}/.cache/tmp"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${UV_CACHE_DIR}" "${PIP_CACHE_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TMPDIR}"

export PATH="${HOME}/.local/bin:${PATH}"

# Modello di ricostruzione: DEVE essere lo stesso di vision.backend nel config.
# Ogni modello ha il suo ambiente (bash Setup_env.sh <modello>).
#   sbatch run_spectra.sh                      -> mast3r
#   BACKEND=vggt_omega sbatch run_spectra.sh   -> vggt_omega (idem pi3, da3, mapanything)
BACKEND="${BACKEND:-mast3r}"
case "${BACKEND}" in
    mast3r)     export UV_PROJECT_ENVIRONMENT=.venv;            UV_EXTRAS="--extra processing --extra mast3r" ;;
    vggt_omega)  export UV_PROJECT_ENVIRONMENT=.venv-vggt_omega;  UV_EXTRAS="--extra processing --extra vggt-omega" ;;
    pi3)         export UV_PROJECT_ENVIRONMENT=.venv-pi3;         UV_EXTRAS="--extra processing --extra pi3" ;;
    da3)         export UV_PROJECT_ENVIRONMENT=.venv-da3;         UV_EXTRAS="--extra processing --extra da3" ;;
    mapanything) export UV_PROJECT_ENVIRONMENT=.venv-mapanything; UV_EXTRAS="--extra processing --extra mapanything" ;;
    *) echo "ERRORE: BACKEND=${BACKEND} sconosciuto (mast3r | vggt_omega | pi3 | da3 | mapanything)"; exit 1 ;;
esac

if ! command -v uv &> /dev/null; then
    echo "ERRORE: uv non trovato. Esegui prima: bash Setup_env.sh ${BACKEND}"
    exit 1
fi
if [ ! -d "${UV_PROJECT_ENVIRONMENT}" ]; then
    echo "ERRORE: ${UV_PROJECT_ENVIRONMENT} non trovato. Esegui prima: bash Setup_env.sh ${BACKEND}"
    exit 1
fi

echo "  Python  : $(uv run ${UV_EXTRAS} python --version)"
echo "  Torch   : $(uv run ${UV_EXTRAS} python -c 'import torch; print(torch.__version__)')"
echo "  CUDA ok : $(uv run ${UV_EXTRAS} python -c 'import torch; print(torch.cuda.is_available())')"
echo "=============================================="

# Scegli lo stage da lanciare (uno solo, decommenta quello che serve).
# Gli output finiscono in RESULTS/<gg-mm-aaaa>/<sample>/<modello>/<stage>/ e
# sovrascrivono la cartella dello stesso giorno e modello se esiste gia'.
uv run ${UV_EXTRAS} spectra full --config configs/default.yaml
#uv run ${UV_EXTRAS} spectra vision --config configs/default.yaml
#uv run ${UV_EXTRAS} spectra registration --config configs/default.yaml

# La pipeline cancella .mast3r_sfm_cache/ da sola alla fine di ogni run
# (vision.output.clean_sfm_cache). Questa passata resta come rete di sicurezza
# per i job interrotti, dove la pulizia in-process non arriva a girare.
echo "  Pulizia cache SfM residua (job interrotti)..."
find RESULTS -name ".mast3r_sfm_cache" -type d -exec rm -rf {} + 2>/dev/null || true
echo "  Fatto."

echo "=============================================="
echo "  Run finished : $(date)"
echo "  Spazio home  : $(df -h $HOME | tail -1 | awk '{print $4}') liberi"
echo "=============================================="