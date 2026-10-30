#!/bin/bash
#SBATCH --job-name=editdist
#SBATCH --account=fc_mvslab
#SBATCH --partition=savio3
#SBATCH --qos=savio_normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.out

# Usage (run from the directory holding edit_distance_claude.py):
#   mkdir -p logs results
#   sbatch --job-name=ed_choptf   --export=ALL,TARGET=choptf   editdist_claude.sbatch
#   sbatch --job-name=ed_train    --export=ALL,TARGET=train    editdist_claude.sbatch
#   sbatch --job-name=ed_proteome --export=ALL,TARGET=proteome editdist_claude.sbatch
#   sbatch --job-name=ed_self --array=0-3 --export=ALL,TARGET=self editdist_claude.sbatch
#
# After an array job:
#   python edit_distance_claude.py --target self --merge --outdir results

set -euo pipefail

TARGET=${TARGET:?set TARGET, e.g. --export=ALL,TARGET=train}
OUTDIR=${OUTDIR:-results}

SHARD=${SLURM_ARRAY_TASK_ID:-0}
NSHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

module load anaconda3
source activate editdistance

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "host=$(hostname) target=$TARGET shard=$SHARD/$NSHARDS cpus=${SLURM_CPUS_PER_TASK}"
python -c "import editdistance, pandas, numpy; print('deps ok')"

python edit_distance_claude.py \
    --target "$TARGET" \
    --outdir "$OUTDIR" \
    --workers "${SLURM_CPUS_PER_TASK}" \
    --shard "$SHARD" \
    --nshards "$NSHARDS"
