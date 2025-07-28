#!/bin/bash
#SBATCH --job-name="lwe_attack"
#SBATCH --output=outputs/attack_%j.out
#SBATCH --partition=all
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=64

# Base output directory
DATA_DIR="../data"
mkdir -p "$DATA_DIR"

CHECKPOINT_DIR="checkpoints"
mkdir -p "$CHECKPOINT_DIR"

CONTAINER_IMAGE="/d/hpc/projects/FRI/cb17769/lwe_container.sif"
#CONTAINER_IMAGE="/vol/csedu-nobackup/project/cbassotto/lwe_container.sif"

# Get the SLURM job ID (replaces %j)
CHECKPOINT_FILE="${CHECKPOINT_DIR}/checkpoint_${SLURM_JOB_ID}"

# Define parameters with the dynamic checkpoint path
PARAMS_JSON=$(cat <<EOF
{
  "n": 128,
  "q": 3329,

  "eta": 2,
  "secret_type": "cbd",

  "num_gen": 4,
  "seed": 0,

  "matrix_config": "dual",

  "num_matrices": $((SLURM_CPUS_PER_TASK - 1)),
  "reduction_factor": 0.875,
  "reduction_resampling": true,
  "checkpoint_filename": "${CHECKPOINT_FILE}",
  "lookback": 3,
  "warmup_steps": 15,
  "flatter_alpha": 0.001,
  "bkz_block_sizes": "4:42:2",
  "crossover": -1,

  "penalty": 4,

  "verbose": true,
  "save_to": "./../reduced_data/"
}
EOF
)

export PYTHONUNBUFFERED=1

# Execute with Singularity
singularity exec ${CONTAINER_IMAGE} python3 attack.py \
  --params "$PARAMS_JSON" \
  --num_attacks 1 \
  --attack_strategy "no" \
  --attack_every 1 \
  --save_strategy "no" \
  --save_every  20 \
  --stop_strategy "hour" \
  --stop_after 46 \
  --save_at_the_end True