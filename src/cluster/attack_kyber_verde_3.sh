#!/bin/bash
#SBATCH --job-name=attack_kyber_verde_3
#SBATCH --output=outputs/attack_kyber_verde_3_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=all

### TO BE RUN WITH: sbatch attack_kyber_verde_3.sh ###

n=350
k=1
q=1489513

hw=12 # 12 or more

max_size=100 # -1, 0 or from 50 to 256 (lower values use less memory)
reduction_samples=-1 # -1, 0.875, 256
reduction_resampling=false # true for LWE, false for MLWE

matrix_config="dual" # dual or salsa

penalty=10 # from 8 to 10
bkz_block_size="20:40:10" # 20:40:10, 40:40:1 or 30:50:10

# min_matrices = (n // (m + 1) + 1) * num_gen * k
num_matrices=16 # same as CPUs or at least min_matrices

# Best parallelization with joblib or process because of CPU-heavy BKZ
parallel_backend="joblib"
update_strategy="mean" # "percentage" if max_size is high, "mean" if max_size is low

DATA_DIR="../data"
mkdir -p "$DATA_DIR"

CONTAINER_IMAGE="/d/hpc/projects/FRI/cb17769/lwe_container.sif"
#CONTAINER_IMAGE="/vol/csedu-nobackup/project/cbassotto/lwe_container.sif"

# Parameters description can be found in toy_example.sh
PARAMS_JSON=$(cat <<EOF
{
  "n": $n,
  "k": $k,
  "q": $q,
  "secret_type": "binary",
  "eta": 2,
  "gaussian_std": 3,
  "hw": $hw,
  "error_type": "gaussian",
  "num_gen": 4,
  "seed": 42,
  "float_type": "ld",
  "matrix_config": "$matrix_config",
  "reduction_samples": $reduction_samples,
  "reduction_resampling": $reduction_resampling,
  "continuous_reduction": true,
  "parallel_backend": "$parallel_backend",
  "min_samples": 0,
  "num_matrices": $num_matrices,
  "reduction_max_size": $max_size,
  "lookback": 4,
  "update_strategy": "$update_strategy",
  "warmup_steps": 10,
  "flatter_alpha": 0.001,
  "bkz_delta": 0.99,
  "bkz_block_sizes": "$bkz_block_size",
  "use_polish": true,
  "interleaved_steps": 0,
  "penalty": $penalty,
  "verbose": true,
  "train_percentages": [0.1, 0.25, 0.5],
  "subsets_with_probs": false,
  "model": "tukey",
  "lr": 0.0001,
  "c_factor": 1.0,
  "epsilon": 1.1,
  "max_iter": 15000,
  "alpha": 0.0001,
  "warm_start": false,
  "fit_intercept": false,
  "tol": 0.0001,
  "use_ransac": false,
  "residual_factor": 1.5,
  "max_trials": 100,
  "normalize_raw_secret": true,
  "save_to": "$DATA_DIR"
}
EOF
)

export PYTHONUNBUFFERED=1

singularity exec ${CONTAINER_IMAGE} python attack.py \
  --params "$PARAMS_JSON" \
  --num_attacks 1 \
  --attack_strategy "no" \
  --attack_every 0 \
  --save_strategy "no" \
  --save_every 0 \
  --stop_strategy "hour" \
  --stop_after 40 \
  --save_at_the_end \
  --train_secret_types "binary" \
  --hw_range 1:15:1