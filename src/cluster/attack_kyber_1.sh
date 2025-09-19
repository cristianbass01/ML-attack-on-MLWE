#!/bin/bash
#SBATCH --job-name=attack_kyber_1
#SBATCH --output=outputs/attack_kyber_1_%j.out
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --partition=all

### TO BE RUN WITH: sbatch attack_kyber_1.sh ###

n=256
k=2
q=3329

hw=6 # can also be set to 5 to beat FRESCA

max_size=0 # -1, 0 or from 50 to 512 (lower values use less memory)
reduction_samples=150 # -1, 0.875, 256, 283, 512
reduction_resampling=true # true for LWE, false for MLWE

matrix_config="dual" # dual or salsa

penalty=4 # 3 or 4
bkz_block_size="20:50:5" # 20:40:10, 40:40:1 or 30:50:10

num_matrices=64 # same as CPUs
parallel_backend="joblib"
update_strategy="mean" # "percentage" or "mean"

DATA_DIR="../data"
mkdir -p "$DATA_DIR"

CONTAINER_IMAGE="/d/hpc/projects/FRI/cb17769/lwe_container.sif"
#CONTAINER_IMAGE="/vol/csedu-nobackup/project/cbassotto/lwe_container.sif"

### PARAMETERS FOR THE ATTACK ###
# n: LWE dimension
# k: LWE dimension rank MLWE
# q: Modulus
# secret_type: Type of secret
# eta: Parameter for secret distribution (centered binomial)
# gaussian_std: Standard deviation for Gaussian noise
# hw: Hamming weight of the secret (-1 for unknown). Need to be multiplied by k for MLWE
# error_type: Type of error distribution
# num_gen: Number of MLWE samples to generate
# seed: Random seed
# float_type: Floating-point type during reduction
# matrix_config: Matrix configuration for reduction. Options: "dual", "salsa", "original"
# reduction_samples: Number of samples for reduction. Null to optimized samples, 0<n<1 fraction of total samples, or integer number of samples
# reduction_resampling: Whether to resample before reduction
# continuous_reduction: Whether to use continuous reduction (full parallelization with save/stop after specified hours)
# parallel_backend: Parallel backend to use ("thread", "process", "joblib"). Best are "process" or "joblib" because of CPU-heavy BKZ
# min_samples: Minimum number of samples
# num_matrices: Number of matrices (0 to use minimal number of matrices)
# reduction_max_size: Maximum size for reduction priority queue (-1 for saving only the best lattice basis, 0 for saving a basis with the best row vectors, >0 for priority queue saving)
# lookback: Lookback (number of steps that the reduction stalls)
# warmup_steps: Number of warmup steps (with Flatter)
# flatter_alpha: Alpha parameter for flatter reduction
# bkz_delta: BKZ delta parameter
# bkz_block_sizes: BKZ block sizes
# use_polish: Whether to use polish
# interleaved_steps: Number of interleaved steps
# penalty: Penalty parameter
# verbose: Verbosity flag
# train_percentages: Percentages of training data
# subsets_with_probs: Whether to prioritize samples based on probabilities
# model: Model type
# lr: Learning rate
# c_factor: Regularization factor (for Tukey)
# epsilon: Epsilon parameter (for Huber)
# max_iter: Maximum number of iterations (for Tukey and RANSAC)
# alpha: Alpha parameter
# warm_start: Whether to use warm start
# fit_intercept: Whether to fit intercept
# tol: Tolerance for convergence
# use_ransac: Whether to use RANSAC
# residual_factor: Residual factor for RANSAC
# max_trials: Maximum number of trials for RANSAC
# normalize_raw_secret: Whether to normalize raw secret

PARAMS_JSON=$(cat <<EOF
{
  "n": $n,
  "k": $k,
  "q": $q,
  "secret_type": "cbd",
  "eta": 2,
  "gaussian_std": 3,
  "hw": $hw,
  "error_type": "cbd",
  "num_gen": 4,
  "seed": 42,
  "float_type": "dd",
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
  "warmup_steps": 0,
  "flatter_alpha": 0.001,
  "bkz_delta": 0.99,
  "bkz_block_sizes": "$bkz_block_size",
  "use_polish": true,
  "interleaved_steps": 0,
  "penalty": $penalty,
  "verbose": true,
  "train_percentages": [0.1, 0.3, 0.6, 1.0],
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
  --save_strategy "hour" \
  --save_every 40 \
  --stop_strategy "hour" \
  --stop_after 160 \
  --save_at_the_end \
  --train_secret_types "cbd" \
  --hw_range 1:10:1