#!/bin/bash
#SBATCH --job-name=lwe_toy_example
#SBATCH --output=outputs/attack_toy_example_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=all

### TO BE RUN WITH: sbatch toy_example.sh ###

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
# parallel_backend: Parallel backend to use ("thread", "process", "joblib")
# min_samples: Minimum number of samples
# num_matrices: Number of matrices (0 to use minimal number of matrices)
# reduction_max_size: Maximum size for reduction priority queue
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
  "n": 32,
  "k": 2,
  "q": 3329,
  "secret_type": "cbd",
  "eta": 2,
  "gaussian_std": 3,
  "hw": -1,
  "error_type": "cbd",
  "num_gen": 4,
  "seed": null,
  "float_type": "d",
  "matrix_config": "dual",
  "reduction_samples": null,
  "reduction_resampling": true,
  "continuous_reduction": false,
  "parallel_backend": "thread",
  "min_samples": 0,
  "num_matrices": 0,
  "reduction_max_size": 1000,
  "lookback": 3,
  "warmup_steps": 10,
  "flatter_alpha": 0.001,
  "bkz_delta": 0.99,
  "bkz_block_sizes": "10:40:10",
  "use_polish": true,
  "interleaved_steps": 0,
  "penalty": 4,
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
  "normalize_raw_secret": true
}
EOF
)

export PYTHONUNBUFFERED=1

singularity exec ${CONTAINER_IMAGE} python attack.py \
  --params "$PARAMS_JSON" \
  --num_attacks 1 \
  --attack_strategy "tour" \
  --attack_every 1 \
  --save_strategy "no" \
  --save_every 0 \
  --stop_strategy "hour" \
  --stop_after 4 \
  --save_at_the_end \
  --train_secret_types "cbd" \
  --hw_range 1:15:1