#!/bin/bash

# Dimensions
n_values=(32)
k_values=(2)

# Moduli
q_values=(3329)

# Secret types and parameters
hw_values=(-1)
secret_type="cbd"

# Preprocessing and reduction parameters
max_size=20
reduction_samples=0.875
matrix_configs=("dual")
penalty_values=(4)
bkz_block_sizes=("20:40:10")

DATA_DIR="../data"
mkdir -p "$DATA_DIR"

CONTAINER_IMAGE="/d/hpc/projects/FRI/cb17769/lwe_container.sif"
#CONTAINER_IMAGE="/vol/csedu-nobackup/project/cbassotto/lwe_container.sif"

# Counter for job numbering
JOB_COUNTER=1

# Nested loops over all parameter combinations
for n in "${n_values[@]}"; do
  for k in "${k_values[@]}"; do
    for q in "${q_values[@]}"; do
      for hw in "${hw_values[@]}"; do
        for penalty in "${penalty_values[@]}"; do
          for matrix_config in "${matrix_configs[@]}"; do
            for bkz_block_size in "${bkz_block_sizes[@]}"; do
              
              # Construct JSON parameters
              sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lwe_${n}_${k}_${q}_${secret_type}_${matrix_config}
#SBATCH --output=outputs/attack_${n}_${k}_${q}_${secret_type}_${matrix_config}_%j.out
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=all

# Define parameters with the dynamic checkpoint path
PARAMS_JSON='{
  "n": '$n',
  "q": '$q',
  "k": '$k',
  "hw": '$hw',
  "secret_type": "'$secret_type'",
  "error_type": "cbd",

  "eta": 2,
  "gaussian_std": 3,

  "float_type": "d",
  "seed": 42,

  "matrix_config": "'$matrix_config'",

  "num_gen": 4,
  "num_matrices": -1,
  "reduction_max_size": '$max_size',
  "reduction_samples": '$reduction_samples',
  "reduction_resampling": false,
  "lookback": 3,
  "warmup_steps": 10,
  "flatter_alpha": 0.001,
  "interleaved_steps": 0,
  "bkz_block_sizes": "'$bkz_block_size'",
  "penalty": '$penalty',

  "model": "tukey",
  "tol": 0.0001,
  "train_percentages": [0.1, 0.3, 0.6, 1.0],

  "verbose": true,
  "save_to": "'$DATA_DIR'",
  "continuous_reduction": true
}'

export PYTHONUNBUFFERED=1

singularity exec ${CONTAINER_IMAGE} python attack.py \
  --params "\$PARAMS_JSON" \
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
EOF

              # Increment counter
              ((JOB_COUNTER++))
            done
          done
        done
      done
    done
  done
done

echo "Submitted $((JOB_COUNTER-1)) jobs."