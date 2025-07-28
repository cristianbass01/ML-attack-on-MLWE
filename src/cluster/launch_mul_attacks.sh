#!/bin/bash

# Define parameter options (as arrays)
n_values=(128)
k_values=(1)
q_values=(3329)
eta=3
secret_types=("binary")
matrix_configs=("dual" "salsa")

penalty_values=(4)
bkz_block_sizes=("20:48:4")

DATA_DIR="../data"
mkdir -p "$DATA_DIR"

#CONTAINER_IMAGE="/d/hpc/projects/FRI/cb17769/lwe_container.sif"
CONTAINER_IMAGE="/vol/csedu-nobackup/project/cbassotto/lwe_container.sif"
ATTACK_FILE="/vol/csedu-nobackup/project/cbassotto/thesis/project/src/cluster/attack.py"

# Counter for job numbering
JOB_COUNTER=1

# Nested loops over all parameter combinations
for n in "${n_values[@]}"; do
  for k in "${k_values[@]}"; do
    for q in "${q_values[@]}"; do
      for secret_type in "${secret_types[@]}"; do
        for penalty in "${penalty_values[@]}"; do
          for matrix_config in "${matrix_configs[@]}"; do
            for bkz_block_size in "${bkz_block_sizes[@]}"; do
              
              # Construct JSON parameters
              sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lwe_${n}_${k}_${q}_${secret_type}_${matrix_config}_${penalty}
#SBATCH --output=outputs/attack_${n}_${k}_${q}_${secret_type}_${matrix_config}_${penalty}_%j.out
#SBATCH --time=6:00:00
#SBATCH --cpus-per-task=48
#SBATCH --partition=jupyterhub
#SBATCH --account=jupyterhub

# Define parameters with the dynamic checkpoint path
PARAMS_JSON='{
  "n": '$n',
  "q": '$q',
  "k": '$k',
  "eta": '$eta',
  "secret_type": "'$secret_type'",

  "seed": 0,

  "matrix_config": "'$matrix_config'",

  "num_gen": 4,
  "num_matrices": 47,
  "reduction_max_size": 1000,
  "reduction_factor": null,
  "reduction_resampling": true,
  "lookback": 3,
  "warmup_steps": 10,
  "flatter_alpha": 0.001,
  "bkz_block_sizes": "'$bkz_block_size'",
  "penalty": '$penalty',

  "model": "tukey",
  "train_percentages": [0.05, 0.1, 0.25, 0.5, 0.75, 1.0],

  "verbose": true,
  "save_to": "'$DATA_DIR'"
}'

export PYTHONUNBUFFERED=1

singularity exec ${CONTAINER_IMAGE} python $ATTACK_FILE \
  --params "\$PARAMS_JSON" \
  --num_attacks 1 \
  --attack_strategy "tour" \
  --attack_every 1 \
  --save_strategy "no" \
  --save_every 0 \
  --stop_strategy "hour" \
  --stop_after 5 \
  --save_at_the_end \
  --train_secret_types "binary" "ternary" "cbd"
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