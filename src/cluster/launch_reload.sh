#!/bin/bash

#CONTAINER_IMAGE="/d/hpc/projects/FRI/cb17769/lwe_container.sif"
CONTAINER_IMAGE="/vol/csedu-nobackup/project/cbassotto/lwe_container.sif"

files=(
  "/vol/csedu-nobackup/project/cbassotto/train_data/data_n_256_k_1_s_binary_78b66.pkl"
)

for file in "${files[@]}"; do
  # Extract the secret type from the filename
  secret_type=$(basename "$file" | sed -n 's/.*_s_\([a-zA-Z]*\)_.*\.pkl/\1/p')

  # Construct JSON parameters and submit sbatch for each file
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_$(basename "$file")
#SBATCH --output=outputs/train_$(basename "$file")_%j.out
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=12
#SBATCH --partition=csedu
#SBATCH --qos=csedu-large
#SBATCH --mem=31G

# Define parameters with the dynamic checkpoint path
PARAMS_JSON='{
  "model": "tukey",
  "max_iter": 15000,
  "tol": 0.001,
  "train_percentages": [0.1, 0.3, 0.6, 1.0]
}'

export PYTHONUNBUFFERED=1

singularity exec --bind /vol/csedu-nobackup/project/cbassotto ${CONTAINER_IMAGE} python attack.py \
  --params "\$PARAMS_JSON" \
  --num_attacks 0 \
  --reload_from "$file" \
  --train_secret_types "$secret_type" \
  --num_training_repeats 10 \
  --hw_range 105:130:1
EOF
done