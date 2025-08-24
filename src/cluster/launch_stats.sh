#!/bin/bash

CONTAINER_IMAGE="/d/hpc/projects/FRI/cb17769/lwe_container.sif"
#CONTAINER_IMAGE="/vol/csedu-nobackup/project/cbassotto/lwe_container.sif"

files=(
  #"../data/data_n_256_k_2_s_cbd_9e60e.pkl" # Kyber 1
  #"../data/data_n_256_k_2_s_cbd_ca888.pkl" # kyber 2
  #"../data/data_n_256_k_3_s_cbd_c7e5c.pkl" # kyber 3

  # VERDE
  #"../data/data_n_256_k_1_s_binary_9ac58.pkl"
  #"../data/data_n_256_k_1_s_binary_b8d75.pkl"
  #"../data/data_n_350_k_1_s_binary_b4d5e.pkl"
  #"../data/data_n_350_k_1_s_binary_bef5b.pkl"

  "../data/data_n_1024_k_1_s_ternary_c7322.pkl"
  #"../data/data_n_1024_k_1_s_ternary_c6b91.pkl"
  #"../data/data_n_1024_k_1_s_ternary_f4cb2.pkl"
)

for file in "${files[@]}"; do

  # Construct JSON parameters and submit sbatch for each file
  sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=train_$(basename "$file")
#SBATCH --output=outputs/train_$(basename "$file")_%j.out
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=12
#SBATCH --partition=all
#SBATCH --mem=64G

# Define parameters with the dynamic checkpoint path
PARAMS_JSON='{
  "secret_type": "cbd",
  "hw": -1,
  "error_type": "cbd",

  "eta": 2,
  "gaussian_std": 3.0,

  "seed": null
}'

export PYTHONUNBUFFERED=1

singularity exec ${CONTAINER_IMAGE} python stats.py \
  --params "\$PARAMS_JSON" \
  --num_secrets 0 \
  --data_path "$file" \
  --top_percent 1
EOF
done


#             JOBID  PARTITION                                               NAME       USER ST       TIME  NODES NODELIST(REASON)
#          59809308        all               train_data_n_256_k_2_s_cbd_ca888.pkl    cb17769 PD       0:00      1 (Priority)
#          59809309        all               train_data_n_256_k_3_s_cbd_c7e5c.pkl    cb17769 PD       0:00      1 (Priority)
#          59809307        all               train_data_n_256_k_2_s_cbd_9e60e.pkl    cb17769 PD       0:00      1 (Resources)
#          59827275        all            train_data_n_350_k_1_s_binary_b4d5e.pkl    cb17769 PD       0:00      1 (Priority)
#          59827290        all            train_data_n_350_k_1_s_binary_b4d5e.pkl    cb17769 PD       0:00      1 (Priority)
#          59827294        all            train_data_n_350_k_1_s_binary_bef5b.pkl    cb17769 PD       0:00      1 (Priority)
#          59827295        all            train_data_n_350_k_1_s_binary_bef5b.pkl    cb17769 PD       0:00      1 (Priority)
#          59827263        all            train_data_n_256_k_1_s_binary_9ac58.pkl    cb17769 PD       0:00      1 (Priority)
#          59827265        all            train_data_n_256_k_1_s_binary_b8d75.pkl    cb17769 PD       0:00      1 (Priority)
#          59827266        all            train_data_n_256_k_1_s_binary_b8d75.pkl    cb17769 PD       0:00      1 (Priority)
#          59827260        all            train_data_n_256_k_1_s_binary_9ac58.pkl    cb17769 PD       0:00      1 (Priority)
#          59809304        all               train_data_n_256_k_3_s_cbd_c7e5c.pkl    cb17769  R      27:52      1 wn169
#          59809303        all               train_data_n_256_k_2_s_cbd_ca888.pkl    cb17769  R    7:37:23      1 wn163
#          59809299        all               train_data_n_256_k_3_s_cbd_c7e5c.pkl    cb17769  R    9:33:38      1 wn108
#          59809298        all               train_data_n_256_k_2_s_cbd_ca888.pkl    cb17769  R    9:33:57      1 wn165
#          59809297        all               train_data_n_256_k_2_s_cbd_9e60e.pkl    cb17769  R    9:34:35      1 wn102
#          59790846        all                lwe_1024_1_41223389_ternary_dual_10    cb17769  R   19:29:15      1 wn110
#          59790842        all              lwe_1024_1_274887787_ternary_salsa_10    cb17769  R   21:12:10      1 wn101
#          59790841        all               lwe_1024_1_41223389_ternary_salsa_10    cb17769  R   21:25:57      1 wn105
#          59790830        all        lwe_1024_1_607817174438671_ternary_salsa_10    cb17769  R   21:28:11      1 wn103