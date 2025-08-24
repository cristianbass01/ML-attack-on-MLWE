#!/bin/bash

# (Toy) n=80, log_2 q = 7: (140MB compressed --> 1GB uncompressed): https://dl.fbaipublicfiles.com/lwe-benchmarking/80_7_omega15_lwe_data_prefix.tar.gz
# (Kyber) n=256, k=2, log_2 q = 12 (625MB compressed --> 5.4GB uncompressed): https://dl.fbaipublicfiles.com/lwe-benchmarking/256_k2_12_omega4_mlwe_data_prefix.tar.gz
# (Kyber) n=256, k=2, log_2 q = 28 (5.1GB compressed --> 14GB uncompressed): https://dl.fbaipublicfiles.com/lwe-benchmarking/256_k2_28_omega4_mlwe_data_prefix.tar.gz
# (Kyber) n=256, k=3, log_2 q = 35 (12GB compressed --> 34GB uncompressed): https://dl.fbaipublicfiles.com/lwe-benchmarking/256_k3_35_omega4_mlwe_data_prefix.tar.gz
# (HE) n=1024, log_2 q = 26 (5GB compressed --> 21GB uncompressed): https://dl.fbaipublicfiles.com/1024_26_omega10_rlwe_data_prefix.tar.gz
# (HE) n=1024, log_2 q = 29 (6 GB compressed --> 24GB uncompressed): https://dl.fbaipublicfiles.com/1024_29_omega10_rlwe_data_prefix.tar.gz
# (HE) n=1024, log_2 q = 50 (18GB compressed --> 46GB uncompressed): https://dl.fbaipublicfiles.com/1024_50_omega10_rlwe_data_prefix.tar.gz
case=$1

case $case in
  toy)
    n=80
    k=1
    logq=7
    omega=15
    problem_type=lwe
    ;;
  kyber_1)
    n=256
    k=2
    logq=12
    omega=4
    problem_type=mlwe
    ;;
  kyber_2)
    n=256
    k=2
    logq=28
    omega=4
    problem_type=mlwe
    ;;
  kyber_3)
    n=256
    k=3
    logq=35
    omega=4
    problem_type=mlwe
    ;;
  he_1)
    n=1024
    k=1
    logq=26
    omega=10
    problem_type=rlwe
    ;;
  he_2)
    n=1024
    k=1
    logq=29
    omega=10
    problem_type=rlwe
    ;;
  he_3)
    n=1024
    k=1
    logq=50
    omega=10
    problem_type=rlwe
    ;;
  *)
    echo "Invalid case. Please choose from: toy, kyber_1, kyber_2, kyber_3, he_1, he_2, he_3."
    exit 1
    ;;
esac

if [ $k -ne 1 ]; then
  k_suffix="_k${k}"
else
  k_suffix=""
fi

data_path=./n${n}${k_suffix}_logq${logq}

if [ -f "$data_path/data.prefix" ]; then
  echo "data.prefix is in data_path"
else
  url_path=https://dl.fbaipublicfiles.com/lwe-benchmarking/${n}${k_suffix}_${logq}_omega${omega}_${problem_type}_data_prefix.tar.gz
  
  tmp_file=/tmp/n${n}${k_suffix}_logq${logq}_data.tar.gz
  wget $url_path -O $tmp_file
  mkdir -p $data_path
  tar -xvf $tmp_file --directory $data_path/
  rm $tmp_file
fi
