#!/bin/bash
set -e

DEF_FILE="lwe_container.def"
OUTPUT_DIR="/d/hpc/projects/FRI/cb17769"
OUTPUT_IMAGE="${OUTPUT_DIR}/lwe_container.sif"

singularity build "${OUTPUT_IMAGE}" "${DEF_FILE}"