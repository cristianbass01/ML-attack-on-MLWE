#!/bin/bash

# Get your current username
USER=$(whoami)

if [ "$#" -gt 0 ]; then
  JOB_NAME_FILTER="$1"
  echo "Cancelling jobs with name containing: $JOB_NAME_FILTER"
  squeue -u "$USER" --format="%.18i %.10P %.50j %.10u %.2t %.10M %.6D %R" | grep "$JOB_NAME_FILTER" | awk '{print $1}' | xargs -r scancel
else
  # List and cancel all jobs for your user
  echo "Cancelling all jobs for user: $USER"
  scancel -u "$USER"
fi