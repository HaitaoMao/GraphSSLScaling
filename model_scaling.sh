#!/bin/bash

# Assign variables
MODEL=$1
DATASET=$2
SEED=$3
CONFIG=$4
# Define the task command
TASK_CMD="python3 ./run_model.py --task SSGCL --model ${MODEL} --dataset ${DATASET} --ratio 1.0 --seed ${SEED} --downstream_ratio 0.1 --downstream_task both --config_file config_model/${CONFIG}"

echo "Running command: ${TASK_CMD}"
eval ${TASK_CMD}