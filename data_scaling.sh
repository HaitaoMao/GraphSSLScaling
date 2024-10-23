#!/bin/bash

# Assign variables
MODEL=$1
DATASET=$2
RATIO=$3
SEED=$4
# Define the task command
TASK_CMD="python3 ./run_model.py --task SSGCL --model ${MODEL} --dataset ${DATASET} --ratio ${RATIO} --seed ${SEED} --downstream_ratio 0.1 --downstream_task both"

echo "Running command: ${TASK_CMD}"
eval ${TASK_CMD}
