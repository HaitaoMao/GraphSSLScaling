#!/bin/bash

ratio=0.1

models=('MVGRLg') 
datasets=('MUTAG')
template='singularity exec --writable-tmpfs --nv /data/zhehua/SIF/mvgrl.sif python3 ./run_model.py --task SSGCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --config_file random_config/mvgrlg'
commands=()

for i in $(seq 0 $ratio 1); do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            command="${template/MODEL_PLACEHOLDER/$model}"
            command="${command/DATASET_PLACEHOLDER/$dataset}"
            command="${command/RATIO_PLACEHOLDER/$i}"
            commands+=("$command")
        done
    done
done

for command in "${commands[@]}";do
    echo $command
done
parallel -j 1 eval ::: "${commands[@]}"