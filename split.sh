#!/bin/bash

epochs=(10)
ratios=(0.03125 0.0625 0.125 0.25 0.5 1)

models=('JOAO')
datasets=('github_stargazers')
template='singularity exec --nv /data/chunlinFeng/SIF/neural_scaling.sif python3 ./run_model.py --task SSGCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --epochs EPOCH_PLACEHOLDER'
commands=()

for epoch in "${epochs[@]}"; do
    for ratio in "${ratios[@]}"; do
        for dataset in "${datasets[@]}"; do
            for model in "${models[@]}"; do
                command="${template/MODEL_PLACEHOLDER/$model}"
                command="${command/DATASET_PLACEHOLDER/$dataset}"
                command="${command/RATIO_PLACEHOLDER/$ratio}"
                command="${command/EPOCH_PLACEHOLDER/$epoch}"
                commands+=("$command")
            done
        done
    done
done

for command in "${commands[@]}"; do
    echo $command
done
parallel -j 1 eval ::: "${commands[@]}"