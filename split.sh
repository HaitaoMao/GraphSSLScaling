#!/bin/bash

epochs=(10)
ratio=0.1
models=('GraphCL')
datasets=('MCF-7')
template='singularity exec --nv /data/chunlinFeng/SIF/neural_scaling.sif python3 ./run_model.py --task SSGCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --epochs EPOCH_PLACEHOLDER'
commands=()

for epoch in "${epochs[@]}"; do
    for i in $(seq 0.1 $ratio 1); do
        for dataset in "${datasets[@]}"; do
            for model in "${models[@]}"; do
                command="${template/MODEL_PLACEHOLDER/$model}"
                command="${command/DATASET_PLACEHOLDER/$dataset}"
                command="${command/RATIO_PLACEHOLDER/$i}"
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