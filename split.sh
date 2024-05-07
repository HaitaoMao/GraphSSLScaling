#!/bin/bash

ratio=0.1

models=('GraphCL' 'MVGRLg')
datasets=('MUTAG')
template_graphcl='singularity exec --nv /data/chunlinFeng/SIF/neural_scaling.sif python3 ./run_model.py --task sgc --model GraphCL --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --config_file random_config/graphcl'
template_mvgrlg='singularity exec --writable-tmpfs --nv /data/zhehua/SIF/mvgrl.sif python3 ./run_model.py --task ssgcl --model MVGRLg --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --config_file random_config/mvgrlg'

commands=()

for i in $(seq 0 $ratio 1); do
    for dataset in "${datasets[@]}"; do
        if [ "$task" == "sgc" ]; then
            command="${template_graphcl/DATASET_PLACEHOLDER/$dataset}"
        elif [ "$task" == "ssgcl" ]; then
            command="${template_mvgrlg/DATASET_PLACEHOLDER/$dataset}"
        else
            echo "Invalid task specified: $task"
            exit 1
        fi
        command="${command/RATIO_PLACEHOLDER/$i}"
        commands+=("$command")
    done
done

for command in "${commands[@]}"; do
    echo "$command"
done

parallel -j 1 eval ::: "${commands[@]}"