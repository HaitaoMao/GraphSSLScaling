#!/bin/bash

ratio=0.1

models=('GraphCL') 
datasets=("github_stargazers" "reddit_threads" "ogbg-molhiv" "ogbg-ppa" "ogbg-code2")
tasks=('original' 'loss') 
#datasets=("MUTAG" "MCF-7" "MOLT-4" "P388" "ZINC_full" "reddit_threads" "github_stargazers")
template='singularity exec --writable-tmpfs --nv /data/zhehua/SIF/mvgrl.sif python3 ./run_model.py --task SSGCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --downstream_ratio 0.1 --downstream_task TASK_PLACEHOLDER'
commands=()

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do  
        for task in "${tasks[@]}"; do  # added loop over tasks
            #for exp in $(seq 0 5); do #1 and 0.5 already tested
            #    i=$(bc <<< "scale=6; 2^(-$exp)")
            for i in $(seq $ratio $ratio 1); do
                command="${template/MODEL_PLACEHOLDER/$model}"
                command="${command/DATASET_PLACEHOLDER/$dataset}"
                command="${command/RATIO_PLACEHOLDER/$i}"
                command="${command/TASK_PLACEHOLDER/$task}"  # replaced TASK_PLACEHOLDER with actual task
                commands+=("$command")
            done
        done
    done
done

for command in "${commands[@]}";do
    echo $command
    eval $command
done
#parallel -j 1 eval ::: "${commands[@]}"
