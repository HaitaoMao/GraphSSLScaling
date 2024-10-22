#!/bin/bash

ratio=0.1

models=('MVGRLg') 
datasets=("github_stargazers" "reddit_threads" "ogbg-molhiv", "ogbg-ppa", "ogbg-code2" )
template='singularity exec --writable-tmpfs --nv /data/zhehua/SIF/mvgrl.sif python3 ./run_model.py --task SSGCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --ratio RATIO_PLACEHOLDER --downstream_ratio 0.1 --downstream_task loss --config_file random_config/mvgrlg'
commands=()

for model in "${models[@]}"; do
    for dataset in "${datasets[@]}"; do  
        #for exp in $(seq 0 5); do #1 and 0.5 already tested
        #    i=$(bc <<< "scale=6; 2^(-$exp)")
        for i in $(seq $ratio $ratio 1); do
            command="${template/MODEL_PLACEHOLDER/$model}"
            command="${command/DATASET_PLACEHOLDER/$dataset}"
            command="${command/RATIO_PLACEHOLDER/$i}"
            commands+=("$command")
        done
    done
done

for command in "${commands[@]}";do
    eval $command
done
#parallel -j 1 eval ::: "${commands[@]}"