#!/bin/bash

models=('DGI' 'GRACE' 'MVGRL' 'CCA' 'BGRL' 'GBT' 'SUGRL' 'SFA' 'COSTA') #'AFGRL'
datasets=('Cora' 'CiteSeer' 'PubMed' 'Computers' 'Photo' 'CS' 'Physics')
template='singularity exec --nv ../SIF/bgpmv116.sif python3 ./run_model.py --task GCL --model MODEL_PLACEHOLDER --dataset DATASET_PLACEHOLDER --config_file random_config/'

commands=()

for ((i=0; i<20; i++)); do
    for dataset in "${datasets[@]}"; do
        for model in "${models[@]}"; do
            config_file="config_${i}"
            command="${template/MODEL_PLACEHOLDER/$model}"
            command="${command/DATASET_PLACEHOLDER/$dataset}"
            command="${command}${config_file}"
            commands+=("$command")
        done
    done
done

for command in "${commands[@]}";do
    echo $command
    done
# parallel -j 4 eval ::: "${commands[@]}"