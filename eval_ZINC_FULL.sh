
#!/bin/bash
models=("34542"	"25980"	"61008"	"46587"	"89759"	"67451")
template='singularity exec --writable-tmpfs --nv /data/zhehua/SIF/mvgrl.sif python3 ./run_model.py --task SSGCL --model MVGRLg --dataset ZINC_full --exp_id EVAL_PLACEHOLDER --train False --downstream_ratio 0.03125 --downstream_task loss --config_file random_config/mvgrlg'
commands=()

for model in "${models[@]}"; do
            command="${template/EVAL_PLACEHOLDER/$model}"
            commands+=("$command")
done

for command in "${commands[@]}";do
    echo $command
    eval $command
done
#parallel -j 1 eval ::: "${commands[@]}"