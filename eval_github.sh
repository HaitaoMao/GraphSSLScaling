
#!/bin/bash
#models=("81256" "22405" "52562" "18956" "13733" "62848" "2272")
models=("81256")
template='singularity exec --writable-tmpfs --nv /data/zhehua/SIF/mvgrl.sif python3 ./run_model.py --task SSGCL --model MVGRLg --dataset github_stargazers --exp_id EVAL_PLACEHOLDER --train False --downstream_ratio 0.1  --config_file random_config/mvgrl_gpu7'
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