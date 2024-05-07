#! /bin/bash

singularity exec --nv /data/chunlinFeng/SIF/neural_scaling.sif python3 ./run_model.py --task SGC  --model GraphCL --dataset ZINC_full --config_file random_config/config_1 
# singularity exec --writable-tmpfs --nv  /data/zhehua/SIF/mvgrl.sif  python3 ./run_model.py --task SSGCL --model MVGRLg --dataset MUTAG --ratio 1 --config_file random_config/mvgrlg

