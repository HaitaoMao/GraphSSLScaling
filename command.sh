#! /bin/bash
singularity exec --nv /data/chunlinFeng/SIF/neural_scaling.sif python3 ./run_model.py --task SSGCL  --model GraphCL --dataset MUTAG --config_file random_config/config_1 