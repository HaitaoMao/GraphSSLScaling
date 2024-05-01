#! /bin/bash
singularity exec --nv /data/chunlinFeng/SIF/neural_scaling.sif python3 ./run_model.py --task SGC  --model GraphCL --dataset ZINC_full --config_file random_config/config_1 