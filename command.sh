#! /bin/bash
singularity exec --writable-tmpfs --nv  /data/zhehua/SIF/mvgrl.sif  python3 ./run_model.py --task SSGCL --model MVGRLg --dataset MUTAG --ratio 1 --config_file random_config/mvgrlg