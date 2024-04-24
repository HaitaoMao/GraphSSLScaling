#! /bin/bash
singularity exec --writable-tmpfs --nv  /data/zhehua/SIF/mvgrl.sif  python3 ./run_model.py --task SSGCL --model MVGRLg --dataset reddit_threads --config_file random_config/mvgrlg