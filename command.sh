#! /bin/bash
singularity exec --writable-tmpfs --nv  /data/zhehua/SIF/mvgrl.sif  python3 ./run_model.py --task GCL --model MVGRLg --dataset MUTAG --config_file random_config/config_1