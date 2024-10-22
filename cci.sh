#!/bin/bash
#SBATCH --job-name=estimate_time          # Job name
#SBATCH --partition=el8-rpi               # Partition
#SBATCH --gres=gpu:1                      # Request GPU resource
#SBATCH --time=06:00:00                   # Time limit hrs:min:sec
#SBATCH --output=slurm_logs/run_cci_%j.log  # Standard output and error log
#SBATCH --error=slurm_logs/run_cci_%j.err   # Error log

SCRIPT_DIR="$HOME/barn/KDD24-BGPM"
LOG_DIR="$HOME/scratch/log"
RAW_DATA_DIR="$HOME/scratch/raw_data"

# Start the enroot container and run the estimate_time.sh script
enroot start --root --rw \
    --mount $SCRIPT_DIR:/KDD24-BGPM \
    --mount $LOG_DIR:/scratch/log \
    --mount $RAW_DATA_DIR:/scratch/raw_data \
    bgpm sh -c '
ls -l /KDD24-BGPM &&
cd /KDD24-BGPM &&
pip list | grep torch &&
rm -f /usr/lib64/libstdc++.so.6 &&
ln -s /root/miniconda3/lib/libstdc++.so.6.0.29 /usr/lib64/libstdc++.so.6 &&
/bin/bash /KDD24-BGPM/split.sh
'