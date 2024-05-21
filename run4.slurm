#!/bin/bash
#SBATCH --job-name=run_split_enroot       # Job name
#SBATCH --partition=el8-rpi               # Partition
#SBATCH --gres=gpu:1                      # Request GPU resource
#SBATCH --time=06:00:00                   # Time limit hrs:min:sec
#SBATCH --output=slurm_logs/run_cci_%j.log  # Standard output and error log
#SBATCH --error=slurm_logs/run_cci_%j.err   # Error log

# Define the path to the script directory
SCRIPT_DIR="$HOME/KDD24-BGPM"

# Start the enroot container and run the split.sh script
enroot start --root --rw --mount $SCRIPT_DIR:/data-valuation bgpm sh -c '
ls -l /data-valuation &&
cd /data-valuation &&
pip list | grep torch &&
rm -f /usr/lib64/libstdc++.so.6 &&
ln -s /root/miniconda3/lib/libstdc++.so.6.0.29 /usr/lib64/libstdc++.so.6 &&
/bin/bash /data-valuation/split4.sh'
