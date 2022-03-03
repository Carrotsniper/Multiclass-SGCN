#!/bin/bash
#SBATCH -N 1 # Request a single node
#SBATCH -c 4 # Request four CPU cores
#SBATCH --gres=gpu # Request one gpu
#SBATCH -p res-gpu-small # Use the res-gpu-small partition
#SBATCH --qos=short # Use the short QOS
#SBATCH -t 2-0 # Set maximum walltime to 2 day
#SBATCH --job-name=full_model # Name of the job
#SBATCH --mem=4G # Request 4Gb of memory

# Load the global bash profile
source /etc/profile

# Load your Python environment
# source pytorch_env/bin/activate

# Run the code
# -u means unbuffered stdio
# Connect to the visdom server on ncc1
# Replace PORT with the actual port of the Visdom server.

module load cuda/11.3


python train.py