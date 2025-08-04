#!/bin/bash
#SBATCH --job-name=synthrad_inference
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Load modules (adjust based on your cluster)
module load cuda/11.8
module load python/3.9

# Activate conda environment (adjust path and env name)
source /path/to/your/conda/etc/profile.d/conda.sh
conda activate your_env_name

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Create logs directory
mkdir -p logs

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "CUDA: $(nvcc --version)"

# Run inference
python inferer_ver7.py

echo "Inference completed!" 