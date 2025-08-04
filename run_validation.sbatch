#!/bin/bash
#SBATCH --job-name=synthrad_validation
#SBATCH --output=logs/validation_%j.out
#SBATCH --error=logs/validation_%j.err
#SBATCH --time=12:00:00
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

# Configuration paths (adjust these to match your setup)
CONFIG_PATH="/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/exp_configs/test_apex3.yaml"
CHECKPOINT_DIR="/media/prajbori/sda/private/github/proj_synthrad/algorithm-template/some_checkpoints/test20-testseed1-nomaskloss/epoch_500"

# Run validation
python trainer_simple.py \
    --config_path "$CONFIG_PATH" \
    --checkpoint_dir "$CHECKPOINT_DIR"

echo "Validation completed!" 