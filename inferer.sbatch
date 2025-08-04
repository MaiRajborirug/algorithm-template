#!/bin/bash
#SBATCH --job-name=inferer
#SBATCH --output=slurm_outputs/inferer-%A.out
#SBATCH --error=slurm_outputs/inferer-%A.err
#SBATCH --partition=batch
#SBATCH --qos=black
#SBATCH --nodes=1                       
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G               
#SBATCH --time=4-00:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mai.rajborirug@gmail.com

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

source activate base
conda activate py311_min

echo "== SLURM ARRAY TASK =="

# Run the distributed training examples
srun python inferer_ver7.py 