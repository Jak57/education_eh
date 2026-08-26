#!/bin/bash
#SBATCH --gpus=1  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p gpu-preempt  # Partition
#SBATCH -t 24:00:00  # Job time limit
#SBATCH -o ./slurm_output_j/slurm-%j.out  # %j = job ID
#SBATCH -e ./slurm_output_j/slurm-%j-error.out


echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

export CUDA_LAUNCH_BLOCKING=1

python3 -m venv myenv
source myenv/bin/activate

echo "Installing requirements"
pip install -r requirements.txt
echo "Finished installing requirements"
python main.py

echo "End time: $(date)"