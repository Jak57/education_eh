#!/bin/bash
# SBATCH -c 6  # Number of Cores per Task
#SBATCH --mem=8192  # Requested Memory
#SBATCH -p cpu  # Partition
#SBATCH -t 12:00:00  # Job time limit
#SBATCH -o slurm-%j.out  # %j = job ID
#SBATCH -e slurm-%j-error.out

echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"

python3 -m venv myenv
source myenv/bin/activate

echo "Installing requirements"
pip install -r requirements.txt
echo "Finished installing requirements"
python L1_anthropic_common_summary.py

echo "End time: $(date)"