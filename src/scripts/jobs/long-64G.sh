#!/bin/bash
JOBNAME=$0.$1
output=$0_$1_$(date +%F_%H-%M-%S)_%N_%j.out

#SBATCH --job-name=${JOBNAME}
#SBATCH --output=${output}
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:v100l:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user=chunqingliang@gmail.com 
#SBATCH --mail-type=ALL



PROGRAM=main.py
VIRT_ENV_LOCATION=~/projects/def-juli/detectron_virt_env/
cd $project/job-scripts

module load cuda cudnn
source ${VIRT_ENV_LOCATION}/bin/activate 

python ${PROGRAM} $1
