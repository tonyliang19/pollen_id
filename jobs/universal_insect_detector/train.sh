#!/bin/bash
source ../.secret.env

PROGRAM=train_universal_insect_detector.py

echo Training using option $1
command="module load cuda cudnn && source ${VIRT_ENV_LOCATION}/bin/activate && ${PROGRAM} $1 -v"

output=$0_$1_$(date +%F_%H-%M-%S)_%N_%j.out

echo Running "$command"

case "$1" in
        fetch| push)
            sbatch --job-name=$0$1 --account=${SLURM_ACCOUNT} --cpus-per-task=1 --mem=4000 --time=0-05:00 --output=$output --wrap="${command}"
            ;;
       train)
             sbatch --job-name=$0$1   --account=${SLURM_ACCOUNT}   --gres=gpu:v100l:1 --cpus-per-task=8 --mem=40000M --time=0-16:00 --output=$output --wrap="${command} --gpu"
            ;;
        *)
            echo $"Wrong action"
            exit 1

esac
