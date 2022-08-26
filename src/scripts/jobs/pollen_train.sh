#!/bin/bash

# custom uses our trainer
# default uses detectron2 default trainer

PROGRAM=custom_train.py

#PROGRAM=default_train.py
VIRT_ENV_LOCATION=~/projects/def-juli/detectron_virt_env/

echo Training using option "$1"
command="module load cuda cudnn && source ${VIRT_ENV_LOCATION}/bin/activate && python ${PROGRAM} --bundle=$1"

output=$0_$1_$(date +%F_%H-%M-%S)_%N_%j.out
JOBNAME=$0.$1

echo Running "$command"

case "$2" in
       train)
             sbatch --job-name="${JOBNAME}"   --account="${SLURM_ACCOUNT}" --gres=gpu:v100l:1 --cpus-per-task=4 --mem=64G --time=48:00:00 --mail-user=chunqingliang@gmail.com --mail-type=ALL --output="outs/$output" --wrap="${command}"
            ;;
       predict)
            sbatch --job-name="${JOBNAME}"   --account="${SLURM_ACCOUNT}" --gres=gpu:p100:1 --cpus-per-task=2 --mem=16000M --time=1-00:00 --output="outs/$output" --wrap="${command}"
           ;;
      validate)
            sbatch --job-name="${JOBNAME}"   --account="${SLURM_ACCOUNT}" --gres=gpu:p100:1 --cpus-per-task=2 --mem=16000M --time=00-02:00 --output="$output" --wrap="${command}"
           ;;
        *)
            echo $"Wrong action"
            exit 2

esac
