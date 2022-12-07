#!/bin/bash

#SBATCH -J train
#SBATCH -o logs/train.out
#SBATCH --gres-flags=enforce-binding

echo "train run"

log=./logs/train_log
rm -rf $log
touch $log

conda activate unifold

echo "Start `date`"

mpirun --oversubscribe -n $num_gpus python train.py 2>&1 | tee -a $log

echo "End `date`"
