#!/bin/bash

#SBATCH -J align
#SBATCH -p normal
#SBATCH -o logs/align.out
#SBATCH --gres-flags=enforce-binding

echo "alignments run"

log=./logs/align_log
rm -rf $log
touch $log

conda activate unifold

data_dir=database

/usr/bin/time -o $log --append \
    python generate_pkl_features.py \
    --fasta_dir ./example_data/fasta \
    --output_dir ./out \
    --data_dir $data_dir \
    --num_workers 1
    2>&1 | tee -a $log

echo "End `date`"
