#!/bin/bash

#SBATCH -p normal
#SBATCH --cpus-per-task=8
##SBATCH --gres-flags=enforce-binding

echo "$fasta run"

log=./logs/log_$fasta
rm -rf $log
touch $log

conda activate unifold

data_dir=database

if [ -f "$output_path/$fasta/features.pkl" ]; then
    echo "$output_path/features.pkl exist."
else
    /usr/bin/time -o $log --append \
        python generate_pkl_one_features.py \
        --fasta_path $fasta_path \
        --output_dir $output_path \
        --data_dir $data_dir \
        --num_workers 8
        2>&1 | tee -a $log
fi

echo "End `date`"
