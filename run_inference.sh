#!/bin/bash

#SBATCH -J inference
#SBATCH -p normal
#SBATCH -o logs/inference.out
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding

echo "inference run"

log=./logs/inference
rm -rf $log
touch $log

conda activate unifold

/usr/bin/time -o $log --append \
    python run_from_pkl.py \
    --pickle_paths $pickle \
    --model_names $model \
    --model_paths $param \
    --output_dir $out \
    --repr_path $repr_path \
    --random_seed $random_seed \
    2>&1 | tee -a $log

if [[ ! -z "${mv_dir}" ]]; then
    mv $out/prot_00000 $mv_dir
fi

echo "End `date`"
