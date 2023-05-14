#!/bin/bash

target=$1
temp=$2

ulimit -s unlimited

export CRFALIGN_HOME=/gpfs/deepfold/project/crfalign.sangjin

export PATH=$CRFALIGN_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CRFALIGN_HOME/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$CRFALIGN_HOME/lib:$PYTHONPATH

export HHLIB=/gpfs/deepfold/programs/build/hhsuite-2.0.13

module purge
module load intel/compiler/latest
module load intel/compiler-rt/latest
module load intel/mpi/latest
#module load gnu9 openblas

source $HOME/anaconda3/etc/profile.d/conda.sh
while [ ! -z $CONDA_PREFIX ]; do conda deactivate; done
conda activate py39

align=$CRFALIGN_HOME/bin/align_hhcrf.py

$align $target $temp out


