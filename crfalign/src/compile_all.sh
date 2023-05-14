#!/bin/bash

#list=$1

export HHLIB=/gpfs/deepfold/programs/build/hhsuite-2.0.13

module purge
module load intel/compiler/latest
module load intel/compiler-rt/latest
module load intel/mpi/latest
module load python/3.9
#path=/share/protein/casp10/TS/server/TSWIR
#cd $path/
    
#list=list_select

#for i in `cat ${list}`
#do
    #echo "$line"
    #train=`echo $line | awk '{print $1}'`
    #fvalue=`echo $line | awk '{print $2}'`

    #echo "$i"
    #train=$i

    ./compile.sh
    ./make.sh

#done 
####################

