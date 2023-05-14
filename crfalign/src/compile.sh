#/bin/bash

#train=$1

#ln -sf RTrees_hhm/RTrees_hhm_${train}.h RTrees_hhm.h

rm -f pCRF_pair_align.o  pCRF_pair_align.so
#g++ -fPIC -g -c pCRF_pair_align.cpp -o  pCRF_pair_align.o
#g++ -shared -o pCRF_pair_align.so pCRF_pair_align.o

make pCRF_pair_align.so 

#cp pCRF_pair_align.so ../lib/pCRF_pair_align_${train}.so
cp pCRF_pair_align.so ../lib

#mv pCRF_hhm_pair_align.so ../lib


