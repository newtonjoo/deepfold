#!/bin/bash

#train=$1 

rm -rf build pyCRF.so

#ln -sf ../lib/pCRF_pair_align_${train}.so pCRF_pair_align.so  
ln -sf ../lib/pCRF_pair_align.so pCRF_pair_align.so  

python setup.py build 
#cp build/*/pyCRF.so .
#cp build/*/pyCRF.so ../lib/pyCRF_${train}.so
cp build/*/pyCRF.*.so ../lib/.

