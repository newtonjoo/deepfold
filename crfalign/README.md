# crfalign
Protein sequence-structure alignment using conditional random field

HMM-HMM Alignment based on Conditional Random Field Machine

## Compile

1. Change directory path
  1. Modify the following line at src/pCRF_pair_align_rev5.cpp

```c++
char *progId="/net/newton/project/hhcrf/data/pCRF_hhm_rev5"
```

2. Bash Environment 

```/bin/sh
export HHLIB=/share/database/build/bin/hhsuite-2.0.13
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/net/newton/project/hhcrf/lib
export PYTHONPATH=$PYTHONPATH:/net/newton/project/hhcrf/lib
```

3. Compile and Build Library

```/bin/sh
% ./src/compile_rev5_all.sh
```
