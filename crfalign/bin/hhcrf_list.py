#!/bin/env python2.7

import os, sys
import argparse
import numpy
from mpi4py import MPI
#from hhcrf5 import *
from hhcrf import *
from hhcrf_pred import *

comm  = MPI.COMM_WORLD
nproc = comm.Get_size ()
me    = comm.Get_rank ()

DefaultStructList = '/home/newton/database/tlist/list.test'

database   = '/share/database'
hhmroot    = '%s/fasta/hhm'  % (database)
ss3root    = '%s/struct/ss3' % (database)
sa3root    = '%s/struct/sa3' % (database)

std=sys.stdout
err=sys.stderr

if __name__ == '__main__':

    targetID, StructList  = getArgs ()

    hhmpath  = './'
    ssfile   = '%s.ss2' % targetID
    #safile   = '%s.a3' % targetID
    safile   = '%s.a3' % targetID
    target   = feature (targetID, hhmpath, ssfile, safile)

    tic = MPI.Wtime ()
    pred = hhcrf_pred (StructList, nstate=3, mode=0, alignout=False, dataCheck=False, verbose=False)
    #print 'Do Align...'
    pred.do_align (target)
    #print 'Complete align....'
    pred.reduce ()
    pred.ranking (method='zscore')
    #print 'Output'
    pred.writeout2()

    toc = MPI.Wtime ()

    if me == 0:
        print('Total Wall Time = %12.4f seconds (%4d cpu cores)' % (toc-tic, nproc))

