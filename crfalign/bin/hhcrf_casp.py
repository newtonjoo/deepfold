#!/usr/bin/env python

import os, sys
import traceback
import argparse
import numpy as np
import pandas as pd
from mpi4py import MPI
from hhcrf import *
import hhcrf_pred as hh
from hhcrf_pred import *
from mydata import *

import glob
#import matplotlib.pyplot as plt

pd.set_option('display.max_columns',30)
pd.set_option('display.max_rows',10)
#pd.set_option('display.height', 1000)
pd.set_option('display.width', 1000)
pd.options.display.float_format = '{:,.2f}'.format

comm  = MPI.COMM_WORLD
nproc = comm.Get_size ()
me    = comm.Get_rank ()

# models (updated the end of May 2014)
#mpath    = '/gpfs/deepfold/project/hhcrf/models'
#rf_model = '%s/rf/250/rf.pkl' % mpath
#gb_model = '%s/gb/300/gb.pkl' % mpath
#li_model = '%s/li/li.pkl'     % mpath
#sv_model = '%s/sv/38/sv.pkl'  % mpath

#DefaultStructList = '/home/newton/database/tlist/list.test'

#database   = '/share/database'
#database   = '/lustre/protein/database'
database   = '/mnt/user/protein/database'

hh.database   = database
hh.hhmroot    = '%s/fasta/hhm'  % (database)
hh.ss3root    = '%s/struct/ss3' % (database)
hh.sa3root    = '%s/struct/sa3' % (database)

std=sys.stdout
err=sys.stderr

if __name__ == '__main__':

    #database = '/lustre/protein/database'
    #database = '/share/database'

    targetID, StructList, args  = getArgs ()

    sortKey = args.sortKey

    #print  targetID, StructList

    hhmpath  = './'
    ssfile   = '%s.ss2' % targetID
    safile   = '%s.a3' % targetID
    #safile   = '%s.sa2' % targetID
    target   = feature (targetID, hhmpath, ssfile, safile)

#   if useCath:
#       hhmroot = '%s/fasta_cath/hhm'  % (database)
#       ss3root = '%s/struct_cath/ss3' % (database)
#       sa3root = '%s/struct_cath/sa3' % (database)
#       StructList = '/home/newton/database/tlist/list.cath'
#       err.write( "use Cath list"

    TIC = MPI.Wtime ()

    pred = hhcrf_pred (nstate=5, mode=0, alignout=False, dataCheck=False, verbose=False) #check hhcrf_pred.py

    if args.error:
        err.write('Error checking start\n')
        pred.read_StructList (StructList, ntemp=1000)
    else:
        if args.shuffle: 
            shuffle = True
        else: 
            shuffle = False

        pred.read_StructList (StructList, shuffle=shuffle)

    if me==0: pred.header()

    if me==0:
        #logfile = '%s.log' % targetID
        logfile = 'hhcrf.log'
        log = open(logfile, 'w')
        log.write('HHCRF Fold Recognition Based on Conditional Random Field Machine\n')
        log.write('command = %s\n'  % (' '.join(sys.argv)))
        log.write('Target = %12s\n' % targetID)
        log.write('Ntemps = %12d\n' % (pred.nstruct))
        #log.write('mact   = %12s\n' % (pred.mact))
        log.write("\n")

        log.write('list     = %s\n' % (StructList))
        log.write('database = %s\n' % (hh.database))
        #log.write('rf_model = %s\n' % (hh.rf_model))
        #log.write('gb_model = %s\n' % (hh.gb_model))
        #log.write('sv_model = %s\n' % (hh.sv_model))
        #log.write('li_model = %s\n' % (hh.li_model))
        
    if me == 0: 
        log.write('Start...\n')
        log.flush()

    tic = MPI.Wtime ()
    pred.do_align (target)
    toc = MPI.Wtime ()

    if me == 0: 
        log.write('Alignment Done (me=%4d)... %12.4fs\n' % (me, toc-tic))
        log.flush()
    print('Alignment Done (me=%4d)... %12.4fs' % (me, toc-tic))

    tic = MPI.Wtime ()
    pred.reduce ()
    toc = MPI.Wtime ()
    print('%4d Reducing Done... %12.4fs' % (me, toc-tic))
    if me == 0: 
        log.write('Reducing Done... %12.4fs\n' % (toc-tic))
        log.flush()

    #pred.ranking (method='zscore')
    if me == 0:
        #try:
        #    log.write('Machine Prediction Start...\n')
        #    log.flush()

        #    pred.pred_all (log)
        #    next = True
        #except Exception as e:
        #    err.write('Fail in Machine Prediction...\n')
        #    traceback.print_exc(file=sys.stderr)
        #    next = False

        pred.df['psum'] = [pred.ali[k].probs_sum for k in xrange(pred.nstruct)]
        pred.df['psum2'] = [pred.ali[k].probs_sum2 for k in xrange(pred.nstruct)]
        pred.df['nmatch2'] = [pred.ali[k].nmatch2 for k in xrange(pred.nstruct)]

    comm.barrier()

    if me == 0 and next:
        # calc qa and argsort
        #pred.df['qa'] = (pred.df['rf'] + pred.df['gb'] + pred.df['sv'] + pred.df['li'])/4.

        #pred.ranks    = np.argsort(pred.df['qa']).tolist()
        #pred.ranks    = np.argsort(pred.df[sortKey]).tolist()
        pred.ranks    = np.argsort(pred.df['psum']).tolist()

        pred.ranks.reverse()

        #outfile = '%s.hhcrf' % (targetID)
        outfile = 'hhcrf.out'
        rankout = 'hhcrf.list'
        pred.print_full(outfile,rankout=rankout,ntop=50,sortKey=sortKey)
        
        log.write('Print Output done...\n')
        log.flush()

    if me == 0 and next:
        #print casp output
        pred.print_casp (ntop=50, outpath='ali')
        
        log.write('Alignment Print Out...\n')
        log.flush()

    if me == 0:
        datapath = 'data'
        if not os.path.exists(datapath): os.mkdir(datapath)
        datafile = '%s/hhcrf.pred' % (datapath)
        try:
            tic = MPI.Wtime()
            #data_save (pred, datafile)
            data_save_cpickle (pred, datafile)
            print('Data Saved... %12.4fs' % (toc-tic))
            toc = MPI.Wtime()
            
            log.write('Data Saved... %12.4fs\n' % (toc-tic))
            log.flush()
        except Exception as e:
            err.write( 'data_save error, check it! %s\n' % datafile)
            traceback.print_exc(file=sys.stderr)

    #print 'Output'
    #pred.predout()
    TOC= MPI.Wtime ()
    comm.barrier()

    print('All Done (me=%4d)... %12.4f' % (me, TOC-TIC))
    if me == 0: 
        log.write( 'All Done (me=%4d)... %12.4f\n' % (me, TOC-TIC))
    #pred.summary ()

    MPI.Finalize()

