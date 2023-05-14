#!/usr/bin/env python

from os.path import abspath, dirname, join
import sys
from hhcrf import *

if __name__ == '__main__':

    targetID = sys.argv[1]
    structID = sys.argv[2]

    try:
        outpath = sys.argv[3]
    except:
        outpath = './'

    try:
        pos = sys.argv[4]
    except:
        pos = 'one'


    hhmpath ='./'
    ssfile = '%s.ss2' % targetID
    #safile = '%s.sa2' % targetID
    safile = '%s.a3' % targetID

    t = feature (targetID, hhmpath, ssfile, safile)

    ssfile = '%s.ss3' % structID
    safile = '%s.sa3' % structID

    if not os.path.exists(ssfile):
        dv = structID[1:3]
        hhmpath ='/mnt/user/protein/database/fasta/hhm/%s' % dv
        ssfile = '/mnt/user/protein/database/struct/ss3/%s/%s.ss3' % (dv,structID)
        safile = '/mnt/user/protein/database/struct/sa3/%s/%s.sa3' % (dv,structID)

    s = feature (structID, hhmpath, ssfile, safile)

    print("source=", s.seq)
    print("target=", t.seq)
    #print s.seq
    #print t.seq

    #print s.hhmpath
    #print t.hhmpath

    #print s.prof
    #print t.prof

    myali = hhcrfAlignment (s,t, align=True, nstate=5, mode=0)
    #myali = hhcrfAlignment (s,t,align=True)

    #print
    #print myali
    #print myali.out
    #ij = myali.out["match_score_arr"]
    #myali.outmatch()
    #print ij
    #print len(ij)
    #for i in range(len(ij)):
    #    print '%4d %12.4f' % (i,ij[i])
    #print
    #print dir(myali)
    #print

    #print "Python Interface:"
    #print myali.scoreLine()
    #myali.outFasta()
    #myali.outPir()

    outfile = '%s/%s-%s.pir' % (outpath,targetID,structID)
    myali.outPir (outfile)

    outfile = '%s/%s-%s.ali' % (outpath,targetID,structID)
    myali.outPir (outfile,reSeq=True,pos=pos)

    outfile = '%s/%s-%s.map' % (outpath,targetID,structID)
    myali.outMap (outfile)

    outfile = '%s/%s-%s.ff' % (outpath,targetID,structID)
    myali.outmatch (outfile,reSeq=True,pos='ca')


