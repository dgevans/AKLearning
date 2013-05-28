# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:30:18 2013

@author: dgevans
"""

import primitives
import bellman
from scipy.stats import beta
import numpy as np
import bayesian
import sys
from mpi4py import MPI
import cPickle

Para = primitives.parameters()
xi = 0.1
#Para = primitives.makeDomain(Para)

#V0 = dict(itertools.izip(Para.domain,-10*np.ones(len(Para.domain))))

#Vf,cf,gf = bellman.solveBellmanEquation(V0,Para)
w = MPI.COMM_WORLD
rank = w.Get_rank()
size = w.Get_size()

def V0(state):
    s,mu = state
    p_d = mu.getMoment(1)
    T = bellman.BellmanMap(Para,p_d)
    from scipy.optimize import root
    return root(lambda V: T(V)-V,-10*np.ones(3)).x[s]
    
s0 = 0
mu0 = bayesian.approximatePosterior(beta(5,5).pdf)
stateHist = bayesian.drawSamplePaths(s0,mu0,Para,N=50,T=2000)

if rank == 0:
    print len(stateHist)
    print 'Solving Bellman'

T = bayesian.BayesianBellmanMap(Para)

V1 = T(V0)
V = primitives.ValueFunction(stateHist,V1)


for i in range(0,1000):
    Vnew = primitives.ValueFunction(stateHist,T(V))
    if rank == 0:
        print np.linalg.norm(Vnew.Vs-V.Vs)
    sys.stdout.flush()
    V = xi*Vnew+(1-xi)*V



fout = file('stateHist'+str(rank)+'.dat','w')
cPickle.dump((size,stateHist),fout)
fout.close()

if rank == 0:
    fout = file('Vf.dat','w')
    cPickle.dump(V,fout)
    fout.close()
    
