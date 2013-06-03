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
    
s0 = 0
mu0 = bayesian.approximatePosterior(lambda p_d:0.5*p_d)
#mu0 = primitives.posterioDistriubtionBeta(5,5)
stateHist = bayesian.drawSamplePaths(s0,mu0,Para,N=32,T=10000,skip=100)



if rank == 0:
    print len(stateHist)
    print 'Solving Bellman'
    sys.stdout.flush()
    print 'computing state momemnts'
    

bayesian.computeStateMoments(stateHist,Para)

if rank==0:
    print 'done'
    print 'getting initial value function'

T = bayesian.BayesianBellmanMap(Para)

V0f = bellman.approximateREValueFunction(Para)

def V0(state):
    s,mu = state
    p_d = mu.getMoment(1)
    return V0f[s](p_d)

V1 = T(V0)
Vdeg = [2,2,2]
V = primitives.ValueFunction(stateHist,V1,deg=Vdeg)

if rank==0:
    print 'done'
    print 'iterating value function'

for i in range(0,1000):
    Vnew = primitives.ValueFunction(stateHist,T(V),deg=Vdeg)
    diff =  np.linalg.norm(Vnew.Vs-V.Vs)
    if rank == 0:
        Vold = np.hstack(map(lambda x:V(x[1]),stateHist.items()))
        print 'diff:',diff
        print np.linalg.norm(Vnew.Vs-Vold)
    sys.stdout.flush()
    if diff < 1e-10:
        break
    V = xi*Vnew+(1-xi)*V

if rank == 0:
    fout = file('Vf.dat','w')
    cPickle.dump(V,fout)
    fout.close()
    fout = file('stateHist'+str(rank)+'.dat','w')
    cPickle.dump((size,stateHist),fout)
    fout.close()
    
