# -*- coding: utf-8 -*-
"""
Created on Sun May 19 14:41:35 2013

@author: dgevans
"""
import sys
import numpy as np
from Spline import Spline
from scipy.integrate import quad
from numpy.polynomial import hermite
from copy import deepcopy
from mpi4py import MPI
import itertools
from scipy.stats import beta

class parameters(object):
    
    gamma = 2.0
    
    delta = 0.05
    
    beta  = 1.0/1.04
    
    z = np.array([1.0])
    
    q = 1.0/np.array([1.03,0.99,0.9])
    
    P = np.array([[0.917, 1.0-0.917],
                  [1-0.5, 0.5]])
                  
    nMax = 50
    
    mMax = 1000
    
    
def makeDomain(Para):
    '''
    Makes the domain for para
    '''
    Para.domain = []
    nMax = Para.nMax
    mMax = Para.mMax
    print 'making domain'
    sys.stdout.flush()
    for i in range(3*nMax*mMax):
        m = i/(3*nMax) + 5
        i = i%(3*nMax)
        n = i/3 + 5
        s = i%3
        Para.domain.append((s,n,m))
    print 'done'
    sys.stdout.flush()
    return Para
    

class posterioDistriubtion(object):
    
    def __init__(self,p_d,pdf,d):
        self.mu = Spline(p_d,pdf,d)
        self.moments = {}
        
    def fit(self,p_d,pdf,d):
        self.mu.fit(p_d,pdf,d)
        
    def __call__(self,p_d):
        if isinstance(p_d,float) or len(p_d) ==1:
            return self.mu.feval1d(float(p_d))
        else:
            return self.mu(p_d)
            
    def getMoment(self,m):
        if not self.moments.has_key(m):
            if m == 1:
                self.moments[1] = quad(lambda p_d: p_d*self(p_d),0.0,1.0,full_output=1)[0]
            else:
                Ep_d = self.getMoment(1)
                mom = quad(lambda p_d: (p_d-Ep_d)**m*self(p_d),0.0,1.0,full_output=1)[0]
                if mom < 0:
                    self.moments[m] = -(-mom)**(1.0/m)
                else:
                    self.moments[m] = mom**(1.0/m)
        return self.moments[m]
    
    def getMoments(self,m = [1,2,3]):
        return np.array(map(self.getMoment,m))
                
class ValueFunction(object):

    def __init__(self,stateHist,Vf,deg=[1,1,1]):
        def getMoments(stateHistItem):
            return stateHistItem[1][1].getMoments()
        def getS(stateHistItem):
            return stateHistItem[1][0]
        def getV(stateHistItem):
            return Vf(stateHistItem[1])[0]
            
        self.deg = deg
    
        w = MPI.COMM_WORLD
        rank = w.Get_rank()
        size = w.Get_size()        
        N = len(stateHist)
        n = N/size
        r = N%size
        
        my_States = itertools.islice(stateHist.iteritems(),rank*n+min(rank,r),(rank+1)*n+min(rank+1,r))                
        my_domain = np.vstack(itertools.imap(getMoments,my_States))
        
        slist = np.hstack(itertools.imap(getS,stateHist.iteritems()))
        
        my_States = itertools.islice(stateHist.iteritems(),rank*n+min(rank,r),(rank+1)*n+min(rank+1,r))
        my_V = np.hstack(itertools.imap(getV,my_States))
        
        #now combine everything
        V = np.zeros(len(stateHist))
        domain = np.zeros((len(stateHist),my_domain.shape[1])) 
        w.Allgather(my_V,V)
        w.Allgather(my_domain,domain)        
        self.Vs = V
        #now fit things
        self.b = []
        for s in range(0,3):
            X = domain[slist==s,:]
            y = V[slist==s]
            A = hermite.hermvander3d(X[:,0],X[:,1],X[:,2],self.deg)
            self.b.append(np.linalg.lstsq(A,y)[0].reshape((self.deg[0]+1,self.deg[1]+1,self.deg[2]+1)))
    
    def __call__(self,state):
        '''
        Calculate the value at a given state
        '''
        s,mu = state
        
        x = mu.getMoments()
        return hermite.hermval3d(x[0],x[1],x[2],self.b[s])
        
    def __add__(self,Vf1):
        if not isinstance(Vf1,ValueFunction):
            return NotImplemented
        else:
            Vf0 = deepcopy(self)
            for s in range(0,3):
                Vf0.b[s] += Vf1.b[s]
            return Vf0
    __radd__ = __add__
    
    def __mul__(self,xi):
        if not isinstance(xi,float):
            return NotImplemented
        else:
            Vf = deepcopy(self)
            for s in range(0,3):
                Vf.b[s] *= xi
            return Vf
    __rmul__ = __mul__
    
    
class posterioDistriubtionBeta(object):
    
    def __init__(self,n,m):
        self.n = n
        self.m = m
        self.moments = {}
        
    def __call__(self,p_d):
        
        return beta(self.n,self.m).pdf(p_d)
            
    def getMoment(self,mom):
        n = self.n
        m = self.m
        mu = beta(n,m)
        if not self.moments.has_key(mom):
            if mom == 1:
                self.moments[mom]= n*1.0/(n+m)
            elif mom == 2:
                self.moments[mom] = ( n*m*1.0/((n+m)**2*(n+m+1)) )**(0.5)
            elif mom == 3:
                Ex = mu.moment(1)
                Ex2 = mu.moment(2)
                Ex3 = mu.moment(3)
                moment = ( Ex3 - 3*Ex2*Ex+2*Ex**3 )
                if moment< 0:
                    self.moments[mom] = -(-moment)**(1.0/3)
                else:
                    self.moments[mom] = moment**(1.0/3)
        return self.moments[mom]
    
    def getMoments(self,m = [1,2,3]):
        return np.array(map(self.getMoment,m))
    