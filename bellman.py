# -*- coding: utf-8 -*-
"""
Created on Sun May 19 16:33:20 2013

@author: dgevans
"""
import numpy as np
from scipy.optimize import root
from Spline import Spline
import itertools
import sys

class BellmanMap:
    
    def __init__(self,Para,p_d):
        self.Para = Para
        self.p_d = p_d
    
    def __call__(self,V):
        Vnew = np.zeros(3)
        for s in range(0,3):
            Vnew[s],_,_ = self.maximizeObjective(s,V)
        return Vnew
        
    def getProbabilityDistribution(self,s,p_d):
        '''
        Computes the step ahead probability distribution
        '''
        shat = min(s,1)
        Pd = np.array([1-p_d,p_d])
        return np.hstack((self.Para.P[shat,0],self.Para.P[shat,1]*Pd))
        
    def maximizeObjective(self,s,Vprime):
        '''
        Maximize the objective function given the state
        '''
        p_d = self.p_d
        q = self.Para.q[s]
        gamma = self.Para.gamma
        beta = self.Para.beta
        if(len(self.Para.z) > 1):
            z = self.Para.z[s]
        else:
            z = self.Para.z
        delta = self.Para.delta
        #construct step ahead probabilitiy distriubtion
        P = self.getProbabilityDistribution(s,p_d)
        #Constant of proportionality relating c and g
        
        alpha = ( (1-gamma) *beta*P.dot(Vprime) / q )**(-1.0/gamma)
        g = (z+q*(1-delta))/(q+alpha)
        c = alpha*g
        V = c**(1-gamma)/(1-gamma)+beta*g**(1-gamma)*P.dot(Vprime)
        
        return V,c,g


def solveBellman(Para,p_d,s):
    T = BellmanMap(Para,p_d)
    
    sol =  root(lambda V:T(V)-V,-10*np.ones(3))
    
    if sol.success:
        return T.maximizeObjective(s,sol.x)
    else:
        V = -10*np.ones(3)
        for t in range(0,1000):
            Vnew = T(V)
            diff = np.linalg.norm(Vnew-V)
            V = Vnew
            if diff < 1e-10:
                break
        return T.maximizeObjective(s,V)

        
        
        

class BayesianBellmanMap:
    
    def __init__(self,Para):
        self.Para = Para
        
    def __call__(self,Vf):
        self.Vf = Vf
        return self.maximizeObjective
        
    def getProbabilityDistribution(self,state):
        '''
        Computes the step ahead probability distribution
        '''
        s,n,m = state
        #compute probability of disaster from beta binomial
        p_d = n*1.0/(n+m)
        shat = min(s,1)
        Pd = np.array([1-p_d,p_d])
        return np.hstack((self.Para.P[shat,0],self.Para.P[shat,1]*Pd))
        
    def maximizeObjective(self,state):
        '''
        Maximize the objective function given the state
        '''
        s,n,m = state
        q = self.Para.q[s]
        gamma = self.Para.gamma
        beta = self.Para.beta
        z = self.Para.z
        delta = self.Para.delta
        #construct step ahead probabilitiy distriubtion
        P = self.getProbabilityDistribution(state)
        #Constant of proportionality relating c and g
        Vprime = np.zeros(3)
        for sprime in range(0,3):
            nprime,mprime = n,m
            if s == 1:
                mprime = min(mprime+1,self.Para.mMax)
            if s == 2:
                nprime = min(nprime+1,self.Para.nMax)
            Vprime[sprime] = self.Vf[(sprime,nprime,mprime)]
        alpha = ( (1-gamma) *beta*P.dot(Vprime) / q )**(-1.0/gamma)
        g = (z+q*(1-delta))/(q+alpha)
        c = alpha*g
        V = c**(1-gamma)/(1-gamma)+beta*g**(1-gamma)*P.dot(Vprime)
        
        return np.hstack([V,c,g])


def solveBellmanEquation(V0,Para):
    '''
    Solve the Bellman Equation by iterating over the value function
    '''
    T = BayesianBellmanMap(Para)
    domain = Para.domain
    print 'creating Vold'
    sys.stdout.flush()
    Vold = np.zeros(len(domain))
    for i in range(0,len(domain)):
        Vold[i] = V0[domain[i]]
    print 'creating Vold done'
    sys.stdout.flush()
    niter = 100
    Vf = V0
    for i in range(0,niter):
        Vnew = T(Vf)
        policies = np.vstack(map(Vnew,domain))
        V = policies[:,0]
        print np.abs(V-Vold).max()
        Vold = V
        Vf = dict(itertools.izip(domain,V))
    
    cf = dict(itertools.izip(domain,policies[:,1]))
    gf = dict(itertools.izip(domain,policies[:,2]))
    
    return Vf,cf,gf
    
def approximateREValueFunction(Para):
    '''
    approximate the rational expectations value function
    '''
    p_ds = np.linspace(0,1)
    Vs = np.zeros((len(p_ds),3))
    for i,p_d in enumerate(p_ds):
        T = BellmanMap(Para,p_d)
        Vs[i,:] = root(lambda V:T(V)-V,-10*np.ones(3)).x
    Vf = [0]*3
    for s in range(0,3):
        Vf[s] = Spline(p_ds,Vs[:,s])
    return Vf
                
        