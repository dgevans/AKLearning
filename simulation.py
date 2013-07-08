# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 12:06:42 2013

@author: dgevans
"""

import bayesian
import numpy as np
import bellman

def simulateRE(s0,T,Para,p_d):
    '''
    Simulate a history using RE solution
    '''
    S = 3
    V = np.zeros(S)
    c = np.zeros(S)
    g = np.zeros(S)
    for s in range(0,S):
        V[s],c[s],g[s] = bellman.solveBellman(Para,p_d,s)
        
    #now compute bond prices
    p = np.zeros(S)
    TMap = bellman.BellmanMap(Para,p_d)
    P = np.zeros((S,S))
    for s in range(0,S):
        P[s,:] = TMap.getProbabilityDistribution(s,p_d)
    cumP = np.cumsum(P,axis=1)    
    for s in range(0,S):
        p[s] = Para.beta*(g[s]/c[s])**(-Para.gamma)*P[s,:].dot(c**(-Para.gamma))
        
    cHist = np.zeros(T)
    gHist = np.zeros(T)
    qHist = np.zeros(T)
    pHist = np.zeros(T)
    s = s0
    for t in range(0,T):
        cHist[t] = c[s]
        gHist[t] = g[s]
        qHist[t] = Para.q[s]
        pHist[t] = p[s]
        r = np.random.rand()
        for sprime in range(0,3):
            if r < cumP[s,sprime]:
                break;
                
        s = sprime
    return cHist,gHist,qHist,pHist


def simulate(state0,T,Vf,Para,p_d):
    '''
    Simulate a history for T periods
    '''
    state = state0
    #construct full transition matrix
    P = np.zeros((3,3))
    for s in range(0,3):
        shat = min(s,1)
        Pd = np.array([1-p_d,p_d])
        P[s,:] = np.hstack((Para.P[shat,0],Para.P[shat,1]*Pd))
    #cumulative probabiliy matrix to draw random shocks
    cumP = np.cumsum(P,axis=1)
    TMap = bayesian.BayesianBellmanMap(Para)
    Gamma = bayesian.BayesMap(Para)
    PolicyFunction = TMap(Vf)
    #initialize
    c = np.zeros(T)
    g = np.zeros(T)
    q = np.zeros(T)
    p = np.zeros(T)
    for t in  range(0,T):
        s,mu = state
        _,c[t],g[t] = PolicyFunction(state)
        q[t] = Para.q[s]
        p[t] = computeBondPrice(state,Vf,Para)
        #now draw next state
        r = np.random.rand()
        for sprime in range(0,3):
            if r < cumP[s,sprime]:
                break;
        muprime = bayesian.approximatePosterior(Gamma(s,sprime,mu))
        #iterate on stae
        state = sprime,muprime
        
    return c,g,q,p
    
    
def computeBondPrice(state,Vf,Para):
    '''
    Computes the price of a bond given beliefs of the agent
    '''
    s,mu = state
    T = bayesian.BayesianBellmanMap(Para)
    Gamma = bayesian.BayesMap(Para)
    Vnew = T(Vf)
    
    _,c,g = Vnew(state)
    S = 3
    cprime = np.zeros(S)
    for sprime in range(0,S):
        muprime = bayesian.approximatePosterior(Gamma(s,sprime,mu))
        _,cprime[sprime],_ = Vnew((sprime,muprime))
        
    P = T.getProbabilityDistribution(state)
    
    return Para.beta*(g/c)**(-Para.gamma)*P.dot(cprime**(-Para.gamma))