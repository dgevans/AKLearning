# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:45:00 2013

@author: dgevans
"""
import sys
sys.path.append('/opt/local/Library/Frameworks/Python.framework/Versions/2.7/lib/python2.7/site-packages')
from scipy.integrate import quad
from scipy.stats import beta
import numpy as np
from primitives import parameters
from scipy.optimize import minimize_scalar
from primitives import posterioDistriubtion
from mpi4py import MPI

if __name__ == "__main__":
    testSimulate()

class BayesMap:
    
    def __init__(self,Para):
        self.Para = Para
    
    def __call__(self,s,sprime,mu):
        self.mu = mu
        self.s = s
        self.sprime = sprime
        norm = quad(self.calculatePosterior,0.0,1.0)[0]
        return lambda p_d: self.calculatePosterior(p_d,normalization=norm)
    
    
    def calculatePosterior(self,p_d,normalization=1.0):
        '''
        Calcluate the posterior
        '''
        shat = min(self.s,1)
        sprimehat = min(self.sprime,1)
        transition_prob = self.Para.P[shat,sprimehat]
        if self.sprime == 2:
            transition_prob *= p_d
        if self.sprime == 1:
            transition_prob *= (1-p_d)
        #now do Bayes rule
        return max(transition_prob*self.prior(p_d)/normalization,0.0)
        
    def prior(self,p_d):
        return self.mu(p_d)

def approximatePosterior(mu,x = np.linspace(0,1,10),tol=1e-3):
    '''
    Approximates the posterior using cubic splines
    '''
    y = np.hstack(map(mu,x))
    muhat = posterioDistriubtion(x,y,[2])
    done = False
    while not done:
        xnew = []
        ynew = []
        done = True
        for i in xrange(0,len(x)-1):
            xhat = (x[i]+x[i+1])/2.0
            mu_xhat = mu(xhat)
            diff = abs(mu_xhat-muhat(xhat))
            reldiff = 1.0
            if mu_xhat != 0.0:
                reldiff = abs(diff/mu_xhat)
            if min(diff,reldiff) > tol:
                xnew.append(xhat)
                ynew.append(mu_xhat)
                done = False
        if not done:
            x = np.hstack((x,np.hstack(xnew)))
            y = np.hstack((y,np.hstack(ynew)))
            muhat.fit(x,y,[2])
            x = np.sort(x)
            y = muhat(x)
    return muhat
    

class BayesianBellmanMap:
    
    def __init__(self,Para):
        self.Para = Para
        self.Gamma = BayesMap(Para)
        
    def __call__(self,Vf):
        self.Vf = Vf
        return self.maximizeObjective
        
    def getProbabilityDistribution(self,state):
        '''
        Computes the step ahead probability distribution
        '''
        s,mu = state
        #compute probability of disaster from beta binomial
        Ep_d = mu.getMoment(1)
        shat = min(s,1)
        Pd = np.array([1-Ep_d,Ep_d])
        return np.hstack((self.Para.P[shat,0],self.Para.P[shat,1]*Pd))
        
    def maximizeObjective(self,state):
        '''
        Maximize the objective function given the state
        '''
        s,mu = state
        if not hasattr(mu,'muprime'):
            mu.muprime = {}
            for sprime in range(0,3):
                mu.muprime[sprime] = approximatePosterior(self.Gamma(s,sprime,mu))
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
            Vprime[sprime] = self.Vf((sprime,mu.muprime[sprime]))
            
        alpha = ( (1-gamma) *beta*P.dot(Vprime) / q )**(-1.0/gamma)
        g = (z+q*(1-delta))/(q+alpha)
        c = alpha*g
        V = c**(1-gamma)/(1-gamma)+beta*g**(1-gamma)*P.dot(Vprime)
        
        return np.hstack([V,c,g])

def drawSamplePaths(s0,mu0,Para,N=20,T=1000):
    '''
    Draw a sequence of realizations of the state and calculate the posterior at
    each history
    '''
    w = MPI.COMM_WORLD
    rank = w.Get_rank()
    size = w.Get_size()
    n = N/size
    r = N%size
    stateHist = {}
    Gamma = BayesMap(Para)
    for i in range(rank*n+min(rank,r), min((rank+1)*n+min(rank+1,r),N)):

        print range(rank*n+min(rank,r), min((rank+1)*n+min(rank+1,r),N))
        print rank,':',rank*n+min(rank,r),min((rank+1)*n+min(rank+1,r),N)
        #setup
        #draw randomly from initial prior
        p_d = drawFromMu(mu0)
        #construct full transition matrix
        P = np.zeros((3,3))
        for s in range(0,3):
            shat = min(s,1)
            Pd = np.array([1-p_d,p_d])
            P[s,:] = np.hstack((Para.P[shat,0],Para.P[shat,1]*Pd))
        #cumulative probabiliy matrix to draw random shocks
        cumP = np.cumsum(P,axis=1)
        
        stateHist[(i,0)] = (s0,mu0)
        for t in range(1,T):
            #print t
            s,mu = stateHist[(i,t-1)]
            r = np.random.rand()
            for sprime in range(0,3):
                if r < cumP[s,sprime]:
                    break;
            muprime = Gamma(s,sprime,mu)
            stateHist[(i,t)] = sprime,approximatePosterior(muprime)
    stateHists = w.allgather(stateHist)
    stateHist = {}
    for hist in stateHists:
        stateHist.update(hist)
    return stateHist


def drawFromMu(mu):
    '''
    Draw randomly from mu using rejection sampling.  Under assumption that mu has
    single maximizer
    '''
    #First choose $M$ such that mu(p_d) < M for all p_d
    res = minimize_scalar(lambda p_d:-mu(p_d),bounds=(0.0,1.0),method='bounded')
    if not res.success:
        raise Exception('Could not find maximal value of mu')
    M = mu(res.x)+1.0
    while True:
        p_d = np.random.rand()
        u = np.random.rand()
        if u < mu(p_d)/M:
            return p_d
            
def testSimulate(mu0=beta(3,3).pdf,T=100):
    Para = parameters()
    
    mu = lambda p_d: 2.0*p_d#beta(5,5).pdf
    p_d = 0.1
    P = np.zeros((3,3))
    Gamma = BayesMap(Para)
    for s in range(0,3):
        shat = min(s,1)
        Pd = np.array([1-p_d,p_d])
        P[s,:] = np.hstack((Para.P[shat,0],Para.P[shat,1]*Pd))
    
    cumP = np.cumsum(P,axis=1)
    sHist = np.zeros(T,dtype=np.int)
    muHist = [mu0]
    for t in range(1,T):
        print t
        mu = muHist[t-1]
        r = np.random.rand()
        s = sHist[t-1]
        for sprime in range(0,3):
            if r < cumP[s,sprime]:
                break;
        sHist[t] = sprime
        muprime = Gamma(s,sprime,mu)
        muHist.append( approximatePosterior(muprime) )
    return muHist,sHist