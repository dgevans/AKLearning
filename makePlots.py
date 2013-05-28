# -*- coding: utf-8 -*-
"""
Created on Mon May 20 09:43:18 2013

@author: dgevans
"""
from primitives import parameters
import bellman
import numpy as np

Para = parameters()
S = 3
p_d = np.linspace(0.0,0.20,100)
c = np.zeros((len(p_d),S))
g = np.zeros((len(p_d),S))
V = np.zeros((len(p_d),S))

for s in range(0,S):
    V[:,s],c[:,s],g[:,s] =zip(* map(lambda p: bellman.solveBellman(Para,p,s),p_d) )
