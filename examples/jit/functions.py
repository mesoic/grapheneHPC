#!/usr/bin/env python 
import math as m
from numba import jit

@jit('f8(f8)', nopython=True)
def erf(x):
    # save the sign of x
    sign = 1 if x >= 0 else -1
    x = abs(x)
    
    # constants
    a1 =  0.254829592
    a2 = -0.284496736
    a3 =  1.421413741
    a4 = -1.453152027
    a5 =  1.061405429
    p  =  0.3275911
    
    # A&S formula 7.1.26
    t = 1.0/(1.0 + p*x)
    y = 1.0 - (((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*m.exp(-x*x)
    return sign*y

@jit('f8(f8,f8,f8)',nopython = True)
def cdf(R, mu, sigma): 
    return 1 + erf( (R - mu) / (m.sqrt(2)*sigma))

@jit('f8(f8,f8[:])',nopython = True)
def cauchyDIS(ef,C):
    return (1./ ( m.pi*C[1] * ( 1  +  ((ef - C[0])/C[1])**2 )) )

@jit('f8(f8,f8[:])',nopython = True)
def gaussDIS(ef,G):
    return (1./m.sqrt(2)/G[1])*np.exp( -((ef - G[0])/(m.sqrt(2)*G[1]))**2)
