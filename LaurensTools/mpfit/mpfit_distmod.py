import os, sys
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname,'mpfit'))

from mpfit import mpfit

def FTripp(mB,c,dm15,p):
    """
    Set up linear function for mpfit to use. 
    DO NOT call this.
    """
    mu =  mB - p[0] - p[1]*(c - np.mean(c)) - p[2]*(dm15 - np.mean(dm15))
    return mu

def myfunctTripp(p, fjac=None, mu=None, mB=None, c=None, dm15=None, emB=None, ec=None, edm15=None, evpec=None):
    """
    Set up chisq to minimize for mpfit. 
    DO NOT call this. 
    """
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    model = FTripp(mB,c,dm15,p)
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    
    return [status, np.sqrt(mu - model)**2/(evpec**2 + emB**2 + (p[1]*ec)**2 + (p[2]*edm15)**2)]

def trippfit(mu,mB,emB,c,ec,dm15,edm15,evpec,initial_guess=[1,1,1], return_chisq=True):
    """
    Do the linear fit. USE THIS FUNCTION DIRECTLY IN YOUR CODE. 
    If return_chisq = True, also returns m.fnorm (chisq) and m.dof (degrees
    of freedom.)
    """
    p0=np.array(initial_guess,dtype='float64')  #initial conditions
    fa = {'mu':mu,'mB':mB,'emB':emB,'c':c,'ec':ec,'dm15':dm15,'edm15':edm15,'evpec':evpec}
    m = mpfit(myfunctTripp, p0, functkw=fa)

    if return_chisq:
        return m.params, m.covar, m.fnorm, m.dof
    else:
        return m.params, m.covar