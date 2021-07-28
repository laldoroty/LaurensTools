import os, sys
import numpy as np

dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dirname,'mpfit'))

from mpfit import mpfit

def Flin(x,p):
    """
    Set up linear function for mpfit to use. 
    DO NOT call this.
    """
    y =  p[0] + p[1]*x 
    return y

def myfunctlin(p, fjac=None, x=None, y=None, xerr=None, yerr=None):
    """
    Set up chisq to minimize for mpfit. This will accept any combination of xerr and yerr (i.e., you do not need either, or both).
    DO NOT call this. 
    """
    # Parameter values are passed in "p"
    # If fjac==None then partial derivatives should not be
    # computed.  It will always be None if MPFIT is called with default
    # flag.
    model = Flin(x, p)
    # Non-negative status value means MPFIT should continue, negative means
    # stop the calculation.
    status = 0
    
    if xerr is not None and yerr is not None:
        return [status, np.sqrt((y-model)**2/(yerr**2 + (p[1]**2)*xerr**2))]
    elif xerr is None and yerr is not None:
        return [status, np.sqrt((y-model)**2/(yerr**2))]
    elif xerr is not None and yerr is not None:
        return [status, np.sqrt((y-model)**2/((p[1]**2)*xerr**2))]
    elif xerr is None and yerr is None:
        return [status, y-model]

def linfit(x,y,ex=None, ey=None, initial_guess=[0,1]):
    """
    Do the linear fit. USE THIS FUNCTION DIRECTLY IN YOUR CODE. 
    """
    p0=np.array(initial_guess,dtype='float64')  #initial conditions
    if ex is not None and ey is not None:
        fa = {'x':x, 'y':y, 'xerr': ex, 'yerr': ey}
    elif ex is None and ey is not None:
        fa = {'x':x, 'y':y, 'yerr': ey}
    elif ex is not None and ey is None:
        fa = {'x':x, 'y':y, 'xerr': ex}
    elif ex is None and ey is None:
        fa = {'x':x, 'y':y}
    m = mpfit(myfunctlin, p0, functkw=fa)
#         chisq=(myfunctlin(m.params, x=x, y=y, yerr=ey)[1]**2).sum()
    return m.params, m.covar