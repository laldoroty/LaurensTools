import numpy as np
from scipy.special import stdtr

def weighted_expectation(xi,exi):
    """
    LNA 20220711
    Calculate the weighted expectation value of X.
    xi -- array of data 
    exi -- array of data errors (square root of variance)
    
    """
    xi,exi = np.array(xi),np.array(exi)
    return np.sum(xi/exi**2)/np.sum(1/exi**2)

def weighted_corr(xi,yi,exi,eyi):
    """
    Pearson Correlation coefficient, weighted by errors.
    xi -- data for random variable X
    yi -- data for random variable Y
    exi -- errors for xi
    eyi -- errors for yi

    """
    xi,yi,exi,eyi = np.array(xi),np.array(yi),np.array(exi),np.array(eyi)

    # Calculate weighted expectation values 
    # of each RV, X and Y: 
    EX = weighted_expectation(xi,exi)
    EY = weighted_expectation(yi,eyi)

    # Arguments to make cov(X,Y), i.e., 
    # cov(X,Y) = E((X-E(X))(Y-E(Y)))
    X_EX = xi-EX
    Y_EY = yi-EY

    # We also need the variance of this quantity:
    err_covarg_XY = np.sqrt(Y_EY**2*exi**2 + X_EX**2*eyi**2)
    err_covarg_XX = np.sqrt(2*(Y_EY**2*exi**2))
    err_covarg_YY = np.sqrt(2*(X_EX**2*eyi**2))

    # Great, now we can calculate the covariances:
    covXY = weighted_expectation(X_EX*Y_EY,err_covarg_XY)
    covXX = weighted_expectation(X_EX**2,err_covarg_XX)
    covYY = weighted_expectation(Y_EY**2,err_covarg_YY)

    # Now, we can calculate the weighted correlation coefficient.
    rho = covXY/(covXX*covYY)

    # We can also compute the associated p-value using a 
    # two-sided t-test.
    t = rho*np.sqrt(len(xi)-2)/np.sqrt(1-rho**2)
    p = stdtr(len(xi)-2,-np.abs(t))*2 # Args are (degrees of freedom, integration limits).

    return rho, p


