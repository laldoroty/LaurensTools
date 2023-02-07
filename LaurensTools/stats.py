import numpy as np
from scipy.special import stdtr
# from scipy.stats import ks_2samp

def weighted_expectation(xi,exi):
    """
    LNA 20220711
    Calculate the weighted expectation value of X.
    (AKA weighted mean.)
    xi -- array of data 
    exi -- array of data errors (square root of variance)

    """
    xi,exi = np.array(xi),np.array(exi)
    return np.sum(xi/exi**2)/np.sum(1/exi**2)

def weighted_variance(xi,exi):
    """
    LNA 20220720
    Returns the variance of a weighted mean.
    xi -- array of data
    exi -- array of data errors (square root of variance)

    """
    xi,exi = np.array(xi),np.array(exi)
    return (1/np.sum(1/exi**2))*np.sum((1/exi**2)*(xi-weighted_expectation(xi,exi))**2)

def weighted_covariance(xi,yi,exi,eyi):
    """
    LNA 20200720
    Returns the weighted covariance of two quantities,
    i.e., the off-diagonal element of the symmetric 2x2 covariance matrix,
    cov(X,Y).
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
    X_EX = xi-EX
    Y_EY = yi-EY
    # print('EX', EX)
    # print('EY', EY)

    # cov_weights = 1/((yi-EY)**2*exi**2 + (xi-EX)**2*eyi**2)  
    # cov_weights /= np.sum(cov_weights)
    # print('cov weights', cov_weights)  

    # numerator = np.sum(cov_weights*(xi-EX)*(yi-EY))
    # print('numerator',numerator)
    # denominator = np.sum(cov_weights)
    # print('denominator',denominator)

    # return numerator/denominator

    return weighted_expectation((X_EX*Y_EY),np.sqrt(Y_EY**2*exi**2+X_EX**2*eyi**2))

def wRMS(xi,exi):
    """
    LNA 20230207
    Returns the weighted RMS of a sample.
    xi -- array of data
    exi -- array of data errors (square root of variance)
    """
    num = np.sum(xi**2/exi**2)
    den = np.sum(1/exi**2)
    return np.sqrt(num/den)

def weighted_corr(xi,yi,exi,eyi):
    """
    Pearson Correlation coefficient, weighted by errors.

    xi -- data for random variable X
    yi -- data for random variable Y
    exi -- errors for xi
    eyi -- errors for yi

    """
    xi,yi,exi,eyi = np.array(xi),np.array(yi),np.array(exi),np.array(eyi)
    assert len(xi) == len(yi) and len(xi) == len(exi) and len(xi) == len(eyi),f'Arrays not of equal length. Array lengths are {len(xi)}, {len(yi)}, {len(exi)}, {len(eyi)}.'

    # Calculate weighted expectation values 
    # of each RV, X and Y: 
    EX = weighted_expectation(xi,exi)
    EY = weighted_expectation(yi,eyi)

    # Arguments to make cov(X,Y), i.e., 
    # cov(X,Y) = E((X-E(X))(Y-E(Y)))
    X_EX = xi-EX
    Y_EY = yi-EY

    # Variance for cov(X,X) and cov(Y,Y) are just the variances
    # directly from your data. 

    # Great, now we can calculate the covariance for X and Y:
    covXY = weighted_covariance(xi,yi,exi,eyi)

    # Now, we can calculate the weighted correlation coefficient.
    rho = covXY/np.sqrt(weighted_variance(xi,exi)*weighted_variance(yi,eyi))

    # We can also compute the associated p-value using a 
    # two-sided t-test.
    t = rho*np.sqrt(len(xi)-2)/np.sqrt(1-rho**2)
    p = stdtr(len(xi)-2,-np.abs(t))*2 # Args are (degrees of freedom, integration limits).

    return rho, p

# def weighted_ks_2samp(x,y,wx,wy):
#     """
#     Calculate the Kolmogorov-Smirnov statistic, modified to
#     incorporate the weight of your samples.

#     """
#     idx_x, idx_y = np.argsort(x), np.argsort(y)
#     x, y = x[idx_x], y[idx_y]
#     wx, wy = wx[idx_x], wy[idx_y]

#     xy = np.concatenate([x,y])

#     cdf_x = np.searchsorted(x, xy, side='right')/len(x)
#     cdf_y = np.searchsorted(y, xy, side='right')/len(y)