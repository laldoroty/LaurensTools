import numpy as np
from scipy.special import stdtr
from scipy.stats import ks_2samp

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
    Returns the weighted covariance of two quantities.

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

    cov_weights = 1/((yi-Y_EY)*exi**2 + (xi-X_EX)*eyi**2)

    numerator = np.sum(cov_weights*X_EX*Y_EY)
    denominator = np.sum(cov_weights)

    return numerator/denominator

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
    print('covXY, covXX, covYY')
    print(covXY)
    print(weighted_variance(xi,exi))
    print(weighted_variance(yi,eyi))

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