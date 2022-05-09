import os, sys
import numpy as np
import os.path as pa
from astropy.cosmology import FlatLambdaCDM
import pycmpfit

def saltmodel(bmax,ebmax,x1,ex1,c,ec,zcmb):
    """
    Fits the SALT2 model for distance modulus for SNe Ia.
    mu = Bmax - M + a*(x1) - b*(c)
    where M, a, and b are fit parameters.
    x1 and c are derived from the light curve fits.
    (Guy et al. 2007).

    Keyword arguments:
    bmax -- Apparent magnitude at maximum B band brightness
    ebmax -- Error of bmax
    x1 -- Stretch from fit to light curve.
    ex1 -- x1 error.
    c -- Color parameter from fit to light curve.
    ec -- Error of c
    zcmb -- CMB redshift
    """

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    mu = cosmo.distmod(zcmb).value
    speedoflight = 3E5 # speed of light, km/s    
    vpec = 300 # assumed peculiar velocity, km/s
    evpec = (5/np.log(10))*(vpec/(speedoflight*zcmb)) # error from peculiar velocity, units are magnitudes 

    # m: Number of samples [len(X_obs)]
    # n: Number of parameters of the function [len(theta)]
    
    def userfunc(m, n, theta, private_data):
        M,a,b = theta
        BMAX, EBMAX, X1, EX1, C, EC, EVPEC, MU = private_data["bmax"], private_data["ebmax"],private_data["x1"],private_data["ex1"],private_data["c"], private_data["ec"], private_data["evpec"], private_data["mu"]
        devs = np.zeros((m), dtype = np.float64)
        user_dict = {"deviates": None} 
        
        for i in range(m):
            devs[i] = (MU[i] - (BMAX[i] - M + a*(X1[i]) - b*C[i]))/(EVPEC[i]**2 + EBMAX[i]**2 + a**2*EX1[i]+ b**2*EC[i]**2)**0.5
        user_dict["deviates"] = devs
        return user_dict

    theta = np.array([-19,0.1,0.1])   
    m, n = len(bmax), len(theta)
    user_data = {"bmax": bmax, "ebmax": ebmax, "x1": x1, "ex1": ex1, "c": c, "ec": ec, "evpec": evpec, "mu": mu}    
    py_mp_par = list(pycmpfit.MpPar() for i in range(n))

    fit = pycmpfit.Mpfit(userfunc, m, theta, private_data=user_data, py_mp_par=py_mp_par)
    fit.mpfit()  # NOTE now theta has been updated
    mp_result = fit.result
    dof = mp_result.nfunc - mp_result.nfree
    chisq = mp_result.bestnorm

    theta_fin = theta
    covariance_matrix = mp_result.covar
    print('Fitted Parameters:', theta_fin)
    print('Covariance matrix: \n', covariance_matrix)
    
    jac = np.array([-1, x1, -c])
    err = np.sqrt(np.matmul(jac, np.matmul(covariance_matrix, jac.T)) + evpec**2)
    
    # theta_fin is the final fit parameters, M, a, d
    # err is the error on MU for EACH SN
    return theta_fin, err, dof, chisq
