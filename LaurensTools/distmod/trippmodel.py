import os, sys
import numpy as np
import os.path as pa
from astropy.cosmology import FlatLambdaCDM
import pycmpfit

def trippmodel(bmax,ebmax,c,ec,dm15,edm15,zcmb):
    """
    Fits the Tripp model for distance modulus for SNe Ia.
    mu = Bmax - M - a*(C-C_avg) - d*(dm15-dm15_avg)
    where M, a, and d are fit parameters.

    Keyword arguments:
    bmax -- Apparent magnitude at maximum B band brightness
    ebmax -- Error of bmax
    c -- Observed color at time of bmax
    ec -- Error of c
    dm15 -- Change in magnitude from bmax to 15 days later
    edm15 -- Error of dm15
    zcmb -- CMB redshift
    """

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    mu = cosmo.distmod(zcmb).value
    speedoflight = 3E5 # speed of light, km/s    
    vpec = 300 # assumed peculiar velocity, km/s
    evpec = (5/np.log(10))*(vpec/(speedoflight*zcmb)) # error from peculiar velocity, units are magnitudes 
    c_avg=np.mean(c)
    dm15_avg=np.mean(dm15)

    # m: Number of samples [len(X_obs)]
    # n: Number of parameters of the function [len(theta)]
    
    def userfunc(m, n, theta, private_data):
        M,a,delta = theta
        BMAX, EBMAX, C, EC, DM15, EDM15, EVPEC, MU = private_data["bmax"], private_data["ebmax"], private_data["c"], private_data["ec"], private_data["dm15"], private_data["edm15"], private_data["evpec"], private_data["mu"]
        devs = np.zeros((m), dtype = np.float64)
        user_dict = {"deviates": None} 
        
        for i in range(m):
#             devs[i] = (MU[i] - (BMAX[i] - M - a*(C[i]-c_avg) - delta*(DM15[i]-dm15_avg)))
            devs[i] = (MU[i] - (BMAX[i] - M - a*(C[i]-c_avg) - delta*(DM15[i]-dm15_avg)))/(EVPEC[i]**2 + EBMAX[i]**2 + a**2*EC[i]**2 + delta**2*EDM15[i]**2)**0.5
        user_dict["deviates"] = devs
        return user_dict

    theta = np.array([-19,0.1,0.1])   
    m, n = len(bmax), len(theta)
    user_data = {"bmax": bmax, "ebmax": ebmax, "c": c, "ec": ec, "dm15": dm15, "edm15": edm15, "evpec": evpec, "mu": mu}    
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
    
    jac = np.array([-1, -c+c_avg, -dm15+dm15_avg])
    err = np.sqrt(np.matmul(jac, np.matmul(covariance_matrix, jac.T)) + evpec**2)
    
    return theta_fin, err, dof, chisq
