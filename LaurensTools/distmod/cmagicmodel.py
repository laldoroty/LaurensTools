import os, sys
import numpy as np
import os.path as pa
from astropy.cosmology import FlatLambdaCDM
import pycmpfit

def cmagicmodel(bbv,ebbv,bmax,ebmax,dm15,edm15,slope,eslope,zcmb):
    cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
    mu = cosmo.distmod(zcmb).value
    print(mu)
    vpec = 300 # assumed peculiar velocity, km/s
    speedoflight = 3E5 # speed of light, km/s
    evpec = (5/np.log(10))*(vpec/(speedoflight*zcmb)) # error from peculiar velocity
    slope_avg=np.mean(slope)
    dm15_avg=np.mean(dm15)
    one_ovr_slope_avg=np.mean(1/slope)

    # m: Number of samples [len(X_obs)]
    # n: Number of parameters of the function [len(theta)]
    
    def userfunc(m, n, theta, private_data):
        M,delta,b2 = theta
        BBV,EBBV,BMAX,EBMAX,DM15,EDM15,SLOPE,ESLOPE,EVPEC,MU = private_data["bbv"],private_data["ebbv"],private_data["bmax"],private_data["ebmax"], private_data["dm15"],private_data["edm15"],private_data["slope"],private_data["eslope"],private_data["evpec"], private_data["mu"]
        devs = np.zeros((m), dtype = np.float64)
        user_dict = {"deviates": None} 
        
        for i in range(m):
            num = MU[i] - (BBV[i] - M - delta*(DM15[i] - dm15_avg) - (b2 - SLOPE[i])*((BMAX[i]-BBV[i])/SLOPE[i] + 0.6 + 1.2*(1/SLOPE[i] - one_ovr_slope_avg)))
            denom = (EVPEC[i]**2 + ((b2/SLOPE[i])*(EBBV[i]))**2 + (delta*EDM15[i])**2 + ((-b2*BMAX[i]/SLOPE[i]**2 + b2*BBV[i]/SLOPE[i]**2 - 1.2*b2/SLOPE[i]**2 - 0.6 - 1.2*one_ovr_slope_avg)*ESLOPE[i])**2 + ((b2/SLOPE[i] - 1)*EBMAX[i])**2)**0.5   
            devs[i] = num/denom
        user_dict["deviates"] = devs
        return user_dict

    theta = np.array([-19,0.2,1.5])   
    m, n = len(bbv), len(theta)
    user_data = {"bbv": bbv, "ebbv": ebbv, "bmax": bmax, "ebmax": ebmax, "dm15": dm15, "edm15": edm15, "slope": slope, "eslope": eslope, "evpec": evpec, "mu": mu}    
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
    
    jac = np.array([-1, -(dm15 - dm15_avg), -((bmax - bbv)/slope + 0.6 + 1.2*(1/slope - np.mean(1/slope)))])
    err = np.sqrt(np.matmul(jac, np.matmul(covariance_matrix,jac.T)) + evpec**2)
    
    return theta_fin, err, dof, chisq