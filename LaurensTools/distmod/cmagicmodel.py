import os, sys
import numpy as np
import os.path as pa
from astropy.cosmology import FlatLambdaCDM
import pycmpfit

def cmagicmodel(bbv,ebbv,bmax,ebmax,dm15,edm15,slope,eslope,zcmb,model='He2018'):
    """
    Arguments for 'model': 
    'He2018': M3 from He et al. 2018.
    'Aldoroty2022': mine
    'nodm15': mine but without the dm15 term. If using this, for dm15 and edm15 arguments,
    put 'None'.
    """

    cosmo = FlatLambdaCDM(H0=70,Om0=0.3)
    mu = cosmo.distmod(zcmb).value
    print(mu)
    vpec = 300 # assumed peculiar velocity, km/s
    speedoflight = 3E5 # speed of light, km/s
    evpec = (5/np.log(10))*(vpec/(speedoflight*zcmb)) # error from peculiar velocity
    slope_avg=np.mean(slope)
    dm15_avg=np.mean(dm15)
    one_ovr_slope_avg=np.mean(1/slope)
    color_avg = np.mean((bmax-bbv)/slope)

    # m: Number of samples [len(X_obs)]
    # n: Number of parameters of the function [len(theta)]
    
    def userfunc(m, n, theta, private_data):
        if model=='He2018' or model=='Aldoroty2022':
            M,delta,b2 = theta
            BBV,EBBV,BMAX,EBMAX,DM15,EDM15,SLOPE,ESLOPE,EVPEC,MU = private_data["bbv"],private_data["ebbv"],private_data["bmax"],private_data["ebmax"], private_data["dm15"],private_data["edm15"],private_data["slope"],private_data["eslope"],private_data["evpec"], private_data["mu"]
        elif model=='nodm15':
            M,b2 = theta
            BBV,EBBV,BMAX,EBMAX,SLOPE,ESLOPE,EVPEC,MU = private_data["bbv"],private_data["ebbv"],private_data["bmax"],private_data["ebmax"],private_data["slope"],private_data["eslope"],private_data["evpec"], private_data["mu"]
        
        devs = np.zeros((m), dtype = np.float64)
        user_dict = {"deviates": None} 
        
        for i in range(m):
            if model=='He2018':
                num = MU[i] - (BBV[i] - M - delta*(DM15[i] - dm15_avg) - (b2 - SLOPE[i])*((BMAX[i]-BBV[i])/SLOPE[i] + 1.2*(1/SLOPE[i] - one_ovr_slope_avg)))
                # denom = (EVPEC[i]**2 + ((b2/SLOPE[i])*(EBBV[i]))**2 + (delta*EDM15[i])**2 + ((-b2*BMAX[i]/SLOPE[i]**2 + b2*BBV[i]/SLOPE[i]**2 - 1.2*b2/SLOPE[i]**2 - 1.2*one_ovr_slope_avg)*ESLOPE[i])**2 + ((b2/SLOPE[i] - 1)*EBMAX[i])**2)**0.5   
                dbbv = b2/SLOPE[i]
                ddm15 = -delta
                dbmax = -(b2 - SLOPE[i])/SLOPE[i]
                dslope = ((BMAX[i] - BBV[i])/SLOPE[i] + 1.2*((1/SLOPE[i]) - one_ovr_slope_avg)) + (SLOPE[i] - b2)*(-(BMAX[i] - BBV[i])/SLOPE[i]**2 - 1.2/SLOPE[i]**2)
                denom = np.sqrt(EBBV[i]**2*dbbv**2 + EDM15[i]**2*ddm15**2 + EBMAX[i]**2*dbmax**2 + ESLOPE[i]**2*dslope**2 + EVPEC[i]**2)
            elif model=='Aldoroty2022':
                num = MU[i] - (BBV[i] - M - delta*(DM15[i] - dm15_avg) - (b2 - SLOPE[i])*((BMAX[i]-BBV[i])/SLOPE[i] - color_avg))
                dbbv = b2/SLOPE[i]
                ddm15 = -delta
                dbmax = -(b2 - SLOPE[i])/SLOPE[i]
                dslope = b2*(BMAX[i] - BBV[i])/SLOPE[i]**2 - color_avg
                denom = np.sqrt(EBBV[i]**2*dbbv**2 + EDM15[i]**2*ddm15**2 + EBMAX[i]**2*dbmax**2 + ESLOPE[i]**2*dslope**2 + EVPEC[i]**2)
            elif model=='nodm15':
                num = MU[i] - (BBV[i] - M - (b2 - SLOPE[i])*((BMAX[i]-BBV[i])/SLOPE[i] - color_avg))
                dbbv = b2/SLOPE[i]
                dbmax = -(b2 - SLOPE[i])/SLOPE[i]
                dslope = b2*(BMAX[i] - BBV[i])/SLOPE[i]**2 + color_avg
                denom = np.sqrt(EBBV[i]**2*dbbv**2 + EBMAX[i]**2*dbmax**2 + ESLOPE[i]**2*dslope**2 + EVPEC[i]**2)

            devs[i] = num/denom
        user_dict["deviates"] = devs
        return user_dict

    if model == 'He2018' or model == 'Aldoroty2022':
        theta = np.array([-19,0.15,1.8])   
        user_data = {"bbv": bbv, "ebbv": ebbv, "bmax": bmax, "ebmax": ebmax, "dm15": dm15, "edm15": edm15, "slope": slope, "eslope": eslope, "evpec": evpec, "mu": mu} 
    elif model == 'nodm15':
        theta = np.array([-19,1])
        user_data = user_data = {"bbv": bbv, "ebbv": ebbv, "bmax": bmax, "ebmax": ebmax, "slope": slope, "eslope": eslope, "evpec": evpec, "mu": mu} 
    m, n = len(bbv), len(theta)
       
    py_mp_par = list(pycmpfit.MpPar() for i in range(n))
    
    print('before fit')
    fit = pycmpfit.Mpfit(userfunc, m, theta, private_data=user_data, py_mp_par=py_mp_par)
    print('after fit')
    fit.mpfit()  # NOTE now theta has been updated
    print('update mpfit')
    mp_result = fit.result
    dof = mp_result.nfunc - mp_result.nfree
    chisq = mp_result.bestnorm

    theta_fin = theta
    covariance_matrix = mp_result.covar
    print('Fitted Parameters:', theta_fin)
    print('Covariance matrix: \n', covariance_matrix)
    
    if model=='He2018':
        jac = np.array([-1, -(dm15 - dm15_avg), -((bmax - bbv)/slope + 1.2*(1/slope - np.mean(1/slope)))])
    elif model=='Aldoroty2022':
        jac = np.array([-1, -(dm15 - dm15_avg), -((bmax - bbv)/slope - color_avg)])
    elif model=='nodm15':
        jac = np.array([-1, -((bmax - bbv)/slope - color_avg)])
    err = np.sqrt(np.matmul(jac, np.matmul(covariance_matrix,jac.T)) + evpec**2)
    
    return theta_fin, err, dof, chisq