import os, sys
import numpy as np
import os.path as pa
from copy import copy
from astropy.cosmology import FlatLambdaCDM
from astropy.table import Table, Column
from .sncosmo_utils import helio_to_cmb
import pycmpfit

class HubbleDiagram(object):
    """
    Makes a Hubble diagram. 

    Input table should be an astropy table. 

    ### TO DO:
    # Input entire LCs OR singular Bmax or BBV values
    # Input either zhelio or zcmb
    # Plot
    """
    def __init__(self,data,model='tripp',cosmo=FlatLambdaCDM(H0=70, Om0=0.3)):
        acceptable_models = ['tripp','salt','He2018','Aldoroty2022']
        self.data = data
        self.model = model
        self.cosmo = cosmo

        # We use these variables later:
        self.covar = None

        if model not in acceptable_models:
            raise ValueError(f'model argument not recognized. Acceptable models are {acceptable_models}.')
        elif model == 'tripp':
            self.input_data = self.data['bmax','ebmax','c','ec','dm15','edm15','zcmb']
        elif model == 'salt':
            self.input_data = self.data['bmax','ebmax','x1','ex1','c','ec','zcmb']
        elif model == 'He2018' or model == 'Aldoroty2022':
            self.input_data = self.data['bbv','ebbv','bmax','ebmax','dm15','edm15','slope','eslope','zcmb']

    def test_dm(self,train_pars,test,train):
        """
        First, set up the distance modulus equations for
        your test data point in each iteration. 
        """
        if self.model=='tripp':
            M,a,d = train_pars
            mu = test['bmax'] - M - a*(test['c']-np.mean(train['c'])) - d*(test['dm15']-np.mean(train['dm15']))
        elif self.model=='salt':
            M,a,b = train_pars
            mu = test['bmax'] - M + a*(test['x1']) - b*(test['x1'])
        elif self.model=='He2018' or self.model=='Aldoroty2022':
            M,delta,b2 = train_pars
            if self.model=='He2018':
                mu = test['bbv'] - M - delta*(test['dm15'] - np.mean(train['dm15'])) - \
                (b2 - test['slope'])*((test['bmax']-test['bbv'])/test['slope'] + \
                    1.2*(1/test['slope'] - np.mean(1/train['slope'])))
            elif self.model=='Aldoroty2022':
                mu = test['bbv'] - M - delta*(test['dm15'] - np.mean(train['dm15'])) - \
                    (b2 - test['slope'])*((test['bmax']-test['bbv'])/test['slope'] - \
                        np.mean((train['bmax']-train['bbv'])/train['slope']))

        mu_expected = self.cosmo.distmod(test['zcmb']).value
        resid = mu - mu_expected
        return mu, resid, test['zcmb']

    def minimize(self,verbose=False):
        """
        Calculate the distance modulus using chi-square
        minimization. 
        """
        expected_mu = self.cosmo.distmod(self.data['zcmb']).value
        speedoflight = 3E5 # speed of light, km/s  
        vpec = 300 # assumed peculiar velocity, km/s
        evpec = (5/np.log(10))*(vpec/(speedoflight*self.data['zcmb'])) # error from peculiar velocity, units are magnitudes 

        self.input_data['mu'] = expected_mu
        self.input_data['evpec'] = evpec

        def userfunc(m, n, theta, private_data=self.input_data):
            if self.model=='tripp':
                M,a,delta = theta
                BMAX, EBMAX, C, EC, DM15, EDM15, EVPEC, MU = private_data["bmax"], private_data["ebmax"], private_data["c"], private_data["ec"], private_data["dm15"], private_data["edm15"], private_data["evpec"], private_data["mu"]
            elif self.model=='salt':
                M,a,b = theta
                BMAX, EBMAX, X1, EX1, C, EC, EVPEC, MU = private_data["bmax"], private_data["ebmax"],private_data["x1"],private_data["ex1"],private_data["c"], private_data["ec"], private_data["evpec"], private_data["mu"]
            elif self.model=='He2018' or self.model=='Aldoroty2022':
                M,delta,b2 = theta
                BBV,EBBV,BMAX,EBMAX,DM15,EDM15,SLOPE,ESLOPE,EVPEC,MU = private_data["bbv"],private_data["ebbv"],private_data["bmax"],private_data["ebmax"], private_data["dm15"],private_data["edm15"],private_data["slope"],private_data["eslope"],private_data["evpec"], private_data["mu"]

            devs = np.zeros((m), dtype = np.float64)
            user_dict = {"deviates": None} 

            for i in range(m):
                if self.model=='tripp':
                    devs[i] = (MU[i] - (BMAX[i] - M - a*(C[i]-np.mean(C)) - delta*(DM15[i]-np.mean(DM15))))/(EVPEC[i]**2 + EBMAX[i]**2 + a**2*EC[i]**2 + delta**2*EDM15[i]**2)**0.5
                elif self.model=='salt':
                    devs[i] = (MU[i] - (BMAX[i] - M + a*(X1[i]) - b*C[i]))/(EVPEC[i]**2 + EBMAX[i]**2 + a**2*EX1[i]+ b**2*EC[i]**2)**0.5
                elif self.model=='He2018':
                    num = MU[i] - (BBV[i] - M - delta*(DM15[i] - np.mean(DM15)) - (b2 - np.mean(SLOPE))*((BMAX[i]-BBV[i])/SLOPE[i] + 1.2*(1/SLOPE[i] - np.mean(1/SLOPE))))
                    dbbv = b2/SLOPE[i]
                    ddm15 = -delta
                    dbmax = -(b2 - SLOPE[i])/SLOPE[i]
                    dslope = ((BMAX[i] - BBV[i])/SLOPE[i] + 1.2*((1/SLOPE[i]) - np.mean(1/SLOPE))) + (SLOPE[i] - b2)*(-(BMAX[i] - BBV[i])/SLOPE[i]**2 - 1.2/SLOPE[i]**2)
                    denom = np.sqrt(EBBV[i]**2*dbbv**2 + EDM15[i]**2*ddm15**2 + EBMAX[i]**2*dbmax**2 + ESLOPE[i]**2*dslope**2 + EVPEC[i]**2)
                    devs[i] = num/denom
                elif self.model=='Aldoroty2022':
                    num = MU[i] - (BBV[i] - M - delta*(DM15[i] - np.mean(DM15)) - (b2 - SLOPE[i])*((BMAX[i]-BBV[i])/SLOPE[i] - np.mean((BMAX-BBV)/SLOPE)))
                    dbbv = b2/SLOPE[i]
                    ddm15 = -delta
                    dbmax = -(b2 - SLOPE[i])/SLOPE[i]
                    dslope = b2*(BMAX[i] - BBV[i])/SLOPE[i]**2 - np.mean((BMAX-BBV)/SLOPE)
                    denom = np.sqrt(EBBV[i]**2*dbbv**2 + EDM15[i]**2*ddm15**2 + EBMAX[i]**2*dbmax**2 + ESLOPE[i]**2*dslope**2 + EVPEC[i]**2)
                    devs[i] = num/denom

            user_dict["deviates"] = devs
            return user_dict

        if self.model=='tripp' or self.model=='salt':
            theta = np.array([-19,0.1,0.1])   
        elif self.model=='He2018' or self.model=='Aldoroty2022':
            theta = np.array([-19,0.15,1.8])   

        user_data_cols = [col for col in self.input_data.colnames]
        user_data = {col: self.input_data[col] for col in user_data_cols}
        m, n = len(self.input_data), len(theta)
        py_mp_par = list(pycmpfit.MpPar() for i in range(n))

        fit = pycmpfit.Mpfit(userfunc, m, theta, private_data=user_data, py_mp_par=py_mp_par)
        fit.mpfit()  # NOTE now theta has been updated
        mp_result = fit.result
        dof = mp_result.nfunc - mp_result.nfree
        chisq = mp_result.bestnorm

        theta_fin = theta
        self.covar = mp_result.covar
        if verbose:
            print('Fitted Parameters:', theta_fin)
            print('Covariance matrix: \n', self.covar)

        if self.model=='tripp':
            jac = np.array([-1, -self.input_data['c']+np.mean(self.input_data['c']), -self.input_data['dm15']+np.mean(self.input_data['dm15'])], dtype='object')
        elif self.model=='salt':
            jac = np.array([-1, self.input_data['x1'], -self.input_data['c']], dtype='object')
        elif self.model=='He2018':
            jac = np.array([-1, -(self.input_data['dm15'] - np.mean(self.input_data['dm15'])), -((self.input_data['bmax'] - self.input_data['bbv'])/self.input_data['slope'] + 1.2*(1/self.input_data['slope'] - np.mean(1/self.input_data['slope'])))], dtype='object')
        elif self.model=='Aldoroty2022':
            jac = np.array([-1, -(self.input_data['dm15'] - np.mean(self.input_data['dm15'])), -((self.input_data['bmax'] - self.input_data['bbv'])/self.input_data['slope'] - np.mean((self.input_data['bmax'] - self.input_data['bbv'])/self.input_data['slope']))], dtype='object')

        err = np.sqrt(np.matmul(jac, np.matmul(self.covar, jac.T)) + evpec**2)
        
        # theta_fin is the final fit parameters, M, a, d
        # the error on theta_fin is in the covariance matrix, self.covar
        # err is the error on MU for EACH SN

        mu = np.array([self.test_dm(theta_fin,self.input_data[i],self.input_data)[0] for i in range(len(self.input_data))])
        mu_expected = self.cosmo.distmod(self.input_data['zcmb']).value
        resid = mu-mu_expected

        self.data['data_mu'] = mu
        self.data['data_mu_err'] = err

        return theta_fin, mu, err, dof, chisq

    def loocv(self):
        """
        Do leave-one-out-cross-validation for your dataset.
        Generates a LOOCV Hubble Diagram and LOOCV Hubble
        residual. 
        """

        test_mus=[]
        test_resids=[]
        test_zcmbs=[]

        N = len(self.input_data)
        indices = np.arange(0,len(self.input_data))
        input_data_copy = copy(self.input_data)
        for i in range(N):
            mask=(indices==i)
            train_args = input_data_copy[~mask]
            test_args = input_data_copy[mask]
            train_params = self.minimize()[0] # [0] is the final param estimate, theta_fin
            test_mu, test_resid, test_zcmb = self.test_dm(train_params,test=test_args,train=train_args)

            train_covar = self.covar 
            

            test_mus.append(test_mu[0])
            test_resids.append(test_resid[0])
            test_zcmbs.append(test_zcmb[0])


        self.data['loocv_mu'] = test_mus
        self.data['loocv_resids'] = test_resids

        return test_mus, test_resids, test_zcmbs