import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
from kapteyn import kmpfit
from numba import jit
from .support_funcs import *
import emcee
import corner

"""
Here's how you use this:

from LaurensTools.cosmology import distmod_new

# Only input the data you need as arrays. Be specific to what they are,
# as shown in the example below, because otherwise the order matters.
hd = distmod_new.HubbleDiagram(bmax=bmax,
                            ebmax=ebmax,
                            bvmax=c,
                            ebvmax=ec,
                            dm15=dm15,
                            edm15=edm15,
                            vpec=vpec,
                            z=zcmb))

# We specify a least squares fit:
guess = [-18,1.5,0.5]
res = hd.fit('ls',guess)
pars = res.params
cov = res.covar

# Or we can specify maximum likelihood estimation:
guess_2 = [-18,1.5,0.5,0.5]
res_2 = hd.fit('mle',guess_2)
params = res_2.x

In this module, we heavily borrow from:
https://emcee.readthedocs.io/en/stable/tutorials/line

"""

def evpec(z,vpec):
    speedoflight = 299792.458 # speed of light, km/s  
    return (5/np.log(10))*(vpec/(speedoflight*z)) # error from peculiar velocity, units are magnitudes 

class tripp():
    """
    Defines model, residual, Jacobian, and log likelihood 
    functions for the standard distance modulus model:
    mu = Bmax - M - a*(c-np.mean(c)) - d*(dm15 - np.mean(dm15)) 
    """

    def param_names(self):
        return ['M','a','d','log(f)']

    def model(self,p,data):
        M,a,d = p
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,evpec = data
        return bmax - M - a*(bvmax-np.mean(bvmax)) - d*(dm15 - np.mean(dm15))

    def resid_func(self,p,data):
        M,a,d = p
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,evpec = data
        num = mu - self.model(p,data)
        den = np.sqrt(evpec**2 + 
            ebmax**2 + a**2*ebvmax**2 + d**2*edm15**2)
        return num/den

    def log_likelihood(self,p,data):
        M,a,d,log_f = p
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,evpec = data
        sigma2 = evpec**2 + ebmax**2 + a**2*ebvmax**2 + d**2*edm15**2 + self.model([M,a,d], data)**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((mu - self.model([M,a,d], data)) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

    def log_prior(self,p):
        M,a,d,log_f = p
        if -20 < M < -16 and 0 < a < 6 and 0 < d < 3 and -10 < log_f < 10:
            return 0.0
        else: return -np.inf

    def log_probability(self,p,data):
        lp = self.log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else: return lp + self.log_likelihood(p,data)

    def jac(self,data):
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,evpec = data
        return np.array([-np.ones(len(bvmax)), -bvmax+np.mean(bvmax), -dm15+np.mean(dm15)], dtype='object').T

class HubbleDiagram():
    def __init__(self,model,H0=70,Om0=0.3,
                    bmax=None, ebmax=None,
                    bvmax=None, ebvmax=None,
                    dm15=None, edm15=None,
                    x1=None, ex1=None,
                    c=None, ec=None,
                    vpec=None,
                    z=None):

        self.bmax=bmax
        self.ebmax=ebmax
        self.bvmax=bvmax
        self.ebvmax=ebvmax
        self.dm15=dm15
        self.edm15=edm15
        self.x1=x1
        self.ex1=ex1
        self.c=c
        self.ec=ec
        self.vpec=vpec
        self.evpec=evpec(z,vpec)
        self.z=z

        self.model = model
        self.cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
        self.mu = self.cosmo.distmod(self.z).value

        models = ['tripp','salt','H18','A23']
        if model not in models:
            raise(ValueError(f'Argument model must be in {models}.'))    
        elif self.model == 'tripp':
            self.mod = tripp()  
            self.input_data = [self.mu,self.bmax,self.ebmax,
                            self.bvmax,self.ebvmax,
                            self.dm15,self.edm15,
                            self.z,self.evpec]
        elif self.model == 'salt':
            self.mod = salt()
            self.input_data = [self.mu,self.bmax,self.ebmax,
                            self.x1,self.ex1,
                            self.c,self.ec,
                            self.z,self.evpec]

    def fit(self,fitmethod,initial_guess,scale_errors=False):
        """
        LNA 20230130

        IF USING fitmethod='ls', i.e. least-squares fitting:
        This is a wrapper for kapteyn.kmpfit,
        which is a wrapper for mpfit. Returns fitobj,
        which is the object from kapteyn.kmpfit, and
        the error for your data points. 

        Attributes for fitobj are:
        https://www.astro.rug.nl/software/kapteyn/kmpfittutorial.html
        Best-fit parameters:        , fitobj.params)
        Asymptotic error:           , fitobj.xerror)
        Error assuming red.chi^2=1: , fitobj.stderr)
        Chi^2 min:                  , fitobj.chi2_min)
        Reduced Chi^2:              , fitobj.rchi2_min)
        Iterations:                 , fitobj.niter)
        Number of free pars.:       , fitobj.nfree)
        Degrees of freedom:         , fitobj.dof) 
        Covariance matrix:          , fitobj.covar)

        IF USING fitmethod='mle', i.e., maximum likelihood estimation:
        This is a wrapper for scipy.optimize.minimize. Attributes are:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html#scipy.optimize.OptimizeResult
        Best-fit parameters:        , fitobj.x)
        Objective function:         , fitobj.fun)
        Jacobian:                   , fitobj.jac)
        Hessian:                    , fitobj.hess)
        Inverse Hessian:            , fitobj.hess_inv)
        Iterations:                 , fitobj.nit)

        If using fitmethod='mcmc', attributes are:
        Iterations: self.niter
        Number of fit parameters: self.ndim
        self.nwalkers
        self.fitobj
        self.sampler
        self.flat_samples
        self.params
        self.xerror

        """

        fit_methods = ['ls','mle','mcmc']
        if fitmethod not in fit_methods:
            raise(ValueError(f'Argument fitmethod must be in {fit_methods}.'))   
        elif fitmethod == 'ls':
            fitobj = kmpfit.Fitter(residuals=self.mod.resid_func, 
                                    data=tuple(self.input_data))
            fitobj.fit(params0=initial_guess)

            if scale_errors:
                print('Errors will be scaled such that chi^2 = dof.')
                redchi = deepcopy(fitobj.rchi2_min)
                scaled_input_data = deepcopy(self.input_data)
                error_vars = [self.ebmax,self.ebvmax,self.edm15,
                                self.ex1,self.ec,self.evpec]
                for var in error_vars:
                    try:
                        idx = [idx for idx, ele in enumerate(self.input_data) if np.array_equal(ele,var)][0]
                        scaled_input_data[idx] *= np.sqrt(redchi)
                    except IndexError: pass

                fitobj = kmpfit.Fitter(residuals=self.mod.resid_func, 
                                    data=tuple(scaled_input_data))
                fitobj.fit(params0=initial_guess)

                jacobian = self.mod.jac(scaled_input_data)
                err = np.zeros(len(scaled_input_data[0]))
                evpec_err = deepcopy(self.evpec)*np.sqrt(redchi)
            else:
                jacobian = self.mod.jac(self.input_data)
                err = np.zeros(len(self.input_data[0]))
                evpec_err = deepcopy(self.evpec)

            for i, j in enumerate(jacobian):
                err[i] = np.sqrt(float(j @ fitobj.covar @ j.T) + evpec_err[i]**2)

            return fitobj, np.array(err)

        elif fitmethod == 'mle' or fitmethod == 'mcmc':
            # MLE and MCMC largely follow https://emcee.readthedocs.io/en/stable/tutorials/line/.
            # Recall that for MLE, you have an additional log_f parameter to guess.
            # So, if you have 3 fit parameters in your model, you need to input a list
            # of length 4 with the guess for log_f as the last entry. 
            nll = lambda *args: -self.mod.log_likelihood(*args)
            fitobj = minimize(nll,initial_guess,args=(self.input_data))

            if fitmethod == 'mle':
                err = np.zeros(len(self.input_data[0]))
                return fitobj, err
            elif fitmethod == 'mcmc':
                ### TODO: THIS DOES NOT WORK YET
                mc = emcee_object()
                print('self.input_data')
                print(len(self.input_data))
                print(self.input_data)
                fitmc = mc.run_emcee(fitobj,5000,self.mod.log_probability,self.input_data)
                def plot_diagnostics(self):
                    mc.plot_diagnostics_(self)

                return mc.params, mc.xerror


    @jit(forceobj=True)
    def loocv(self,fitmethod,initial_guess,scale_errors=False):
        """
        Leave-one-out cross-validation for your dataset. 
        Generates LOOCV distance moduli and residuals. 
        """

        def get_pars(p):
            if fitmethod == 'ls':
               return p.params
            elif fitmethod == 'mle':
                return p.x

        test_mus, test_resids, test_errs = []

        N = len(self.input_data[0])
        indices = np.arange(0,len(self.input_data[0]+1,1))
        original_input_data = copy(self.input_data)
        for i in range(N):
            mask=(indices==i)
            # Need to redefine self.input_data so self.fit() will use
            # it without data i. So, from here on, self.input_data
            # is the training data set, missing data point i. We'll
            # put it back to its original state at the end. 
            self.input_data = [dat[~mask] for dat in original_input_data]
            test_args = original_input_data[mask]
            train_fitobj, train_err = self.fit(fitmethod,initial_guess,scale_errors)
            test_mu = self.mod.model(get_pars(train_fitobj),test_args)

            ### THIS FUNCTION IS INCOMPLETE. 

        # Now that we're done, put your self.input_data back to its rightful place.
        self.input_data = original_input_data