import numpy as np
import matplotlib.pyplot as plt
from copy import copy, deepcopy
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
from kapteyn import kmpfit
from numba import jit
from .support_funcs import *
from .models import *
from ..stats import wRMS
import emcee
import os

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
class HubbleDiagram():
    def __init__(self,model,H0=70,Om0=0.3,
                    bmax=None, ebmax=None,
                    bvmax=None, ebvmax=None,
                    dm15=None, edm15=None,
                    x1=None, ex1=None,
                    c=None, ec=None,
                    frni=None, efrni=None, # delete later
                    pew4000=None, epew4000=None, # delete this later
                    bbv=None, ebbv=None,
                    slope=None, eslope=None,
                    vpec=None,
                    z=None,
                    names=None,
                    info_tags=None):

        self.bmax,self.ebmax=np.array(bmax,dtype='float64'),np.array(ebmax,dtype='float64')
        self.bvmax,self.ebvmax=np.array(bvmax,dtype='float64'),np.array(ebvmax,dtype='float64')
        self.dm15,self.edm15=np.array(dm15,dtype='float64'),np.array(edm15,dtype='float64')
        self.x1,self.ex1=np.array(x1,dtype='float64'),np.array(ex1,dtype='float64')
        self.c,self.ec=np.array(c,dtype='float64'),np.array(ec,dtype='float64')
        self.frni,self.efrni=np.array(frni,dtype='float64'),np.array(efrni,dtype='float64') # delete later
        self.pew4000,self.epew4000=np.array(pew4000,dtype='float64'),np.array(epew4000,dtype='float64') # delete this later
        self.bbv,self.ebbv = np.array(bbv,dtype='float64'),np.array(ebbv,dtype='float64')
        self.slope,self.eslope = np.array(slope,dtype='float64'),np.array(eslope,dtype='float64')
        self.z=np.array(z,dtype='float64')
        self.vpec,self.evpec=np.array(vpec,dtype='float64'),evpec(self.z,np.array(vpec,dtype='float64'))
        
        self.model = model
        self.cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
        self.mu = self.cosmo.distmod(self.z).value

        self.names = np.array(names)
        self.info_tags = np.array(info_tags)

        self.fit_params = None
        self.resids = None
        self.eresids = None

        models = ['tripp','salt','FRNi','H18','A23','slope']
        # models = ['tripp','salt','H18','A23','slope']
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
                            self.frni,self.efrni, # delete later
                            self.pew4000,self.epew4000, # delete later
                            self.z,self.evpec]
        elif self.model == 'FRNi':
            self.mod = FRNi()
            self.input_data = [self.mu,self.bmax,self.ebmax,
                            self.frni,self.efrni,
                            self.c,self.ec,
                            self.z,self.evpec]
        elif self.model == 'H18' or self.model == 'A23':
            self.input_data = [self.mu,self.bmax,self.ebmax,
                            self.dm15,self.edm15,
                            self.bbv,self.ebbv,
                            self.slope,self.eslope,
                            self.z,self.evpec]
            if self.model == 'H18':
                self.mod = H18()
            elif self.model == 'A23':
                self.mod = A23()
        elif self.model == 'slope':
            self.mod = Slope()
            self.input_data = [self.mu,self.bmax,self.ebmax,
                            self.bbv,self.ebbv,
                            self.slope,self.eslope,
                            self.z,self.evpec]

        # Finally, ditch everything with NaNs. The fitters
        # don't like them. If there are NaNs, of course. 
        nan_check = [np.isnan(np.array(dlist)).any() for dlist in self.input_data]
        if np.array(nan_check).any():
            old_input_data = deepcopy(self.input_data)
            self.input_data = []
            drop_idx = []
            for dlist in old_input_data:
                idx = np.argwhere(np.isnan(dlist)).flatten()
                for i in idx:
                    if i not in drop_idx:
                        drop_idx.append(i)

            mask = [~np.isin(i,drop_idx) for i in range(len(old_input_data[0]))]
            for dlist in old_input_data:
                self.input_data.append(dlist[mask])

            if self.model == 'tripp':
                self.mu,self.bmax,self.ebmax, \
                self.bvmax,self.ebvmax, \
                self.dm15,self.edm15, \
                self.z,self.evpec = self.input_data
            elif self.model == 'salt':
                self.mu,self.bmax,self.ebmax, \
                self.x1,self.ex1, \
                self.c,self.ec, \
                self.frni,self.efrni, \
                self.pew4000,self.epew4000, \
                self.z,self.evpec = self.input_data
            elif self.model == 'FRNi':
                self.mu,self.bmax,self.ebmax, \
                self.frni,self.efrni, \
                self.c,self.ec, \
                self.z,self.evpec = self.input_data
            elif self.model == 'H18' or self.model == 'A23':
                self.mu,self.bmax,self.ebmax, \
                self.dm15,self.edm15, \
                self.bbv,self.ebbv, \
                self.slope,self.eslope, \
                self.z,self.evpec = self.input_data
            elif self.model == 'slope':
                self.mu,self.bmax,self.ebmax, \
                self.bbv,self.ebbv, \
                self.slope,self.eslope, \
                self.z,self.evpec = self.input_data

    def fit(self,fitmethod,initial_guess,scale_errors=False,mcmc_niter=5000,
            plot_mcmc_diagnostics=False,save_mcmc_diagnostic_plots=True,
            savepath='distmod_figs',snf=False):
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
        Iterations:                 , self.niter
        Number of fit parameters:   , self.ndim
        Number of walkers           , self.nwalkers
        emcee object:               , self.sampler
        MCMC sample chain:          , self.flat_samples
        Best-fit parameters:        , self.params
        Lower & upper error         , self.xerror

        as well as anything else accessible from 
        self.sampler, which is an instance of:
        https://emcee.readthedocs.io/en/stable/user/sampler/ 

        Set snf=True if using SNfactory data. This will
        hide the zeropoint in the diagnostic plots. 

        """

        fit_methods = ['ls','mle','mcmc']
        if fitmethod not in fit_methods:
            raise(ValueError(f'Argument fitmethod must be in {fit_methods}.')) 
        
        def get_pars(p):
            if fitmethod == 'ls':
               return p.params
            elif fitmethod == 'mle':
                return p.x
            elif fitmethod == 'mcmc':
                return p.params
   
        if fitmethod == 'ls':
            fitobj = kmpfit.Fitter(residuals=self.mod.resid_func, 
                                    data=tuple(self.input_data))
            fitobj.fit(params0=initial_guess)

            if scale_errors:
                print('Errors will be scaled such that chi^2 = dof.')
                redchi = deepcopy(fitobj.rchi2_min)
                scaled_input_data = deepcopy(self.input_data)
                error_vars = [self.ebmax,self.ebvmax,self.edm15,
                                self.ex1,self.ec,
                                self.ebbv,self.eslope,
                                self.evpec]
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

            self.fit_params = get_pars(fitobj)
            self.resids = self.mod.model(self.fit_params,self.input_data) - self.mu
            self.eresids = np.array(err)

            return fitobj, self.resids, self.eresids

        elif fitmethod == 'mle' or fitmethod == 'mcmc':
            # MLE and MCMC largely follow https://emcee.readthedocs.io/en/stable/tutorials/line/.
            # Recall that for MLE, you have an additional log_f parameter to guess.
            # So, if you have 3 fit parameters in your model, you need to input a list
            # of length 4 with the guess for log_f as the last entry. 
            nll = lambda *args: -self.mod.log_likelihood(*args)
            fitobj = minimize(nll,initial_guess,args=(self.input_data))

            if fitmethod == 'mle':
                err = np.zeros(len(self.input_data[0]))
                self.fit_params = get_pars(fitobj)[:-1]
                self.resids = self.mod.model(self.fit_params,self.input_data) - self.mu
                self.eresids = err
                return fitobj, self.resids, self.eresids
            elif fitmethod == 'mcmc':
                mc = emcee_object(self.mod)
                fitmc = mc.run_emcee(mcmc_niter,self.input_data)
                
                if plot_mcmc_diagnostics:
                    if savepath is not 'distmod_figs':
                        savepath = os.path.join(savepath,'distmod_figs')
                    path_exists = os.path.exists(savepath)
                    if not path_exists:
                        print('Creating save directory for MCMC diagnostic plots:',
                              '\n',os.path.join(os.getcwd(),savepath))
                        os.makedirs(savepath)
                    mc.plot_diagnostics_(self.mod.param_names().keys(),save_mcmc_diagnostic_plots,
                                         savepath,snf=snf)
                    print('savepath for mcmc diag. plot', savepath)

                all_par_est = mc.flat_samples.T[:-1].T
                all_possible_models = np.apply_along_axis(self.mod.model,1,arr=all_par_est,data=self.input_data)

                err = np.sqrt(np.apply_along_axis(np.std,0,all_possible_models)**2 + self.evpec**2)
                self.fit_params = get_pars(mc)[:-1]
                self.resids = self.mod.model(self.fit_params,self.input_data) - self.mu
                self.eresids = err
                return mc, self.resids, self.eresids

    def wrms(self):
        return wRMS(self.resids,self.eresids)

    # @jit(forceobj=True)
    # def loocv(self,fitmethod,initial_guess,scale_errors=False):
    #     """
    #     Leave-one-out cross-validation for your dataset. 
    #     Generates LOOCV distance moduli and residuals. 
    #     """

    #     def get_pars(p):
    #         if fitmethod == 'ls':
    #            return p.params
    #         elif fitmethod == 'mle':
    #             return p.x
    #         elif fitmethod == 'mcmc':
    #             return p.params

    #     test_mus, test_resids, test_errs = []

    #     N = len(self.input_data[0])
    #     indices = np.arange(0,len(self.input_data[0]+1,1))
    #     original_input_data = copy(self.input_data)
    #     for i in range(N):
    #         mask=(indices==i)
    #         # Need to redefine self.input_data so self.fit() will use
    #         # it without data i. So, from here on, self.input_data
    #         # is the training data set, missing data point i. We'll
    #         # put it back to its original state at the end. This is
    #         # probably bad practice, but that's how it's going to be
    #         # for now. 
    #         self.input_data = [dat[~mask] for dat in original_input_data]
    #         test_args = original_input_data[mask]
    #         train_fitobj, train_err = self.fit(fitmethod,initial_guess,scale_errors)
    #         test_mu = self.mod.model(get_pars(train_fitobj),test_args)

    #         ### THIS FUNCTION IS INCOMPLETE. 

    #     # Now that we're done, put your self.input_data back to its rightful place.
    #     self.input_data = original_input_data

