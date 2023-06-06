import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner
import os.path as pa
from copy import deepcopy
from ..utils import plotaesthetics

class emcee_object():
    """
    Creates emcee results object so that in the
    main module, fitmethod='mcmc' can be treated
    exactly the same as the other fitmethods,
    i.e., so the results are attributes of an
    object. 

    Attributes are:
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
    """
    def __init__(self,model):
        self.model = model
        self.niter, self.ndim, self.nwalkers = None, None, None
        self.npars, self.sampler = len(self.model.param_names().keys()), None
        self.flat_samples = None
        self.params = []
        self.xerror = []
    
    def run_emcee(self,niter,data):
        self.niter = niter
        ninit = 32
        pos = np.zeros((32,self.npars)).T
        for i, par in enumerate(self.model.param_names().keys()):
            pos[i] = np.random.uniform(self.model.param_names()[par][0],self.model.param_names()[par][1],ninit)
        
        pos = pos.T
        self.nwalkers, self.ndim = pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,self.model.log_probability,args=tuple(data))
        run = self.sampler.run_mcmc(pos,self.niter,progress=True,skip_initial_state_check=False)

        # Discard and flatten
        tau = self.sampler.get_autocorr_time()
        discard = int(2*np.max(tau))
        self.flat_samples = self.sampler.get_chain(discard=discard,thin=15,flat=True)

        for i in range(self.ndim):
            mcmc = np.percentile(self.flat_samples[:,i], [16,50,84])
            q = np.diff(mcmc)
            self.params.append(np.array(mcmc[1]))
            mcmc_errors = np.array([q[0],q[1]])
            self.xerror.append(mcmc_errors)

        self.params = np.array(self.params)
        self.xerror = np.array(self.xerror)

    def plot_diagnostics_(self,param_names,save,savepath,snf=False):
        """
        Set snf=True if you are using SNfactory data.
        This will hide the zeropoint for M.
        """
        
        fig_1, axes = plt.subplots(self.npars, figsize=(10,15), sharex=True)
        samples = self.sampler.get_chain()
        labels = list(param_names)
        for i in range(self.ndim):
            ax = axes[i]
            if snf:
                samples[:,:,0] = samples[:,:,0] - np.mean(samples[:,:,0])
                labels[0] = 'M - <M>'
                ax.plot(samples[:,:,i], 'k', alpha=0.2, marker='None')
            else:
                ax.plot(samples[:,:,i], 'k', alpha=0.2, marker='None')
            ax.set_xlim(0,len(samples))
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("Step number")
        plt.tight_layout()
        if save:
            plt.savefig(pa.join(savepath,f'mcmc_chains_{self.model.name()}.png'),
                        dpi=300,bbox_inches='tight')
        plt.show()

        if snf:
            snf_flat_samples = deepcopy(self.flat_samples)
            snf_flat_samples = np.transpose(snf_flat_samples)
            mean_M = np.mean(snf_flat_samples[0])
            snf_flat_samples[0] = snf_flat_samples[0] - mean_M
            snf_flat_samples = np.transpose(snf_flat_samples)
            labels[0] = 'M - <M>'
            fig_2 = corner.corner(snf_flat_samples,labels=labels)
        else:
            fig_2 = corner.corner(self.flat_samples,labels=labels)
        
        fig_2.subplots_adjust(right=2,top=2) # Necessary if you're using the
                                             # LaurensTools.utils.plotaesthetics
        if save:
            fig_2.savefig(pa.join(savepath,f'mcmc_corner_{self.model.name()}.png'),
                        dpi=300,bbox_inches='tight')