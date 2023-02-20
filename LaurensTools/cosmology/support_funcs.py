import numpy as np
import matplotlib.pyplot as plt
import emcee
import corner

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
    def __init__(self):
        self.niter, self.ndim, self.nwalkers = None, None, None
        self.fitobj, self.sampler = None, None
        self.flat_samples = None
        self.params = []
        self.xerror = []
    
    def run_emcee(self,fitobj,niter,model,data):
        self.niter = niter
        self.fitobj = fitobj
        ninit = 32
        # pos = fitobj.x + 1e-4 * np.random.randn(32,len(fitobj.x))
        pos = np.zeros((32,len(fitobj.x))).T
        for i, par in enumerate(model.param_names().keys()):
            pos[i] = np.random.uniform(model.param_names()[par][0],model.param_names()[par][1],ninit)
        
        pos = pos.T
        self.nwalkers, self.ndim = pos.shape
        self.sampler = emcee.EnsembleSampler(self.nwalkers,self.ndim,model.log_probability,args=tuple(data))
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

    def plot_diagnostics_(self,param_names):
        fig_1, axes = plt.subplots(len(self.fitobj.x), figsize=(10,15), sharex=True)
        samples = self.sampler.get_chain()
        labels = list(param_names)
        for i in range(self.ndim):
            ax = axes[i]
            ax.plot(samples[:,:,i], 'k', alpha=0.2, marker='None')
            ax.set_xlim(0,len(samples))
            ax.set_ylabel(labels[i])
        axes[-1].set_xlabel("Step number")
        plt.tight_layout()
        plt.show()

        fig_2 = corner.corner(self.flat_samples,labels=labels)
