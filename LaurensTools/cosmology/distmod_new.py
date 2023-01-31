import numpy as np
from scipy.optimize import minimize
from astropy.cosmology import FlatLambdaCDM
from kapteyn import kmpfit

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

"""

def evpec(z,vpec):
    speedoflight = 299792.458 # speed of light, km/s  
    return (5/np.log(10))*(vpec/(speedoflight*z)) # error from peculiar velocity, units are magnitudes 

class tripp():
    """
    Defines model, residual, and log likelihood functions for
    the standard distance modulus model:
    mu = Bmax - M - a*(c-np.mean(c)) - d*(dm15 - np.mean(dm15)) 
    """

    def model(self,p,data):
        M,a,d = p
        bmax,bvmax,dm15 = data
        return bmax - M - a*(bvmax-np.mean(bvmax)) - d*(dm15 - np.mean(dm15))

    def resid_func(self,p,data):
        M,a,d = p
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,vpec = data
        num = mu - (bmax - M - a*(bvmax-np.mean(bvmax)) - 
            d*(dm15 - np.mean(dm15)))
        den = np.sqrt(evpec(z,vpec)**2 + 
            ebmax**2 + a**2*ebvmax**2 + d**2*edm15**2)
        return num/den

    def log_likelihood(self,p,data):
        M,a,d,log_f = p
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,vpec = data
        sigma2 = ebmax**2 + a**2*ebvmax**2 + d**2*edm15**2 + self.model([M,a,d], [bmax,bvmax,dm15])**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((mu - self.model([M,a,d], [bmax,bvmax,dm15])) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

class salt():
    """
    Defines model, residual, and log likelihood functions for
    the standard SALT distance modulus model:
    mu = Bmax - M + a*x1 - b*c 
    """

    def model(self,p,data):
        M,a,b = p
        bmax,x1,c = data
        return bmax - M + a*x1 - b*c

    def resid_func(self,p,data):
        M,a,b = p
        mu,bmax,ebmax,x1,ex1,c,ec,z,vpec = data
        num = mu - (bmax - M + a*x1 - b*c)
        den = np.sqrt(evpec(z,vpec)**2 + 
            ebmax**2 + a**2*ex1**2 + b**2*ec**2)
        return num/den

    def log_likelihood(self,p,data):
        M,a,b,log_f = p
        mu,bmax,ebmax,x1,ex1,c,ec,z,vpec = data
        sigma2 = ebmax**2 + a**2*ex1**2 + b**2*ec**2 + self.model([M,a,b], [bmax,x1,c])**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((mu - self.model([M,a,b], [bmax,x1,c])) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

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
        self.z=z

        self.model = model
        self.cosmo = FlatLambdaCDM(H0=H0,Om0=Om0)
        self.mu = self.cosmo.distmod(self.z).value

        models = ['tripp','salt','H18','A23']
        if model not in models:
            raise(ValueError(f'Argument model must be in {models}.'))    
        elif self.model == 'tripp':
            self.mod = tripp()  
            self.input_data = (self.mu,self.bmax,self.ebmax,
                            self.bvmax,self.ebvmax,
                            self.dm15,self.edm15,
                            self.z,self.vpec)
        elif self.model == 'salt':
            self.mod = salt()
            self.input_data = (self.mu,self.bmax,self.ebmax,
                            self.x1,self.ex1,
                            self.c,self.ec,
                            self.z,self.vpec)

    def fit(self,fitmethod,initial_guess):
        """
        LNA 20230130

        IF USING fitmethod='ls', i.e. least-squares fitting:
        This is a wrapper for kapteyn.kmpfit,
        which is a wrapper for mpfit. Attributes
        for fitobj are:
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

        """

        fit_methods = ['ls','mle']
        if fitmethod not in fit_methods:
            raise(ValueError(f'Argument fitmethod must be in {fit_methods}.'))   
        elif fitmethod == 'ls':
            fitobj = kmpfit.Fitter(residuals=self.mod.resid_func, 
                                    data=self.input_data)
            fitobj.fit(params0=initial_guess)
            return fitobj

        elif fitmethod == 'mle':
            # Recall that for MLE, you have an additional log_f parameter to guess.
            # So, if you have 3 fit parameters in your model, you need to input a list
            # of length 4 with the guess for log_f as the last entry. 
            nll = lambda *args: -self.mod.log_likelihood(*args)
            fitobj = minimize(nll,initial_guess,args=(list(self.input_data)))
            return fitobj