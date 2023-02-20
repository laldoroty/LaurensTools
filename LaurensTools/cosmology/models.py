import numpy as np

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
        """
        Parameter names with corresponding initializing bounds for
        MCMC fitting. 
        """
        return {'M': [-22,-16],'a': [0,6],'d': [0,3],'log(f)': [-10,10]}

    def model(self,p,data):
        M,a,d = p
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,evpec = data
        return bmax - M - a*(bvmax-np.mean(bvmax)) - d*(dm15 - np.mean(dm15))

    def resid_func(self,p,data):
        M,a,d = p
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,evpec = data
        num = self.model(p,data) - mu
        den = np.sqrt(evpec**2 + 
            ebmax**2 + a**2*ebvmax**2 + d**2*edm15**2)
        return num/den

    def log_likelihood(self,p,data):
        M,a,d,log_f = p
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,evpec = data
        sigma2 = evpec**2 + ebmax**2 + a**2*ebvmax**2 + d**2*edm15**2 + self.model([M,a,d], data)**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((self.model([M,a,d], data) - mu) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

    def log_prior(self,p):
        M,a,d,log_f = p
        bounds_dict = self.param_names()
        if bounds_dict['M'][0] < M < bounds_dict['M'][1] \
        and bounds_dict['a'][0] < a < bounds_dict['a'][1] \
        and bounds_dict['d'][0] < d < bounds_dict['d'][1] \
        and bounds_dict['log(f)'][0] < log_f < bounds_dict['log(f)'][1]:
            return 0.0
        else: return -np.inf

    def log_probability(self,p,*data):
        lp = self.log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else: return lp + self.log_likelihood(p,data)

    def jac(self,data):
        mu,bmax,ebmax,bvmax,ebvmax,dm15,edm15,z,evpec = data
        return np.array([-np.ones(len(bvmax)), -bvmax+np.mean(bvmax), -dm15+np.mean(dm15)], dtype='object').T

class salt():
    """
    Defines model, residual, Jacobian, and log likelihood 
    functions for the standard distance modulus model with
    SALT parameters as inputs:
    mu = Bmax - M + a*x1 - b*c 
    """

    def param_names(self):
        """
        Parameter names with corresponding initializing bounds for
        MCMC fitting. 
        """
        return {'M': [-22,-17], 'a': [-2,2], 'b': [-2,5], 'log(f)': [-10,10]}

    def model(self,p,data):
        M,a,b = p
        mu,bmax,ebmax,x1,ex1,c,ec,z,evpec = data
        return bmax - M - a*x1 - b*c

    def resid_func(self,p,data):
        M,a,b = p
        mu,bmax,ebmax,x1,ex1,c,ec,z,evpec = data
        num = self.model(p,data) - mu
        den = np.sqrt(evpec**2 + 
            ebmax**2 + a**2*ex1**2 + b**2*ec**2)
        return num/den

    def log_likelihood(self,p,data):
        M,a,b,log_f = p
        mu,bmax,ebmax,x1,ex1,c,ec,z,evpec = data
        sigma2 = evpec**2 + ebmax**2 + a**2*ex1**2 + b**2*ec**2 + self.model([M,a,b], data)**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((self.model([M,a,b], data) - mu) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

    def log_prior(self,p):
        M,a,b,log_f = p
        bounds_dict = self.param_names()
        if bounds_dict['M'][0] < M < bounds_dict['M'][1] \
        and bounds_dict['a'][0] < a < bounds_dict['a'][1] \
        and bounds_dict['b'][0] < b < bounds_dict['b'][1] \
        and bounds_dict['log(f)'][0] < log_f < bounds_dict['log(f)'][1]:
            return 0.0
        else: return -np.inf

    def log_probability(self,p,*data):
        lp = self.log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else: return lp + self.log_likelihood(p,data)

    def jac(self,data):
        mu,bmax,ebmax,x1,ex1,c,ec,z,evpec = data
        return np.array([-np.ones(len(x1)), x1, -c], dtype='object').T