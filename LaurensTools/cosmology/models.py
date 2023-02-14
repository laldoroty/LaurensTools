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
        return ['M','a','d','log(f)']

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
        if -20 < M < -16 and 0 < a < 6 and 0 < d < 3 and -10 < log_f < 10:
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
        return ['M','a','b','log(f)']

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
        if -20 < M < -16 and -3 < a < 3 and -1 < b < 1 and -10 < log_f < 10:
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