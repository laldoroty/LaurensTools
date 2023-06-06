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

    def name(self):
        return 'tripp'

    def param_names(self):
        """
        Parameter names with corresponding initializing bounds for
        MCMC fitting. 
        """
        return {'M': [-20,-18],'a': [-4,4],'d': [-2,1.5],'log(f)': [-10,0]}

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

    def name(self):
        return 'salt'

    def param_names(self):
        """
        Parameter names with corresponding initializing bounds for
        MCMC fitting. 
        """
        return {'M': [-21,-17], 'a': [-1,3], 'b': [-3,6], 'log(f)': [-10,0]}

    def model(self,p,data):
        M,a,b = p
        mu,bmax,ebmax,x1,ex1,c,ec,frni,efrni,pew4000,epew4000,z,evpec = data # delete frni and pew4000 later
        return bmax - M - a*x1 - b*c

    def resid_func(self,p,data):
        M,a,b = p
        mu,bmax,ebmax,x1,ex1,c,ec,frni,efrni,pew4000,epew4000,z,evpec = data # delete frni and pew4000 later
        num = self.model(p,data) - mu
        den = np.sqrt(evpec**2 + 
            ebmax**2 + a**2*ex1**2 + b**2*ec**2)
        return num/den

    def log_likelihood(self,p,data):
        M,a,b,log_f = p
        mu,bmax,ebmax,x1,ex1,c,ec,frni,efrni,pew4000,epew4000,z,evpec = data # delete frni and pew4000 later
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
        mu,bmax,ebmax,x1,ex1,c,ec,frni,efrni,pew4000,epew4000,z,evpec = data # delete frni and pew4000 later
        return np.array([-np.ones(len(x1)), x1, -c], dtype='object').T
    
# class FRNi():
#     """
#     Defines model, residual, Jacobian, and log likelihood 
#     functions for the standard distance modulus model with
#     F R(Ni II) replacing dm15, and the SALT3 color parameter:
#     mu = Bmax - M + a*frni - b*c 
#     """

#     def name(self):
#         return 'FRNi'

#     def param_names(self):
#         """
#         Parameter names with corresponding initializing bounds for
#         MCMC fitting. 
#         """
#         return {'M': [-26,-18], 'a': [-1,3], 'b': [-2,5], 'log(f)': [-10,10]}

#     def model(self,p,data):
#         M,a,b = p
#         mu,bmax,ebmax,frni,efrni,c,ec,z,evpec = data
#         return bmax - M - a*frni - b*c

#     def resid_func(self,p,data):
#         M,a,b = p
#         mu,bmax,ebmax,frni,efrni,c,ec,z,evpec = data
#         num = self.model(p,data) - mu
#         den = np.sqrt(evpec**2 + 
#             ebmax**2 + a**2*efrni**2 + b**2*ec**2)
#         return num/den

#     def log_likelihood(self,p,data):
#         M,a,b,log_f = p
#         mu,bmax,ebmax,frni,efrni,c,ec,z,evpec = data
#         sigma2 = evpec**2 + ebmax**2 + a**2*efrni**2 + b**2*ec**2 + self.model([M,a,b], data)**2 * np.exp(2 * log_f)
#         return -0.5 * np.sum((self.model([M,a,b], data) - mu) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

#     def log_prior(self,p):
#         M,a,b,log_f = p
#         bounds_dict = self.param_names()
#         if bounds_dict['M'][0] < M < bounds_dict['M'][1] \
#         and bounds_dict['a'][0] < a < bounds_dict['a'][1] \
#         and bounds_dict['b'][0] < b < bounds_dict['b'][1] \
#         and bounds_dict['log(f)'][0] < log_f < bounds_dict['log(f)'][1]:
#             return 0.0
#         else: return -np.inf

#     def log_probability(self,p,*data):
#         lp = self.log_prior(p)
#         if not np.isfinite(lp):
#             return -np.inf
#         else: return lp + self.log_likelihood(p,data)

#     def jac(self,data):
#         mu,bmax,ebmax,frni,efrni,c,ec,z,evpec = data
#         return np.array([-np.ones(len(frni)), frni, -c], dtype='object').T

class H18():
    """
    Defines model, residual, Jacobian, and log likelihood
    functions for the CMAGIC distance model from 
    He et al. 2018:
    mu = BBV - M - delta*(dm15 - np.mean(dm15)) - 
        (b2 - slope)*((bmax - BBV)/slope) + 0.6 + 1.2*(1/slope - np.mean(1/slope)))
    """

    def name(self):
        return 'H18'
    
    def param_names(self):
        return {'M': [-20,-16], 'delta': [-2,2], 'b2': [-5,5], 'log(f)': [-20,10]}
    
    def model(self,p,data):
        M,delta,b2 = p
        mu,bmax,ebmax,dm15,edm15,bbv,ebbv,slope,eslope,z,evpec = data
        return bbv - M - delta*(dm15 - np.mean(dm15)) - \
                (b2 - slope)*(((bmax-bbv)/slope) + 0.6 + \
                    1.2*(1/slope - np.mean(1/slope)))
    
    def resid_func(self,p,data):
        M,delta,b2 = p
        mu,bmax,ebmax,dm15,edm15,bbv,ebbv,slope,eslope,z,evpec = data
        num = self.model(p,data) - mu

        dbbv = b2/slope
        ddm15 = -delta
        dbmax = -(b2 - slope)/slope
        # dslope = 1.2*b2*slope**(-2) + 1.2*b2*np.mean(1/slope) + 0.6 - 1.2*np.mean(1/slope)
        dslope = ((bmax - bbv)/slope + 1.2*((1/slope) - np.mean(1/slope))) + (slope - b2)*(-(bmax - bbv)/slope**2 - 1.2/slope**2)
        # dslope = 0
        den = np.sqrt(ebbv**2*dbbv**2 + edm15**2*ddm15**2 + 
                        ebmax**2*dbmax**2 + eslope**2*dslope**2 + evpec**2)
        return num/den
    
    def log_likelihood(self,p,data):
        M,delta,b2,log_f = p
        mu,bmax,ebmax,dm15,edm15,bbv,ebbv,slope,eslope,z,evpec = data

        dbbv = b2/slope
        ddm15 = -delta
        dbmax = -(b2 - slope)/slope
        # dslope = 1.2*b2*slope**(-2) + 1.2*b2*np.mean(1/slope) + 0.6 - 1.2*np.mean(1/slope)
        dslope = ((bmax - bbv)/slope + 1.2*((1/slope) - np.mean(1/slope))) + (slope - b2)*(-(bmax - bbv)/slope**2 - 1.2/slope**2)
        # dslope = 0
        sigma2 = ebbv**2*dbbv**2 + edm15**2*ddm15**2 + ebmax**2*dbmax**2 + eslope**2*dslope**2 + evpec**2 + self.model([M,delta,b2], data)**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((self.model([M,delta,b2], data) - mu) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

    def log_prior(self,p):
        M,delta,b2,log_f = p
        bounds_dict = self.param_names()
        if bounds_dict['M'][0] < M < bounds_dict['M'][1] \
        and bounds_dict['delta'][0] < delta < bounds_dict['delta'][1] \
        and bounds_dict['b2'][0] < b2 < bounds_dict['b2'][1] \
        and bounds_dict['log(f)'][0] < log_f < bounds_dict['log(f)'][1]:
            return 0.0
        else: return -np.inf

    def log_probability(self,p,*data):
        lp = self.log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else: return lp + self.log_likelihood(p,data)

    def jac(self,data):
        mu,bmax,ebmax,dm15,edm15,bbv,ebbv,slope,eslope,z,evpec = data
        return np.array([-np.ones(len(bmax)), -(dm15 - np.mean(dm15)), 
                         -((bmax - bbv)/slope + 0.6 + 1.2*(1/slope - np.mean(1/slope)))], 
                         dtype='object').T
    
class A23():
    """
    Defines model, residual, Jacobian, and log likelihood
    functions for the CMAGIC distance model from 
    Aldoroty et al. 2023:
    mu = BBV - M - delta*(dm15 - np.mean(dm15)) - 
        (b2 - slope)*((bmax - BBV)/slope) - np.mean((bmax - BBV)/slope))
    """

    def name(self):
        return 'A23'
    
    def param_names(self):
        return {'M': [-20,-18], 'delta': [-3,3], 'b2': [-3,5], 'log(f)': [-10,0]}
    
    def model(self,p,data):
        M,delta,b2 = p
        mu,bmax,ebmax,dm15,edm15,bbv,ebbv,slope,eslope,z,evpec = data
        return bbv - M - delta*(dm15 - np.mean(dm15)) - \
                (b2 - slope)*(((bmax-bbv)/slope) - np.mean((bmax-bbv)/slope))
    
    def resid_func(self,p,data):
        M,delta,b2 = p
        mu,bmax,ebmax,dm15,edm15,bbv,ebbv,slope,eslope,z,evpec = data
        num = self.model(p,data) - mu

        dbbv = b2/slope
        ddm15 = -delta
        dbmax = -(b2 - slope)/slope
        dslope = b2*(bmax - bbv)*slope**(-2) - np.mean((bmax-bbv)/slope)
        den = np.sqrt(ebbv**2*dbbv**2 + edm15**2*ddm15**2 + 
                        ebmax**2*dbmax**2 + eslope**2*dslope**2 + evpec**2)
        return num/den
    
    def log_likelihood(self,p,data):
        M,delta,b2,log_f = p
        mu,bmax,ebmax,dm15,edm15,bbv,ebbv,slope,eslope,z,evpec = data

        dbbv = b2/slope
        ddm15 = -delta
        dbmax = -(b2 - slope)/slope
        dslope = b2*(bmax - bbv)*slope**(-2) - np.mean((bmax-bbv)/slope)
        sigma2 = ebbv**2*dbbv**2 + edm15**2*ddm15**2 + ebmax**2*dbmax**2 + eslope**2*dslope**2 + evpec**2 + self.model([M,delta,b2], data)**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((self.model([M,delta,b2], data) - mu) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

    def log_prior(self,p):
        M,delta,b2,log_f = p
        bounds_dict = self.param_names()
        if bounds_dict['M'][0] < M < bounds_dict['M'][1] \
        and bounds_dict['delta'][0] < delta < bounds_dict['delta'][1] \
        and bounds_dict['b2'][0] < b2 < bounds_dict['b2'][1] \
        and bounds_dict['log(f)'][0] < log_f < bounds_dict['log(f)'][1]:
            return 0.0
        else: return -np.inf

    def log_probability(self,p,*data):
        lp = self.log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else: return lp + self.log_likelihood(p,data)

    def jac(self,data):
        mu,bmax,ebmax,dm15,edm15,bbv,ebbv,slope,eslope,z,evpec = data
        return np.array([-np.ones(len(bmax)), -(dm15 - np.mean(dm15)), 
                         (bbv - bmax)/slope + np.mean((bmax - bbv)/slope)], 
                         dtype='object').T

class Slope():
    """
    Defines model, residual, Jacobian, and log likelihood
    functions for the CMAGIC distance model from 
    Aldoroty et al. 2023:
    mu = BBV - M - b2*((bmax - BBV)/slope) - np.mean((bmax - BBV)/slope))
    """
    def name(self):
        return 'slope'
    
    def param_names(self):
        return {'M': [-20,-18], 'b2': [-5,5], 'log(f)': [-10,10]}
    
    def model(self,p,data):
        M,b2 = p
        mu,bmax,ebmax,bbv,ebbv,slope,eslope,z,evpec = data
        return bbv - M - b2*(((bmax-bbv)/slope) - np.mean((bmax-bbv)/slope))
    
    def resid_func(self,p,data):
        M,b2 = p
        mu,bmax,ebmax,bbv,ebbv,slope,eslope,z,evpec = data
        num = self.model(p,data) - mu

        dbbv = b2/slope
        dbmax = -(b2 - slope)/slope
        dslope = b2*(bmax - bbv)*slope**(-2) - np.mean((bmax-bbv)/slope)
        den = np.sqrt(ebbv**2*dbbv**2 + 
                        ebmax**2*dbmax**2 + eslope**2*dslope**2 + evpec**2)
        return num/den
    
    def log_likelihood(self,p,data):
        M,b2,log_f = p
        mu,bmax,ebmax,bbv,ebbv,slope,eslope,z,evpec = data

        dbbv = b2/slope
        dbmax = -(b2 - slope)/slope
        dslope = b2*(bmax - bbv)*slope**(-2) - np.mean((bmax-bbv)/slope)
        sigma2 = ebbv**2*dbbv**2 + ebmax**2*dbmax**2 + eslope**2*dslope**2 + evpec**2 + self.model([M,b2], data)**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((self.model([M,b2], data) - mu) ** 2 / sigma2 + np.log(sigma2)) + np.log(2*np.pi)

    def log_prior(self,p):
        M,b2,log_f = p
        bounds_dict = self.param_names()
        if bounds_dict['M'][0] < M < bounds_dict['M'][1] \
        and bounds_dict['b2'][0] < b2 < bounds_dict['b2'][1] \
        and bounds_dict['log(f)'][0] < log_f < bounds_dict['log(f)'][1]:
            return 0.0
        else: return -np.inf

    def log_probability(self,p,*data):
        lp = self.log_prior(p)
        if not np.isfinite(lp):
            return -np.inf
        else: return lp + self.log_likelihood(p,data)

    def jac(self,data):
        mu,bmax,ebmax,bbv,ebbv,slope,eslope,z,evpec = data
        return np.array([-np.ones(len(bmax)),  
                         (bbv - bmax)/slope + np.mean((bmax - bbv)/slope)], 
                         dtype='object').T