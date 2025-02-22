import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
from lmfit import Model
import matplotlib.pyplot as plt

"""
NOTE TO SELF: This is pretty messy. I want to
rewrite it. 
"""

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def moving_avg(xvals, array):
    # Smooths out the spectrum so you don't get a bad pEW measurement 
    # from an unusual spike or dip in noisy data.
    # Use this to find the continuum only, not in Gaussian fitting. 
    mv_avg = []
    x_avg = []
    for i in np.arange(4,len(array)-4,1):
        mv_avg.append(np.mean([array[i+4],array[i+3],array[i+2],array[i+1],array[i],array[i-1],array[i-2],array[i-3],array[i-4]]))
        x_avg.append(xvals[i])
    return np.array(x_avg), np.array(mv_avg)

def gaus(x,a,x0,sigma):
    # The 1 is because I have normalized the spectrum to the 
    # continuum around the line.
    return 1 + a*(1/(sigma*np.sqrt(2*np.pi)))*np.exp(-(x-x0)**2/(2*sigma**2))

def normalize(wave,flux,start_lam,end_lam,var=None):
    """
    LNA 20211111
    Returns the normalized flux and wavelength over the wavelength range specified,
    i.e., around the continuum. 

    Keyword arguments --
    wave -- spectrum wavelength
    flux -- spectrum flux (not normalized)
    start_lam -- beginning endpoint for determining location of continuum
    end_lam -- ending endpoint for determining location of continuum
    var -- if supplied, trims variance array to same location as wave and flux arrays

    """
    wavavg, fluxavg = moving_avg(wave,flux)
    start_idx = find_nearest(wavavg, start_lam)
    end_idx = find_nearest(wavavg, end_lam)

    continuum = interp1d([wavavg[start_idx],wavavg[end_idx]], [fluxavg[start_idx], fluxavg[end_idx]],bounds_error=False)

    start_idx = find_nearest(wave, start_lam)
    end_idx = find_nearest(wave, end_lam)

    if var is None or var.all() == None:
        return wave[start_idx:end_idx], flux[start_idx:end_idx]/continuum(wave[start_idx:end_idx])
    else:
        return wave[start_idx:end_idx], flux[start_idx:end_idx]/continuum(wave[start_idx:end_idx]), var[start_idx:end_idx]

def pEW(wave, flux, var, start_lam, end_lam, absorption=True, return_error=True, just_the_pEW=True, plotline=True, saveplot=True, plotname='pEW.png'):
    """
    LNA 20210930
    Calculates the psuedo-equivalent width of a line.

    Keyword arguments --
    wave -- spectrum wavelength
    flux -- spectrum flux (not normalized)
    err -- spectrum error (sqrt(variance))
    start_lam -- beginning endpoint for determining location of continuum
    end_lam -- ending endpoint for determining location of continuum
    absorption -- Boolean. If True, the input line is an absorption line. If false, the input line is an emission line.
    just_the_pEW -- Boolean. If true, returns only the pseudo-equivalent width. If False, returns pEW, the normalized flux, 
                    its corresponding wavelengths, a function for the fit Gaussian, and the fit Gaussian parameters. 

    """
    normalized_wave,normalized_lineonly,var_lineonly = normalize(wave,flux,start_lam,end_lam,var)

    n = len(normalized_wave)
    mean = np.sum(normalized_wave*normalized_lineonly)/np.sum(normalized_lineonly)  
    sigma = np.sqrt(sum((normalized_wave-mean)**2)/n)
    gmodel = Model(gaus)

    if absorption == True:
        initial_guess = gmodel.make_params(a=-1,x0=mean,sigma=sigma)
        gmodel.set_param_hint('a', max=0)
        gmodel.set_param_hint('x0', min=mean-100, max=mean+100)
        # pars, cov = curve_fit(gaus,normalized_wave,normalized_lineonly, p0=[1,-1,mean,sigma], bounds=[[1-0.000001,-1.5,-np.inf,-np.inf],[1+0.000001,0,np.inf,np.inf]])

    elif absorption == False:
        initial_guess = gmodel.make_params(a=1,x0=mean,sigma=sigma)
        gmodel.set_param_hint('a', min=0)
        gmodel.set_param_hint('x0', min=mean-100, max=mean+100)
        # pars, cov = curve_fit(gaus,normalized_wave,normalized_lineonly, p0=[1,1,mean,sigma], bounds=[[1-0.000001,0,-np.inf,-np.inf],[1+0.000001,1.5,np.inf,np.inf]])
        

    err_lineonly = np.sqrt(var_lineonly)
    result = gmodel.fit(normalized_lineonly, initial_guess, x=normalized_wave, nan_policy='propagate')

    # if err == None and return_error==True:
    #     raise ValueError('Array must be supplied as an argument for err if return_error == True.')
    # elif err == None:
    #     result = gmodel.fit(normalized_lineonly, initial_guess, x=normalized_wave)
    # elif err.any() != None and return_error==True: 
    #     err_lineonly = err[normalized_mask]
    #     result = gmodel.fit(normalized_lineonly, initial_guess, x=normalized_wave, weights=1/err_lineonly)


    pardict = result.params.valuesdict()
    pars = [*pardict.values()]
    # print('fit parameters dictionary', pardict)

    if plotline:
        plt.plot(normalized_wave,normalized_lineonly)
        plt.plot(normalized_wave,gaus(normalized_wave,*pars))
        plt.xlabel('Wavelength')
        plt.ylabel('Normalized flux')
        
        if saveplot:
            plt.savefig(plotname, dpi=300, bbox_inches='tight')

        plt.show()
    
    stepsize=1
    wavelength_range = np.arange(start_lam,end_lam,stepsize)

    pew = abs(np.sum(1 - gaus(wavelength_range,*pars))*stepsize)
    # pew = np.sum((1-normalized_lineonly)*stepsize)

    if return_error:
        # scaled_err = err*np.sqrt(result.redchi)
        # dpew_dlam = -(pardict['a'])/(pardict['sigma']**2)*np.sum((normalized_wave - pardict['x0'])*np.exp(-(normalized_wave - pardict['x0'])**2/(2*pardict['sigma']**2)))
        # dpew_dlam0 = stepsize*(pardict['a'])/(pardict['sigma']**2)*np.sum((normalized_wave - pardict['x0'])*np.exp(-(normalized_wave - pardict['x0'])**2/(2*pardict['sigma']**2)))
        # dpew_da = stepsize*np.sum(np.exp(-(normalized_wave - pardict['x0'])**2/(2*pardict['sigma']**2)))
        # dpew_dsigma = stepsize*(pardict['a'])/(pardict['sigma']**3)*np.sum((normalized_wave - pardict['x0'])**2*np.exp(-(normalized_wave - pardict['x0'])**2/(2*pardict['sigma']**2)))
        
        # epew = np.sqrt(result.covar[0][0]**2*dpew_da**2 + result.covar[1][1]**2*dpew_dlam0**2 + result.covar[2][2]**2*dpew_dsigma**2)
        try:
            # epew = np.sqrt(2*np.pi)*np.sqrt(pardict['sigma']**2*result.covar[0][0]**2 + pardict['a']**2*result.covar[2][2]**2)
            # print('covariance matrix')
            # print(result.covar)
            # epew = np.sqrt(np.sum(var_lineonly))
            if pew < 6:
                epew = 0
            else:
                epew = np.sqrt(result.covar[0][0])

        except:
            epew = 0
            print('epEW calculation failed, value set to 0')
    else:
        epew = 0

    if just_the_pEW == True:
        return pew
    elif just_the_pEW == False:
        fitgauss = lambda x: gaus(x,*pars)

        return pew, epew, normalized_lineonly, normalized_wave, fitgauss, pars
    else:
        raise ValueError('just_the_pEW must be boolean, i.e., True or False')

def pEW_mcmc(wave,flux,var,start_lam,end_lam,absorption,just_the_pEW,Niter=225):
    pewlist = []
    for i in range(Niter):
        pewlist.append(pEW(wave, flux, var, start_lam-np.random.randint(-10,11), end_lam+np.random.randint(-10,11), absorption=absorption, return_error=False, just_the_pEW=just_the_pEW, plotline=False))
    
    return np.mean(pewlist), np.std(pewlist)

def line_velocity(wave, flux, var, start_lam, end_lam, rest_lam, absorption=True, plotline=True):
    """
    LNA20201007
    Calculates the velocity of an absorption or emission line in km/s.

    Keyword arguments --
    wave -- spectrum wavelength
    flux -- spectrum flux
    start_lam -- beginning endpoint for determining location of continuum
    end_lam -- ending endpoint for determining location of continuum
    rest_lam -- rest frame wavelength of line
    absorption -- Boolean. If True, the input line is an absorption line. If false, the input line is an emission line.

    """
    wave = np.array(wave)
    flux = np.array(flux)
    var = np.array(var)

    start_idx = find_nearest(wave,start_lam)
    end_idx = find_nearest(wave,end_lam)

    wave_lineonly = wave[start_idx:end_idx]
    flux_lineonly = flux[start_idx:end_idx]
    var_lineonly = var[start_idx:end_idx]

    n = len(wave_lineonly)
    mean = np.sum(wave_lineonly*flux_lineonly)/np.sum(flux_lineonly)  
    sigma = np.sqrt(sum((wave_lineonly-mean)**2)/n)

    continuum = interp1d([wave_lineonly[0],wave_lineonly[-1]], [flux_lineonly[0], flux_lineonly[-1]],bounds_error=False)

    flux_lineonly = flux_lineonly/continuum(wave_lineonly)

    gmodel = Model(gaus)

    if absorption == True:
        initial_guess = gmodel.make_params(a=-1,x0=mean,sigma=sigma)
        gmodel.set_param_hint('a', max=0)
        gmodel.set_param_hint('x0', min=mean-100, max=mean+100)
        # pars, cov = curve_fit(gaus,normalized_wave,normalized_lineonly, p0=[1,-1,mean,sigma], bounds=[[1-0.000001,-1.5,-np.inf,-np.inf],[1+0.000001,0,np.inf,np.inf]])
    elif absorption == False:
        initial_guess = gmodel.make_params(a=1,x0=mean,sigma=sigma)
        gmodel.set_param_hint('a', min=0)
        gmodel.set_param_hint('x0', min=mean-100, max=mean+100)
        # pars, cov = curve_fit(gaus,normalized_wave,normalized_lineonly, p0=[1,1,mean,sigma], bounds=[[1-0.000001,0,-np.inf,-np.inf],[1+0.000001,1.5,np.inf,np.inf]])
    else:
        raise ValueError('absorption must be boolean, i.e., True or False')

    result = gmodel.fit(flux_lineonly, initial_guess, x=wave_lineonly, weights=1/var_lineonly)

    pardict = result.params.valuesdict()
    pars = [*pardict.values()]
    
    if absorption == True:
        observed_location = np.argmin(gaus(wave_lineonly,*pars))
    elif absorption == False:
        observed_location = np.argmax(gaus(wave_lineonly,*pars))


    # if absorption == True:
    #     pars, cov = curve_fit(gaus,wave_lineonly,flux_lineonly, p0=[-1,mean,sigma], bounds=[[-1.5,-np.inf,-np.inf],[0,np.inf,np.inf]])
    #     observed_location = np.argmin(gaus(wave_lineonly,*pars))
    # elif absorption == False:
    #     pars, cov = curve_fit(gaus,wave_lineonly,flux_lineonly, p0=[1,mean,sigma], bounds=[[0,-np.inf,-np.inf],[1.5,np.inf,np.inf]])
    #     observed_location = np.argmax(gaus(wave_lineonly,*pars))
    # else:
    #     raise ValueError('absorption must be boolean, i.e., True or False')

    if plotline:
        plt.plot(wave_lineonly,flux_lineonly,marker='none')
        plt.plot(wave_lineonly,gaus(wave_lineonly,*pars),marker='none')
        plt.axvline(x=pars[1], linestyle='--',color='black')
        plt.xlabel('Wavelength')
        plt.ylabel('Normalized flux')
        
        # if saveplot:
        #     plt.savefig(plotname, dpi=300, bbox_inches='tight')

        plt.show()

    # Eqn 1, Galbany et al. 2016:
    v = 3E5*(((rest_lam - wave_lineonly[observed_location])/rest_lam + 1)**2 - 1)/(((rest_lam - wave_lineonly[observed_location])/rest_lam + 1)**2 + 1)

    dv_d_obs_lam = 3E5*(4*rest_lam*(wave_lineonly[observed_location]**2 - 3*rest_lam*wave_lineonly[observed_location] + rest_lam**2))/(wave_lineonly[observed_location]**2 - 4*rest_lam*wave_lineonly[observed_location] + 5*rest_lam**2)**2

    ev = abs(dv_d_obs_lam*result.covar[1][1])
    ev_2 = 3E5*result.covar[1][1]/wave_lineonly[observed_location]

    return v, ev, ev_2