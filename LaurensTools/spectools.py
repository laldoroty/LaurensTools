import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def moving_avg(xvals, array):
    # Smooths out the spectrum so you don't get a bad pEW measurement from an unusual spike or dip in noisy data.
    # Use this to find the continuum only, not in Gaussian fitting. 
    mv_avg = []
    x_avg = []
    for i in np.arange(4,len(array)-4,1):
        mv_avg.append(np.mean([array[i+4],array[i+3],array[i+2],array[i+1],array[i],array[i-1],array[i-2],array[i-3],array[i-4]]))
        x_avg.append(xvals[i])
    return np.array(x_avg), np.array(mv_avg)

def gaus(x,shift,a,x0,sigma):
    return shift + a*np.exp(-(x-x0)**2/(2*sigma**2))

def pEW(wave, flux, start_lam, end_lam, absorption=True, just_the_pEW=True, plotline=True, saveplot=True, plotname='pEW.png'):
    """
    LNA 20210930
    Calculates the psuedo-equivalent width of a line.

    Keyword arguments --
    wave -- spectrum wavelength
    flux -- spectrum flux
    start_lam -- beginning endpoint for determining location of continuum
    end_lam -- ending endpoint for determining location of continuum
    absorption -- Boolean. If True, the input line is an absorption line. If false, the input line is an emission line.
    just_the_pEW -- Boolean. If true, returns only the pseudo-equivalent width. If False, returns pEW, the normalized flux, 
                    its corresponding wavelengths, a function for the fit Gaussian, and the fit Gaussian parameters. 

    """
    wavavg, fluxavg = moving_avg(wave,flux)

    start_idx = find_nearest(wavavg, start_lam)
    end_idx = find_nearest(wavavg, end_lam)

    continuum = interp1d([wavavg[start_idx],wavavg[end_idx]], [fluxavg[start_idx], fluxavg[end_idx]],bounds_error=False)

    normalized = flux/continuum(wave)
    normalized_mask = np.invert(np.isnan(normalized))
    if absorption == True:
        normalized_lineonly = normalized[normalized_mask]
    elif absorption == False:
        normalized_lineonly = normalized[normalized_mask]
    else:
        raise ValueError('absorption must be boolean, i.e., True or False')

    normalized_wave = wave[normalized_mask]

    n = len(normalized_wave)
    mean = np.sum(normalized_wave*normalized_lineonly)/np.sum(normalized_lineonly)  
    sigma = np.sqrt(sum((normalized_wave-mean)**2)/n)

    if absorption == True:
        pars, cov = curve_fit(gaus,normalized_wave,normalized_lineonly, p0=[1,-1,mean,sigma], bounds=[[1-0.000001,-1.5,-np.inf,-np.inf],[1+0.000001,0,np.inf,np.inf]])

    elif absorption == False:
        pars, cov = curve_fit(gaus,normalized_wave,normalized_lineonly, p0=[1,1,mean,sigma], bounds=[[1-0.000001,0,-np.inf,-np.inf],[1+0.000001,1.5,np.inf,np.inf]])

    if plotline:
        plt.plot(normalized_wave,normalized_lineonly)
        plt.plot(normalized_wave,gaus(normalized_wave,pars[0],pars[1],pars[2],pars[3]))
        plt.xlabel('Wavelength')
        plt.ylabel('Normalized flux')
        
        if saveplot:
            plt.savefig(plotname, dpi=300, bbox_inches='tight')

        plt.show()

    # fwhm = 2.355*pars[3]
    # amp = pars[1]

    # pew = abs(fwhm*amp)

    # pew = abs(pars[1]*pars[3]*np.sqrt(2*np.pi))
    
    stepsize=1
    wavelength_range = np.arange(start_lam,end_lam,stepsize)
    # pew = (end_lam-start_lam) - np.trapz(wavelength_range, gaus(wavelength_range,pars[0],pars[1],pars[2],pars[3]))

    pew = np.sum(1 - gaus(wavelength_range,*pars))*stepsize

    if pew < 0:
        pew = 0

    if just_the_pEW == True:
        return pew
    elif just_the_pEW == False:
        fitgauss = lambda x: gaus(x,pars[0],pars[1],pars[2],pars[3])

        return pew, normalized, normalized_wave, continuum, fitgauss, pars
    else:
        raise ValueError('just_the_pEW must be boolean, i.e., True or False')


def line_velocity(wave, flux, start_lam, end_lam, true_lam, absorption=True):
    """
    LNA20201007
    Calculates the velocity of an absorption or emission line in km/s.

    Keyword arguments --
    wave -- spectrum wavelength
    flux -- spectrum flux
    start_lam -- beginning endpoint for determining location of continuum
    end_lam -- ending endpoint for determining location of continuum
    absorption -- Boolean. If True, the input line is an absorption line. If false, the input line is an emission line.

    """

    start_idx = find_nearest(wave,start_lam)
    end_idx = find_nearest(wave,end_lam)

    wave_lineonly = wave[start_idx:end_idx]
    flux_lineonly = flux[start_idx:end_idx]

    n = len(wave_lineonly)
    mean = np.sum(wave_lineonly*flux_lineonly)/np.sum(flux_lineonly)  
    sigma = np.sqrt(sum((wave_lineonly-mean)**2)/n)

    if absorption == True:
        pars, cov = curve_fit(gaus,wave_lineonly,flux_lineonly, p0=[1,-1,mean,sigma], bounds=[[1-0.000001,-1.5,-np.inf,-np.inf],[1+0.000001,0,np.inf,np.inf]])
        observed_location = np.argmin(gaus(wave_lineonly,pars[0],pars[1],pars[2],pars[3]))
    elif absorption == False:
        pars, cov = curve_fit(gaus,wave_lineonly,flux_lineonly, p0=[1,1,mean,sigma], bounds=[[1-0.000001,0,-np.inf,-np.inf],[1+0.000001,1.5,np.inf,np.inf]])
        observed_location = np.argmax(gaus(wave_lineonly,pars[0],pars[1],pars[2],pars[3]))
    else:
        raise ValueError('absorption must be boolean, i.e., True or False')

    # Eqn 1, Galbany et al. 2016:
    v = 3E5*(((true_lam - wave_lineonly[observed_location])/true_lam + 1)**2 - 1)/(((true_lam - wave_lineonly[observed_location])/true_lam + 1)**2 + 1)

    return v