import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit

def pEW(wave, flux, start_lam, end_lam, absorption=True, just_the_pEW=True):
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

    def gaus(x,shift,a,x0,sigma):
        return shift + a*np.exp(-(x-x0)**2/(2*sigma**2))

    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array-value)).argmin()
        return idx

    start_idx = find_nearest(wave, start_lam)
    end_idx = find_nearest(wave, end_lam)

    continuum = interp1d([wave[start_idx],wave[end_idx]], [flux[start_idx], flux[end_idx]],bounds_error=False)

    normalized = flux/continuum(wave)
    normalized_mask = np.invert(np.isnan(normalized))
    if absorption == True:
        # The +100 is a vertical shift to ensure all values are positive. 
        # Otherwise, the Gaussian fit doesn't work.
        # Same for the (-1)--we want the Gaussian to be pointing *up*. 
        normalized = normalized[normalized_mask]*(-1)+100
    elif absorption == False:
        normalized = normalized[normalized_mask]+100
    else:
        raise ValueError('absorption must be boolean, i.e., True or False')

    normalized_wave = wave[normalized_mask]

    n = len(normalized_wave)
    mean = np.sum(normalized_wave*normalized)/np.sum(normalized)  
    sigma = np.sqrt(sum((normalized_wave-mean)**2)/n)

    pars, cov = curve_fit(gaus,normalized_wave,normalized, p0=[100,1,mean,sigma])

    ### UNDO THE SHIFT AND SIGN FLIP. 
    fitgauss = lambda x: gaus(x,pars[0],pars[1],pars[2],pars[3])

    fwhm = 2.355*pars[3]
    amp = pars[1]

    pew = abs(fwhm*amp)

    if just_the_pEW == True:
        return pew
    elif just_the_pEW == False:
        return pew, normalized, normalized_wave, fitgauss, pars
    else:
        raise ValueError('just_the_pEW must be boolean, i.e., True or False')