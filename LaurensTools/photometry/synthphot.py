import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.io import fits


def synth_lc_tophat(wave,flux,var,lower_filter_edge,upper_filter_edge,zp,ezp):
    """
    20210520 LNA
    Make a synthetic magnitude from a spectrum using a top hat filter.
    """
    int_ = []
    var_ = []
    wav_ = []

    for i in range(len(wave)):
        if wave[i] >= lower_filter_edge and wave[i] <= upper_filter_edge:
            int_.append(flux[i])
            var_.append(var[i])
            wav_.append(wave[i])
            
    flux_ = np.trapz(int_,wav_)

    var_ = np.array(var_)
    fluxerr = np.sqrt(np.sum(var_))
    mag = -2.5*np.log10(flux_/zp)
    
    # This assumes no error in zero point:
    #emag = 1.09*(np.sqrt(np.sum(var_))/flux_)
    # This includes error in zero point:
    emag = np.sqrt((1.09*fluxerr/flux_)**2 + 2*(1.09*ezp/zp)**2)
    
    return mag, emag

def synth_lc_bessel(wave,flux,var,standard='vega'):
    """
    Make a synthetic magnitude from a spectrum using Bessel filters. 
    Vega is the only available standard, currently. 
    Response function from: https://ui.adsabs.harvard.edu/abs/2012PASP..124..140B/abstract
    Zero points from table 3.
    """

    responsefunc = pd.read_csv('bessel_simon_2012_UBVRI_response.csv')
    photometry = {'U': None,
           'B': None,
           'V': None,
           'R': None,
           'I': None}
    ephotometry = {'U': None,
           'B': None,
           'V': None,
           'R': None,
           'I': None}
    band_index = {'U': 0,
           'B': 1,
           'V': 2,
           'R': 3,
           'I': 4}

    if standard=='vega':
        hdul=fits.open('alpha_lyr_stis_010.fits')
        # Units are ergs/s/cm^2/AA
        st_wav = hdul[1].data['WAVELENGTH']
        st_flux = hdul[1].data['FLUX']
        st_flux_error = np.sqrt(hdul[1].data['STATERROR']**2 + hdul[1].data['SYSERROR']**2)
        zeropoints = {'U': -0.023,
                       'B': -0.023,
                       'V': -0.023,
                       'R': -0.023,
                       'I': -0.023}
    
    cov = np.zeros((5,5))
    covref = np.zeros((5,5))
    jacobian = []
    diag = []
    meanflux = []
    spectra = []
    
    for band in photometry:
        if band == 'U':
            spectra.append(np.zeros(len(flux)))
            meanflux.append(0)
            diag.append(0)
            jacobian.append(0)
        elif band != 'U':
            F = 0
            Fref = 0
            s = []
            d = 0

            responsefunc_interp = interp1d(responsefunc['lam_%s' % band], responsefunc['%s' % band],kind='linear')

            for i in range(len(wave)):
                if wave[i] > min(responsefunc['lam_%s' % band]) and wave[i] < max(responsefunc['lam_%s' % band]):
                    F += flux[i]*responsefunc_interp(wave[i])*(wave[i]-wave[i-1])
                    d += var[i]
                    s.append(flux[i])
                else:
                    s.append(0)
            diag.append(d)
            meanflux.append(np.sum(s)/(len(wave)-1))
            spectra.append(s)

            for i in range(len(st_wav)):
                if st_wav[i] > min(responsefunc['lam_%s' % band]) and st_wav[i] < max(responsefunc['lam_%s' % band]):
                    Fref += st_flux[i]*responsefunc_interp(st_wav[i])*(st_wav[i]-st_wav[i-1])

            jacobian.append(1.09/F)
            photometry[band] = -2.5*np.log10(F/Fref) + zeropoints[band]
    jacobian = np.array(jacobian)

    for i in range(len(cov)):
        for j in range(len(cov[i])):
            if i == j:
                cov[i,i] = diag[i]

            elif i != j and cov[i,j] == 0:
                sigsq = 0
                for k in range(len(spectra[i])):
                    sigsq += (spectra[i][k]-meanflux[i])*(spectra[j][k]-meanflux[j])
                sigsq = (1/(len(spectra[i])-1))*sigsq
                cov[i,j] = sigsq
                cov[j,i] = sigsq
                
    for band in ephotometry:
        ephotometry[band] = np.sqrt(np.matmul(jacobian,(np.matmul(cov,jacobian.T))))
                
    return photometry, ephotometry