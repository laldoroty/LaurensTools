import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.io import fits

dirname = os.path.dirname(os.path.abspath(__file__))

def load_vegaspec():
    """
    20220603 LNA
    Loads the Vega spectrum into your code, wherever you are. 
    
    """
    hdul=fits.open(os.path.join(dirname,'alpha_lyr_stis_010.fits'))
    # Units are ergs/s/cm^2/AA
    st_wav = hdul[1].data['WAVELENGTH']
    st_flux = hdul[1].data['FLUX']
    st_flux_error = np.sqrt(hdul[1].data['STATERROR']**2 + hdul[1].data['SYSERROR']**2)
    return np.array(st_wav), np.array(st_flux), np.array(st_flux_error)

def load_filtersys(sys):
    """
    20220627 LNA
    Loads the transmission curves for the specified filter system.
    sys -- filter system. Options are 'bessell', 'sdss', 'lsst', 'lsst_filteronly'.
    Bessell response function from: https://ui.adsabs.harvard.edu/abs/2012PASP..124..140B/abstract Zero points from table 3. Compare errors to the errors in: https://arxiv.org/pdf/0910.3330.pdf 
    For SDSS, response function from: https://arxiv.org/abs/1002.3701 (http://www.ioa.s.u-tokyo.ac.jp/~doi/sdss/SDSSresponse.html)
    SDSS zeropoints are copied from SNooPy.

    ***NEED TO CHECK SDSS FILTERS AND ZEROPOINTS.***
    """

    filtersys = {}

    if sys == 'bessell':
        filters = ['U','B','V','R','I']
        responsefunc = pd.read_csv(os.path.join(dirname,'bessel_simon_2012_UBVRI_response.csv'))
        zeropoints = {'U': -0.023,
                       'B': -0.023,
                       'V': -0.023,
                       'R': -0.023,
                       'I': -0.023}
        for band in filters:
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}

    elif sys == 'sdss':
        filters = ['u','g','r','i','z']
        zeropoints = {'u': 12.4757864,
        'g': 14.2013159905,
        'r': 14.2156544329,
        'i': 13.7775438954,
        'z': 11.8525822106}

        for band in filters:
            responsefunc = pd.read_csv(os.path.join(dirname,f'filters/sdss_filters/{band}6.dat'))
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}

    elif sys == 'lsst':
        zeropoints = {}
        filters = ['u','g','r','i','z','y']
        for band in filters:
            responsefunc = pd.read_csv(os.path.join(dirname,f'filters/lsst_filters/full/LSST_LSST.{band}.dat'),sep=' ')
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}
            zeropoints[band] = 0.0

    elif sys == 'lsst_filteronly':
        zeropoints = {}
        filters = ['u','g','r','i','z','y']
        for band in filters:
            responsefunc = pd.read_csv(os.path.join(dirname,f'filters/lsst_filters/filter_only/LSST_LSST.{band}_filter.dat'), sep=' ')
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}
            zeropoints[band] = 0.0

    return filtersys, zeropoints


def synth_lc_tophat(wave,flux,var,lower_filter_edge,upper_filter_edge,zp,ezp):
    """
    20210520 LNA
    Make a synthetic magnitude from a spectrum using a top hat filter.

    Keyword arguments:
    wave -- Spectrum wavelength
    flux -- Spectrum flux
    var -- Spectrum variance
    lower_filter_edge -- Lower bound on top hat filter
    upper_filter_edge -- Upper bound on top hat filter
    zp -- Zero point
    ezp -- Zero point error

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

def synth_lc(wave,flux,var,sys=None,standard='vega',convert_to_ergs=False):
    """
    Make a synthetic magnitude from a spectrum using Bessell, SDSS, or LSST filters. 
    Vega is the only available standard, currently. 

    Keyword arguments:
    wave -- Data spectrum wavelength
    flux -- Data spectrum flux
    var -- Data spectrum variance
    sys -- Which filter system to use. Valid values are: ['bessell', 'sdss', 'lsst', 'lsst_filteronly'].
    standard -- Which standard spectrum to use. Valid values are: ['vega'].
    """

    if sys is None:
        raise ValueError('Must specify filter system.')

    h = 6.626E-27 # Planck constant, erg s
    c = 3E10 # Speed of light, cm/s
    photometry = {}
    ephotometry = {}

    if standard=='vega':
        st_wav, st_flux, st_flux_error = load_vegaspec()
        responsefunc, zeropoints = load_filtersys(sys)
        filters = [x for x in responsefunc.keys()]

        F = 0
        Fref = 0
        var_F = 0

        for band in filters:
            if convert_to_ergs:
                responsefunc['%s' % band] = h*c*responsefunc['%s' % band]
                normalization_const = responsefunc['%s' % band].max()
                responsefunc['%s' % band] = responsefunc['%s' % band]/normalization_const

            responsefunc_interp = interp1d(responsefunc['lam_%s' % band], responsefunc['%s' % band],kind='linear')

            for i in range(len(wave)):
                if wave[i] > min(responsefunc['lam_%s' % band]) and wave[i] < max(responsefunc['lam_%s' % band]):
                    F += flux[i]*responsefunc_interp(wave[i])*(wave[i]-wave[i-1])
                    var_F += var[i]*responsefunc_interp(wave[i])**2*(wave[i]-wave[i-1])**2

            for i in range(len(st_wav)):
                if st_wav[i] > min(responsefunc['lam_%s' % band]) and st_wav[i] < max(responsefunc['lam_%s' % band]):
                    Fref += st_flux[i]*responsefunc_interp(st_wav[i])*(st_wav[i]-st_wav[i-1])

            photometry[band] = -2.5*np.log10(F/Fref) + zeropoints[band]
            ephotometry[band] = np.sqrt((1.09/F)**2*var_F)
                
    return photometry, ephotometry