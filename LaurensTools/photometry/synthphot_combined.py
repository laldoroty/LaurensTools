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
        # NOTE: Built-in Bessell filters are normalized photons/cm^2/s/A.
        responsefunc = pd.read_csv(os.path.join(dirname,'filters/bessel_simon_2012_UBVRI_response.csv'))
        zeropoints = {'U': -0.023,
                       'B': -0.023,
                       'V': -0.023,
                       'R': -0.023,
                       'I': -0.023}
        for band in zeropoints.keys():
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}

    elif sys == 'sdss':
        zeropoints = {'u': 12.4757864,
        'g': 14.2013159905,
        'r': 14.2156544329,
        'i': 13.7775438954,
        'z': 11.8525822106}

        for band in zeropoints.keys():
            responsefunc = pd.read_csv(os.path.join(dirname,f'filters/sdss_filters/{band}6.dat'))
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}

    elif sys == 'lsst':
        # LSST filters are given in photon count transmission.
        # ZPs are in wavelength units, not frequency. And Vega system.
        zeropoints = {'u': 4.45E-9,
        'g': 5.22E-9,
        'r': 2.46E-9,
        'i': 1.37E-9,
        'z': 9.04E-10,
        'y': 6.95E-10}
        for band in zeropoints.keys():
            responsefunc = pd.read_csv(os.path.join(dirname,f'filters/lsst_filters/full/LSST_LSST.{band}.dat'),sep=' ')
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}

    elif sys == 'lsst_filteronly':
        # LSST filters are given in photon count transmission. 
        # ZPs are in wavelength units, not frequency. And Vega system.
        zeropoints = {'u': 3.99E-9,
        'g': 5.31E-9,
        'r': 2.48E-9,
        'i': 1.37E-9,
        'z': 9.02E-10,
        'y': 6.17E-10}
        for band in zeropoints.keys():
            responsefunc = pd.read_csv(os.path.join(dirname,f'filters/lsst_filters/filter_only/LSST_LSST.{band}_filter.dat'), sep=' ')
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}

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

def synth_lc(wave,flux,var,sys=None,standard='vega',convert_spec=None,convert_response=None):
    """
    Make a synthetic magnitude from a spectrum using Bessell, SDSS, or LSST filters. 
    Vega is the only available standard, currently. 

    Keyword arguments:
    wave -- Data spectrum wavelength
    flux -- Data spectrum flux
    var -- Data spectrum variance
    sys -- Which filter system to use. Valid values are: ['bessell', 'sdss', 'lsst', 'lsst_filteronly'].
    standard -- Which standard spectrum to use. Valid values are: ['vega'].
    convert_spec -- 'ergs' or 'photons'. 'ergs' converts the input data spectrum to ergs. 'photons' converts it to photons. 
    convert_response -- 'ergs' or 'photons'. 'ergs' converts the response function to ergs. 'photons' converts it to photons. 
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

        if convert_spec == 'ergs':
            flux *= h*c/wave
            # Vega spectrum already in ergs. 
        elif convert_spec == 'photons':
            flux /= h*c/wave
            st_flux /= h*c/wave
        elif convert_spec is None:
            print('Data spectrum units not converted.')
        else: 
            assert convert_spec is not None and convert_spec != 'ergs' and convert_spec != 'photons', "Acceptable inputs for convert_spec are ergs or photons!" 

        F = 0
        Fref = 0
        var_F = 0

        for band in filters:
            if convert_response == 'ergs':
                responsefunc[band]['transmission'] *= h*c/responsefunc[band]['wavelength']
                normalization_const = responsefunc[band]['transmission'].max()
                responsefunc[band]['transmission'] /= normalization_const
            elif convert_response == 'photons':
                responsefunc[band]['transmission'] /= h*c/responsefunc[band]['wavelength']
                normalization_const = responsefunc[band]['transmission'].max()
                responsefunc[band]['transmission'] /= normalization_const
            elif convert_response is None:
                print('Response function units not converted.')
            else: 
                assert convert_response is not None and convert_response != 'ergs' and convert_response != 'photons', "Acceptable inputs for convert_response are ergs or photons!" 

            responsefunc_interp = interp1d(responsefunc[band]['wavelength'], responsefunc[band]['transmission'],kind='linear')

            for i in range(len(wave)):
                if wave[i] > min(responsefunc[band]['wavelength']) and wave[i] < max(responsefunc[band]['wavelength']):
                    F += flux[i]*responsefunc_interp(wave[i])*(wave[i]-wave[i-1])
                    var_F += var[i]*responsefunc_interp(wave[i])**2*(wave[i]-wave[i-1])**2

            for i in range(len(st_wav)):
                if st_wav[i] > min(responsefunc[band]['wavelength']) and st_wav[i] < max(responsefunc[band]['wavelength']):
                    Fref += st_flux[i]*responsefunc_interp(st_wav[i])*(st_wav[i]-st_wav[i-1])

            photometry[band] = -2.5*np.log10(F/Fref) + zeropoints[band]
            ephotometry[band] = np.sqrt((1.09/F)**2*var_F)
                
    return photometry, ephotometry