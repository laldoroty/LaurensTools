import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from astropy.io import fits
import matplotlib.pyplot as plt

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
    sys -- filter system. Options are 'bessell', 'sdss', 'lsst', 'lsst_filteronly', 'snftophat'.
    Bessell response function from: https://ui.adsabs.harvard.edu/abs/2012PASP..124..140B/abstract Zero points from table 3. Compare errors to the errors in: https://arxiv.org/pdf/0910.3330.pdf 
    For SDSS, response function from: https://arxiv.org/abs/1002.3701 (http://www.ioa.s.u-tokyo.ac.jp/~doi/sdss/SDSSresponse.html)
    SDSS zeropoints are copied from SNooPy.

    ***NEED TO CHECK SDSS FILTERS AND ZEROPOINTS.***
    """

    filtersys = {}

    if sys == 'bessell':
        # NOTE: Built-in Bessell filters are normalized photons/cm^2/s/A.
        responsefunc = pd.read_csv(os.path.join(dirname,'filters/bessel_simon_2012_UBVRI_response.csv'))
        zeropoints = {'U': 0.0,
                       'B': 0.0,
                       'V': 0.0,
                       'R': 0.0,
                       'I': 0.0}

        # THIS MAKES THIS ONLY WORK FOR VEGA MAGS RIGHT NOW. 
        vwave,vflux,verr = load_vegaspec()

        for band in zeropoints.keys():
            filtersys[band] = {'wavelength': responsefunc[f'lam_{band}'], 'transmission': responsefunc[f'{band}']}
            filter_interp = interp1d(responsefunc[f'lam_{band}'],responsefunc[f'{band}'],bounds_error=False)
            dlam = np.diff(vwave)
            filt = np.nan_to_num(filter_interp(vwave[1:]))
            numerator = np.sum(vwave[1:]*filt*vflux[1:]*dlam)
            denominator = np.sum(vwave[1:]*filt*dlam)
            zeropoints[band] = -2.5*np.log10(numerator/denominator)

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
            zeropoints[band] = 2.5*np.log10(zeropoints[band])

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
            zeropoints[band] = 2.5*np.log10(zeropoints[band])
    elif sys == 'snftophat':
        filter_edges = {'u': (3300., 4102.),
            'b': (4102., 5100.),
            'v': (5200., 6289.),
            'r': (6289., 7607.),
            'i': (7607., 9200.)}
        filtersys = {}
        zeropoints = {}
        for fname, edges in filter_edges.items():
            wave = [edges[0]-1., edges[0], edges[1], edges[1]+1.]
            trans = [0., 1., 1., 0.]
            filtersys[fname] = {'wavelength': wave, 'transmission': trans}
            zeropoints[fname] = 0.

    return filtersys, zeropoints

def synth_lc_tophat(wave,flux,var,lower_filter_edge=None,upper_filter_edge=None,standard='vega',spec_units='ergs',verbose=False):
    """
    20210520 LNA
    Make a synthetic magnitude from a spectrum using a top hat filter.

    Keyword arguments:
    wave -- Spectrum wavelength
    flux -- Spectrum flux
    var -- Spectrum variance
    lower_filter_edge -- Lower bound on top hat filter
    upper_filter_edge -- Upper bound on top hat filter

    """
    acceptable_spec_units = ['ergs','photons']
    if lower_filter_edge is None or upper_filter_edge is None:
        raise ValueError('Must specify filter edges.')

    h = 6.626E-27 # Planck constant, erg s
    c = 3E18 # Speed of light, AA/s

    if standard=='vega':
        st_wav, st_flux, st_flux_error = load_vegaspec()

        if spec_units in acceptable_spec_units:
            if spec_units == 'ergs':
                if verbose:
                    print('Spectrum units ergs/cm^2/s/AA. Converting to photons.')
                flux /= h*c/wave
                st_flux /= h*c/st_wav
            elif spec_units == 'photons' and verbose:
                print('Spectrum units not converted; already in photons.')
        else:
            raise ValueError(f'Acceptable spec_units arguments are {acceptable_spec_units}.')

        dlam = np.diff(wave)
        st_dlam = np.diff(st_wav)

        wave_cut = ((wave > lower_filter_edge) & (wave < upper_filter_edge))[1:]
        st_wave_cut = ((st_wav > lower_filter_edge) & (st_wav < upper_filter_edge))[1:]

        F = np.sum((wave[1:]*flux[1:]*dlam)[wave_cut])
        Fref = np.sum((st_wav[1:]*st_flux[1:]*st_dlam)[st_wave_cut])
        var_F = np.sum((wave[1:]**2*var[1:]**2*dlam**2)[wave_cut])

        mag = -2.5*np.log10(F/Fref)
        emag = np.sqrt((1.09/F)**2*var_F)
    
    return mag, emag

def band_flux(wave,flux,var,sys=None,standard='vega',spec_units='ergs',verbose=False):
    """
    Retrieve flux in all bands for Bessell, SDSS, SNfactory top hat, or LSST filters.
    Filters are provided in photon counts, so this code converts ergs to photons
    for the input flux.
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

    acceptable_spec_units = ['ergs','photons']
    if sys is None:
        raise ValueError('Must specify filter system.')

    h = 6.626E-27 # Planck constant, erg s
    c = 3E18 # Speed of light, AA/s
    F = {}
    eF = {}

    if standard=='vega':
        st_wav, st_flux, st_flux_error = load_vegaspec()
        responsefunc, zeropoints = load_filtersys(sys)
        filters = [x for x in responsefunc.keys()]

        if spec_units in acceptable_spec_units:
            if spec_units == 'ergs':
                if verbose:
                    print('Spectrum units ergs/cm^2/s/AA. Converting to photons.')
                flux /= h*c/wave
                st_flux /= h*c/st_wav
            elif spec_units == 'photons' and verbose:
                print('Spectrum units not converted; already in photons.')
        else:
            raise ValueError(f'Acceptable spec_units arguments are {acceptable_spec_units}.')

        F_ = 0
        Fref = 0
        var_F = 0
        dlam = np.diff(wave)
        st_dlam = np.diff(st_wav)

        for band in filters:
            responsefunc_interp = interp1d(responsefunc[band]['wavelength'], responsefunc[band]['transmission'],kind='linear',bounds_error=False)

            F_ = np.sum(flux[1:]*np.nan_to_num(responsefunc_interp(wave)[1:])*wave[1:]*dlam)
            var_F = np.sum(wave[1:]**2*var[1:]*np.nan_to_num(responsefunc_interp(wave)[1:])**2*dlam**2)
            Fref = np.sum(st_flux[1:]*np.nan_to_num(responsefunc_interp(st_wav)[1:])*st_wav[1:]*st_dlam)

            F[band] = F_/Fref
            eF[band] = np.sqrt(var_F)

    return F, eF

def synth_lc(wave,flux,var,sys=None,standard='vega',spec_units='ergs',verbose=False,return_flux=False):
    """
    Make a synthetic magnitude from a spectrum using Bessell, SDSS, SNfactory top hat, or LSST filters. 
    Filters are provided in photon counts, so this code converts ergs to photons
    for the input flux. 
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

    flux, eflux = band_flux(wave,flux,var,sys=sys,standard=standard,spec_units=spec_units,verbose=verbose)
    photometry = {}
    ephotometry = {}

    for band in flux.keys():
        photometry[band] = -2.5*np.log10(flux[band])
        ephotometry[band] = np.sqrt((1.09/flux[band])**2*eflux[band]**2)
                
    if return_flux:
        return photometry, ephotometry, flux, eflux
    else:
        return photometry, ephotometry