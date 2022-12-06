import numpy as np
import numpy.polynomial.polynomial as poly
from scipy.interpolate import interp1d
import pandas as pd
import sncosmo
import matplotlib.pyplot as plt

grid = np.arange(-20.0,51.0,1)
M0 = pd.read_csv('salt3-k21-templates/salt3_template_0.dat', sep=' ', names=['epoch','wavelength','flux'])
varM0 = pd.read_csv('salt3-k21-templates/salt3_lc_variance_0.dat', sep=' ', names=['epoch','wavelength','vflux'])

M1 = pd.read_csv('salt3-k21-templates/salt3_template_1.dat', sep=' ', names=['epoch','wavelength','flux'])
varM1 = pd.read_csv('salt3-k21-templates/salt3_lc_variance_1.dat', sep=' ', names=['epoch','wavelength','vflux'])

covM0M1 = pd.read_csv('salt3-k21-templates/salt3_lc_covariance_01.dat', sep=' ', names=['epoch','wavelength','vflux'])

coeffsfile = open('salt3-k21-templates/salt3_color_correction.dat')
SALT2CL_B = 4300 # central B-band wavelength
SALT2CL_V = 5430 # central V-band wavelength

l_lo = (2800 - SALT2CL_B)/(SALT2CL_V-SALT2CL_B)
l_hi = (8000 - SALT2CL_B)/(SALT2CL_V-SALT2CL_B)

file_coeffs=[]
for line in coeffsfile.readlines():
    try:
        file_coeffs.append(float(line))
    except:
        pass
        
coeffs = [0, 1-sum(file_coeffs)]
for c in file_coeffs:
    coeffs.append(c)

def CLp(lam):
    l = (lam-SALT2CL_B)/(SALT2CL_V-SALT2CL_B)
    return poly.polyval(l, coeffs)

def dCLp(lam):
    l = (lam-SALT2CL_B)/(SALT2CL_V-SALT2CL_B)
    dcoeffs = poly.polyder(coeffs)
    return poly.polyval(l, dcoeffs)

def CL(lam):
    colorlaw = np.empty(len(lam))
    for i in range(len(lam)):
        l = (lam[i]-SALT2CL_B)/(SALT2CL_V-SALT2CL_B)
        if l > l_lo and l < l_hi:
            colorlaw[i] = -float(CLp(lam[i]))
        elif l < l_lo:
            colorlaw[i] = -float(dCLp(l_lo)*(l-l_lo) + CLp(l_lo))
        elif l > l_hi:
            colorlaw[i] = -float(dCLp(l_hi)*(l-l_hi) + CLp(l_hi))
    return colorlaw

def salt3_modelfunc(lam,x0,x1,c):
    return x0*(M0['flux'] + x1*M1['flux'])*np.exp(c*CL(lam))

def param_err_to_mag_err(sncosmo_result,magsys):
    """
    Converts SALT3 parameter error to magnitude error. 
    sncosmo_result -- output from ``result, fitted_model = sncosmo.fit_lc()'',
        where model =sncosmo.Model(source='salt3').
    magsys -- set magsystem from sncosmo. e.g., sncosmo.get_magsystem('vega').
    
    """
    param_names = sncosmo_result.param_names
    vparam_names = sncosmo_result.vparam_names

    # Identify location of unused parameters
    # (i.e. ones that you didn't fit in sncosmo)
    missing_idx = None
    if param_names == vparam_names:
        pass
    else:
        missing_idx = []
        for i, par in enumerate(param_names):
            if par not in vparam_names:
                missing_idx.append(i)

    # Build your jacobian matrix. 
    cov = sncosmo_result.covariance
    jac = np.empty([len(param_names),len(M0['wavelength'])])
    x0_idx = vparam_names.index('x0')
    x1_idx = vparam_names.index('x1')
    c_idx = vparam_names.index('c')

    dx0 = -2.5/(np.log(10)*sncosmo_result.parameters[x0_idx])
    dx1_coeff = -2.5/(np.log(10)*salt3_modelfunc(M0['wavelength'],sncosmo_result.parameters[x0_idx],sncosmo_result.parameters[x1_idx],sncosmo_result.parameters[c_idx]))

    print('dx0')
    print(type(dx0))
    print(dx0)

    dx1_func = sncosmo_result.parameters[x0_idx]*M1['flux']*np.exp(sncosmo_result.parameters[c_idx]*CL(M0['wavelength']))
    dx1 = dx1_coeff*dx1_func
    # print('dx1')
    # print(type(dx1))
    # print(dx1.shape)
    # print('min dx1',min(dx1))
    # print('max dx1', max(dx1))
    # print(dx1)
    # dx1_df = pd.DataFrame({'epoch': M0['epoch'], 'wavelength': 
    #     M0['wavelength'],'dx1': dx1})
    # print(dx1_df)

    dc = np.array(-2.5*CL(M0['wavelength'])/np.log(10))

    jac[x0_idx] = np.full(shape=len(dx1),fill_value=dx0)
    jac[x1_idx] = dx1
    jac[c_idx] = dc

    # print('jac', jac)

    all_the_jacobians = np.transpose(jac)

    # Add a row and column of 0 into the covariance
    # matrix for the fit parameter that was fixed.
    missing_idx_inv = np.array([i for i in range(len(param_names)) if i not in missing_idx])
    modified_cov = np.zeros((len(param_names),len(param_names)), dtype=cov.dtype)
    modified_cov[np.array(missing_idx_inv).reshape(-1,1), np.array(missing_idx_inv)] = cov

    # Matrix multiplication to get the flux error in flux units
    # for the entire model, F(p,\lambda).
    mag_err = []
    for j in all_the_jacobians:
        mag_err.append(np.sqrt(j @ modified_cov @ np.transpose(j)))

    print('min mag err', min(mag_err))
    print('max mag err', max(mag_err))

    mag_err_df = pd.DataFrame({'epoch': M0['epoch'], 'wavelength': 
        M0['wavelength'],'mag_err': mag_err})

    return mag_err_df

def param_err_to_mag_err_2(result,model,band):
    """
    result -- result from ``result, model = sncosmo.fit_lc()''
    model -- model from ``result, model = sncosmo.fit_lc()''
    band -- sncosmo band. Must be registered to sncosmo with:
                filt = sncosmo.Bandpass(wave,flux,name='name',wave_unit=u.AA)
                sncosmo.register(filt, filt.name)
    """
    # Get resulting fit parameters.
    param_names = result.param_names
    z_idx = param_names.index('z')
    x0_idx = param_names.index('x0')
    x1_idx = param_names.index('x1')
    c_idx = param_names.index('c')

    z = result.parameters[z_idx] # Heliocentric redshift
    x0 = result.parameters[x0_idx]
    x1 = result.parameters[x1_idx]
    c = result.parameters[c_idx]

    # Get epochs from model.
    epochs = M0['epoch'].unique()
    modelwave = M0['wavelength'].unique()

    # Get transmission function and integrate.
    sncosmo_bandpass = sncosmo.get_bandpass(band)
    band_wave = sncosmo_bandpass.wave 
    band_trans = sncosmo_bandpass.trans # In photon units, observer frame
    central_wave = sncosmo_bandpass.wave_eff

    # band_wave_deredshifted = band_wave/(1+z) # not gonna lie, not understanding
    #                                         # how to use this in the integral.

    band_integrand = band_trans*band_wave
    trans_integral = np.trapz(band_integrand,band_wave)

    mag_err_list = []
    for epoch in epochs:
        single_varM0 = varM0[varM0['epoch'] == epoch]['vflux']
        single_varM1 = varM1[varM1['epoch'] == epoch]['vflux']
        single_covM0M1 = covM0M1[covM0M1['epoch'] == epoch]['vflux']

        varM0_func = interp1d(modelwave,single_varM0)
        varM1_func = interp1d(modelwave,single_varM1)
        covM0M1_func = interp1d(modelwave,single_covM0M1)

        varM0_central = varM0_func(central_wave)
        varM1_central = varM1_func(central_wave)
        covM0M1_central = covM0M1_func(central_wave)

        var_flux = ((x0*np.exp(CL([central_wave]))*trans_integral)**2) * \
                                    varM0_central + x1*varM1_central + \
                    2*x1*covM0M1_central*np.sqrt(varM0_central*varM1_central)

        mag_err = np.sqrt(((2.5/np.log(10))**2)*(model.bandflux(band,epoch)**2)/var_flux)
        mag_err_list.append(mag_err[0])

    return mag_err_list