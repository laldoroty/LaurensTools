import numpy as np
import numpy.polynomial.polynomial as poly
import pandas as pd

grid = np.arange(-20.0,51.0,1)
M0 = pd.read_csv('salt3-k21-templates/salt3_template_0.dat', sep=' ', names=['epoch','wavelength','flux'])
varM0 = pd.read_csv('salt3-k21-templates/salt3_lc_variance_0.dat', sep=' ', names=['epoch','wavelength','vflux'])

M1 = pd.read_csv('salt3-k21-templates/salt3_template_1.dat', sep=' ', names=['epoch','wavelength','flux'])
varM1 = pd.read_csv('salt3-k21-templates/salt3_lc_variance_1.dat', sep=' ', names=['epoch','wavelength','vflux'])

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

def param_err_to_mag_err(sncosmo_result):
    param_names = sncosmo_result.param_names
    jac = np.empty([len(param_names),len(M0['wavelength'])])
    x0_idx = param_names.index('x0')
    x1_idx = param_names.index('x1')
    c_idx = param_names.index('c')

    dx0 = -2.5/(np.log(10)*sncosmo_result.parameters[x0_idx])
    dx1_coeff = -2.5/(np.log(10)*salt3_modelfunc(M0['wavelength'],sncosmo_result.parameters[x0_idx],sncosmo_result.parameters[x1_idx],sncosmo_result.parameters[c_idx]))
    dx1_func = sncosmo_result.parameters[x0_idx]*M1['flux']*np.exp(sncosmo_result.parameters[c_idx]*CL(M0['wavelength']))
    dx1 = np.array(dx1_coeff*dx1_func)
    dc = np.array(-2.5*CL(M0['wavelength'])/np.log(10))

    jac[x0_idx] = np.full(shape=len(dx1),fill_value=dx0)
    jac[x1_idx] = dx1
    jac[c_idx] = dc

    return jac