import pycmpfit
import os
import sys
import numpy as np
import os.path as pa
from astropy.table import Table
from scipy.interpolate import interp1d

# print(pa.abspath(pycmpfit.__file__))
"""

"""

def get_template(band_a,band_b):
    CDIR = pa.dirname(pa.abspath(__file__))
    LCfPCA_dir = os.path.join(CDIR, 'LCfPCA_He')
    # r_vals = [4.1, 3.1, 2.33, 1.48]
    bandlist = ['B', 'V', 'R', 'I']

    if band_a.upper() in bandlist or band_b.upper() in bandlist:
        pctemp = f'/colormodel_{band_a.upper()}{band_b.upper()}.txt'
        # r = r_vals[bandlist.index(band.upper())]
    else:
        raise ValueError('Only B-V, B-R, and B-I colors are available.')

    fPCAfile = LCfPCA_dir + pctemp
    return fPCAfile

def FITTER(PHA_phot, color_phot, ecolor_phot, band_a=None, band_b=None):
    """
    LNA 20220623

    Fits a color curve using fPCA templates from He et al. 2018.
    
    PHA_phot -- Phase of input photometry data.
    color_phot -- B-V color for input data.
    ecolor_phot -- error for the B-V color.
    band_a -- First bandpass, usually B.
    band_b -- Second bandpass, usually V.
    """
    if band_a or band_b is None:
        end = 40.0
    elif band_a and band_b in ['B','V','R','I']:
        end = 50.0
    else:
        raise ValueError('Values for band_a and band_b can be None, B, V, R, or I.')
    
    # if PHA_grid is None:
    #     PHA_grid = np.arange(-10.0,end,0.1)

    # Get the templates for band_a and band_b:
    AstBasis_a = Table.read(get_template(band_a,band_b), format='ascii')

    # Make Interpolation Models for band_a
    phaselist = AstBasis_a['phase']   # Phase relative Maximum of this passband
    BsVec0_a = AstBasis_a['mean']
    BsVec1_a = AstBasis_a['FPC1']
    BsVec2_a = AstBasis_a['FPC2']
    BsVec3_a = AstBasis_a['FPC3']
    BsVec4_a = AstBasis_a['FPC4']

    imod0_a = interp1d(phaselist, BsVec0_a, fill_value='extrapolate')
    imod1_a = interp1d(phaselist, BsVec1_a, fill_value='extrapolate')
    imod2_a = interp1d(phaselist, BsVec2_a, fill_value='extrapolate')
    imod3_a = interp1d(phaselist, BsVec3_a, fill_value='extrapolate')
    imod4_a = interp1d(phaselist, BsVec4_a, fill_value='extrapolate')

    # m: Number of samples [len(X_obs)]
    # n: Number of parameters of the function [len(theta)]

    def modelcurve(phase, theta):
        a0, b0 = 1.0, 1.0
        tof, mof, a1, a2, a3, a4 = theta
        mphase = phase-tof
        # Note fPCA effective mphase [-9.0, 40.0]  # WARNING: you may redefine this !
        if mphase >= -10.0 and mphase <= end:
            VecP0_a = float(imod0_a(mphase))
            VecP1_a = float(imod1_a(mphase))
            VecP2_a = float(imod2_a(mphase))
            VecP3_a = float(imod3_a(mphase))
            VecP4_a = float(imod4_a(mphase))

            y = mof + a0*VecP0_a + a1*VecP1_a + a2*VecP2_a + a3*VecP3_a + a4*VecP4_a 
        else:
            y = np.nan
        return y

    def userfunc(m, n, theta, private_data):
        X, Y, eY = private_data["x"], private_data["y"], private_data["ey"]
        devs = np.zeros((m), dtype=np.float64)
        user_dict = {"deviates": None}
        for i in range(m):
            y_mod = modelcurve(X[i], theta)
            if not np.isnan(y_mod):
                devs[i] = (Y[i] - y_mod) / eY[i]
            else:
                devs[i] = 0.0  # deweight data-points out of effective mphase !
        user_dict["deviates"] = devs
        return user_dict

    def LSFit(X_obs, Y_obs, eY_obs, theta):
        m, n = len(X_obs), len(theta)
        user_data = {"x": X_obs, "y": Y_obs, "ey": eY_obs}
        py_mp_par = list(pycmpfit.MpPar() for i in range(6))
        py_mp_par[0].limited[0] = 1
        py_mp_par[0].limited[1] = 1
        py_mp_par[0].limits[0] = -10.0
        py_mp_par[0].limits[1] = 10.0
        # Use only the first two PC components for each color:
        # py_mp_par[5].fixed = 1
        # py_mp_par[4].fixed = 1
        # py_mp_par[6].fixed = 0
        # py_mp_par[7].fixed = 0
        # py_mp_par[8].fixed = 1
        # py_mp_par[9].fixed = 1

        fit = pycmpfit.Mpfit(userfunc, m, theta,
                             private_data=user_data, py_mp_par=py_mp_par)
        fit.mpfit()  # NOTE now theta has been updated
        mp_result = fit.result
        return mp_result, theta

    theta = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])   # trivial initial guess
    res = LSFit(X_obs=PHA_phot, Y_obs=color_phot, eY_obs=ecolor_phot, theta=theta)
    theta_fin = res[1]
    errors = res[0].xerror
    covariance_matrix = res[0].covar

    color_grid = [modelcurve(phase + theta_fin[0], theta_fin)
                for phase in phaselist]
    days = phaselist.data.tolist()

    lc_interp = interp1d(phaselist,color_grid)

    def e_cc():
    # Calculates the error for the entire color curve. 
        dq0 = 0
        all_the_jacobians = np.column_stack(
            (np.zeros(len(phaselist)), np.ones(len(phaselist)), BsVec1_a, BsVec2_a, BsVec3_a, BsVec4_a))
        cc_error = []
        for jac in all_the_jacobians:
            j = np.sqrt(np.matmul(jac,np.matmul(covariance_matrix,np.transpose(jac))))
            cc_error.append(j)
        return np.array(cc_error)

    cc_error = e_cc()

    # def return_model_mag(phase, theta):
    #     a0, b0 = 1.0, 1.0
    #     tof, mof, a1, a2, a3, a4, b1, b2, b3, b4 = theta
    #     VecP0_a = (imod0_a(phaselist))
    #     VecP1_a = (imod1_a(phaselist))
    #     VecP2_a = (imod2_a(phaselist))
    #     VecP3_a = (imod3_a(phaselist))
    #     VecP4_a = (imod4_a(phaselist))

    #     VecP0_b = (imod0_b(phaselist))
    #     VecP1_b = (imod1_b(phaselist))
    #     VecP2_b = (imod2_b(phaselist))
    #     VecP3_b = (imod3_b(phaselist))
    #     VecP4_b = (imod4_b(phaselist))
    #     y = mof + a0*VecP0_a + a1*VecP1_a + a2*VecP2_a + a3*VecP3_a + a4*VecP4_a 
    #     + b0*VecP0_b + b1*VecP1_b + b2*VecP2_b + b3*VecP3_b + b4*VecP4_b
    #     return y


    return color_grid, phaselist, theta_fin, errors, covariance_matrix, cc_error
