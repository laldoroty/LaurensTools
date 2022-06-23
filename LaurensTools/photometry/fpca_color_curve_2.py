import numpy as np
import os 
import os.path as pa
from astropy.table import Table
from scipy.interpolate import splrep, splev

def cc_to_lc(phase, color):

    """
    LNA 20220623
    Converts a color curve to a light curve using results from Burns et al. 2014 and FPCA templates from He et al. 2018. 
    Only works for B-band right now. 
    phase -- Days corresponding to color. Must be with respect to days since B-band maximum.
    color -- B-V color.

    """
    
    # Make a spline to the color and find tmax: 
    spl = splrep(phase,color)
    x = np.linspace(min(phase),max(phase),1000)
    fitcurve = splev(x, spl)
    tmax = x[np.argmax(fitcurve)]

    # Eqn 3 from Burns et al. 2014, rearranged: 
    dm15 = (28.65 + 13.74*1.1 - tmax)/13.74

    def dm15_to_beta1(dm15):
        """
        Fit with my own data.
        """
        return dm15*5.58625155 - 5.34931309

    CDIR = pa.dirname(pa.abspath(__file__))
    LCfPCA_dir = os.path.join(CDIR, 'LCfPCA_He')
    temp_path = os.path.join(LCfPCA_dir, 'bandSpecific_B.txt')
    template = Table.read(temp_path, format='ascii')

    phaselist = template['phase']   # Phase relative Maximum of this passband
    meanlc = template['mean']
    vec1 = template['FPC1']

    return phaselist, meanlc + vec1*dm15_to_beta1(dm15)