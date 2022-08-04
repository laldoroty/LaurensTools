import numpy as np

def mag_err_to_flux_err(Fref,mref,m,em):
    """
    Convert magnitude error to flux error.
    Useful for sncosmo.
    Fref -- Reference flux
        You can do MAGSYS.zpbandflux('band') in sncosmo for this.
    mref -- reference magnitude
        You can do MAGSYS.band_flux_to_mag(MAGSYS.zpbandflux('band'),'band') for this.
    m -- data magnitude
    em -- data magnitude error
    """
    return Fref*np.power(10,(mref-m)/2.5)*np.log(10)*em

class constants:
    """
    Access constants easily. 
    e.g.:
    from LaurensTools.utils.constants_and_conversions import constants
    const = constants()
    print(const.h)
    """
    def __init__(self):
        self.h = 6.626E-27 # Planck constant, erg s
        self.c = 3E10 # Speed of light, cm/s

class convert_units:
    """
    Convert units easily.
    """
    def __init__(self,val_in,current_units,new_units):
        units = ['pc','m']
        if current_units not in units or new_units not in units:
            raise ValueError(f'Supported units are {units}.')