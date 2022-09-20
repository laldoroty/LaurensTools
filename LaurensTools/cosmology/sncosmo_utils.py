import math
import numpy as np
from astropy.coordinates import SkyCoord
from astropy import units as u

"""
These functions are copied and pasted from
sncosmo.
"""
def radec_to_xyz(ra, dec):
    # Modified to add the try/except statement
    try:
        x = math.cos(np.deg2rad(dec)) * math.cos(np.deg2rad(ra))
        y = math.cos(np.deg2rad(dec)) * math.sin(np.deg2rad(ra))
        z = math.sin(np.deg2rad(dec))
    except:
        coord = SkyCoord('%s %s' % (ra, dec), unit=(u.hourangle,u.deg))
        x = math.cos(np.deg2rad(coord.dec.degree)) * math.cos(np.deg2rad(coord.ra.degree))
        y = math.cos(np.deg2rad(coord.dec.degree)) * math.sin(np.deg2rad(coord.ra.degree))
        z = math.sin(np.deg2rad(coord.dec.degree))

    return np.array([x, y, z], dtype=np.float64)

def cmb_dz(ra, dec):
    # See http://arxiv.org/pdf/astro-ph/9609034
    CMBcoordsRA = 167.98750000 # J2000 
    CMBcoordsDEC = -7.22000000

    # J2000 coords from NED\n",
    CMB_DZ = 371000. / 299792458.
    CMB_RA = 168.01190437
    CMB_DEC = -6.98296811
    CMB_XYZ = radec_to_xyz(CMB_RA, CMB_DEC)
    coords_xyz = radec_to_xyz(ra, dec)
    dz = CMB_DZ * np.dot(CMB_XYZ, coords_xyz)

    return dz

def helio_to_cmb(z, ra, dec):
    # Convert from heliocentric redshift to CMB-frame redshift.
    "    Parameters\n",
    "    ----------\n",
    "    z : float\n",
    "        Heliocentric redshift.\n",
    "    ra, dec: float\n",
    "        RA and Declination in degrees (J2000).\n",
    "    \"\"\"\n",

    dz = -cmb_dz(ra, dec)
    one_plus_z_pec = math.sqrt((1. + dz) / (1. - dz))
    one_plus_z_CMB = (1. + z) / one_plus_z_pec

    return one_plus_z_CMB - 1.