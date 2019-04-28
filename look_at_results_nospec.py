
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack, join
from astropy.io import fits

from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.constants as consts
from astropy.cosmology import Planck13
from astroquery.vizier import Vizier
from astropy.constants import c as speed_of_light



# prefix, target = 'M2FS16', 'A04'
# outfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'mtlz_{}_{}_full.fits'.format(prefix,target)))
prefix, target = 'M2FS18', 'A23'
