
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


ra_clust,dec_clust,z_clust = 210.25864,2.87847,0.252
kpc_p_amin = Planck13.kpc_comoving_per_arcmin(z_clust)
cluster = SkyCoord(ra=ra_clust*u.deg,dec=dec_clust*u.deg)


outfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'allzs_plus_mtlz_M2FS16_A02_full.fits'))







mega_table = Table.read(outfile,format='fits')

plt.figure()
plt.plot(mega_table['Proj_R_Comoving_Mpc'],mega_table['velocity'],'b.')
# plt.plot(mega_table['Proj_R_Comoving_Mpc'],-1*mega_table['velocity'],'b.')
plt.xlabel("Projected Radius [Comoving Mpc]")
plt.ylabel('dv [km/s]')
xmin,xmax = plt.xlim()
ymin,ymax = -5000,5000
plt.ylim(ymin,ymax)
plt.hlines(0.,xmin,xmax,'k','--',label='zero')
plt.show()