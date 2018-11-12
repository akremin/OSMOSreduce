from astropy import utils
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from astropy.cosmology import Planck15 as plnk
import astropy.constants as consts

reds = Table.read('bothsides_A07.txt',format='ascii.tab')
field = Table.read('kremin_M2FS_16A07_319.7045_0.5603.field',format='ascii.tab')

finaltable = Table(names =['GalaxyName','RA','DEC','zbest','correlation','template','epoch','mag'],dtype=['S6',float,float,float,float,int,float,float])

for row in reds:
    fieldrow = field[field['ID']==row['Galaxy']]
    finaltable.add_row([row['Galaxy'],fieldrow['RA'],fieldrow['DEC'],row['zbest'],row['correlation'],row['template'],fieldrow['EPOCH'],fieldrow['MAG']])
#finaltable.write('A07_matchedtable_radecred.csv',format='ascii.csv')
#finaltable.write('A07_matchedtable_radecred.fits',format='fits')

            


member_coords = SkyCoord(finaltable['RA']*u.deg,finaltable['DEC']*u.deg)

clus_red = 0.276487
clus_cent = SkyCoord(319.70445632*u.deg,0.56034244*u.deg)

member_separations = clus_cent.separation(member_coords)
clus_mpcperasec = (1./plnk.arcsec_per_kpc_comoving(clus_red))*(1*u.Mpc)/(1000*u.kpc)
member_seps_mpc = member_separations.arcsecond*u.arcsec * clus_mpcperasec

clus_mpcperasec = (1./plnk.arcsec_per_kpc_comoving(clus_red))*(1*u.Mpc)/(1000*u.kpc)
member_seps_mpc = member_separations.arcsecond*u.arcsec * clus_mpcperasec

member_velocities = ((finaltable['zbest']- clus_red)/(1+clus_red))*consts.c*(1*u.km)/(1000*u.m)


plt.plot(member_seps_mpc,member_velocities,'b.',markersize=12); plt.ylim(-5000,5000); plt.xlim(0,5); plt.ylabel('LoS Velocity (km/s)',size='x-large'); plt.xlabel(r'r (h${}^{-1}$ Mpc)',size='x-large'); plt.title('\'A07\' SDSS J211849.06+003337.2',size='xx-large'); plt.savefig('phasespace_A07.png',dpi=600); plt.show()
