
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


ra_clust,dec_clust,z_clust = 240.8291, 3.2790,0.2198#210.25864,2.87847,0.252
kpc_p_amin = Planck13.kpc_comoving_per_arcmin(z_clust)
cluster = SkyCoord(ra=ra_clust*u.deg,dec=dec_clust*u.deg)
scalar_c = consts.c.to(u.km/u.s).value


prefix, target = 'M2FS16', 'A04'
outfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'allzs_plus_mtlz_{}_{}_full.fits'.format(prefix,target)))


complete_table = Table.read(outfile,format='fits')

measured_table = complete_table[np.bitwise_not(complete_table['SDSS_only'])]

corcut = 0.3

spec_overlap = measured_table[np.logical_not(measured_table['sdss_zsp'].mask)]
spec_overlap = spec_overlap[((spec_overlap['cor']>corcut))]
if len(spec_overlap)==0:
    raise(TypeError,"There were no overlapping spectra and SDSS")
blue_cam = np.array([fib[0]=='b' for fib in spec_overlap['FIBNAME']])
red_cam = np.bitwise_not(blue_cam)

plt.figure()
plt.title('Redshifts with cor > {}'.format(corcut),fontsize=14)
plt.hist(spec_overlap['z_est_bary'],bins=300); plt.xlim(0.18,0.32)
ymin,ymax = plt.ylim()
plt.vlines(.23,ymin,ymax,'k','--')


# plt.figure()
# plt.title('Redshifts with cor > {} matched'.format(corcut),fontsize=14)
# plt.errorbar(spec_overlap['z_est_bary'][blue_cam],spec_overlap['Published_Vlos'][blue_cam]/scalar_c,fmt='b.',yerr=spec_overlap['Published_Vlos_e'][blue_cam]/scalar_c)#,markercolor='b',markerstyle='.')
# plt.errorbar(spec_overlap['z_est_bary'][red_cam],spec_overlap['Published_Vlos'][red_cam]/scalar_c,fmt='r.',yerr=spec_overlap['Published_Vlos_e'][red_cam]/scalar_c)#,markercolor='b',markerstyle='.')
# plt.xlabel("Barycentric Redshift",fontsize=14)
# plt.ylabel("Barycentric Redshift",fontsize=14)
# plt.plot([0,.5],[0,.5],'k--')


deviations = scalar_c * (spec_overlap['z_est_bary']-spec_overlap['sdss_zsp'])/(1+z_clust)
    #spec_overlap['velocity']
    #scalar_c*((spec_overlap['z_est_bary']/(1+spec_overlap['z_est_bary']))-(spec_overlap['Published_Vlos']/(1+spec_overlap['Published_Vlos'])))
meandev,meddev = np.mean(deviations),np.median(deviations)


# plt.figure()
# plt.title('Redshifts with cor > {} matched'.format(corcut),fontsize=14)
# plt.errorbar(spec_overlap['z_est_bary'][blue_cam],deviations[blue_cam],fmt='b.',xerr=spec_overlap['Published_Vlos_e'][blue_cam]/scalar_c,yerr=spec_overlap['Published_Vlos_e'][blue_cam])
# plt.errorbar(spec_overlap['z_est_bary'][red_cam],deviations[red_cam],fmt='r.',xerr=spec_overlap['Published_Vlos_e'][red_cam]/scalar_c,yerr=spec_overlap['Published_Vlos_e'][red_cam])
# plt.xlabel('Barycentric Redshift',fontsize=14)
# plt.ylabel("This work - Published  [km/s]",fontsize=14)
# xmin,xmax = plt.xlim()
# plt.hlines(0.,xmin,xmax,'k','--',label='zero')
# plt.hlines(meddev,xmin,xmax,'g','--',label='median')
# plt.hlines(meandev,xmin,xmax,'c','--',label='mean')
# plt.legend(loc='best')


plt.figure()
plt.title('{} Redshifts with cor > {} matched'.format(target,corcut),fontsize=14)
plt.xlabel("This work - Published  [km/s]",fontsize=14)
plt.ylabel("Number Counts",fontsize=14)
plt.hist(deviations,bins=20)
ymin,ymax = plt.ylim()
plt.vlines(meandev,0,ymax,colors='k',label='mean')
plt.vlines(meddev,0,ymax,colors='g',label='median')
plt.legend(loc='best')



plt.figure()
plt.plot(measured_table['Proj_R_Comoving_Mpc'],measured_table['velocity'],'b.')
# plt.plot(measured_table['Proj_R_Comoving_Mpc'],-1*measured_table['velocity'],'b.')
plt.xlabel("Projected Radius [Comoving Mpc]")
plt.ylabel('dv [km/s]')
xmin,xmax = plt.xlim()
ymin,ymax = -5000,5000
plt.ylim(ymin,ymax)
plt.hlines(0.,xmin,xmax,'k','--',label='zero')



plt.figure()
plt.hist(measured_table['z_est_bary'],bins=300);
plt.title("{} hist".format(target),fontsize=18)
plt.xlabel("Bary Redshift",fontsize=18)
ymin,ymax = plt.ylim()
plt.vlines(z_clust,ymin,ymax,'k','--')


plt.figure()
plt.plot(spec_overlap['z_est_bary'],spec_overlap['sdss_zsp'],'b.')
plt.plot([0,1],[0,1],'k--')
plt.title("{}".format(target),fontsize=18)
plt.xlabel('z_est_bary',fontsize=18)
plt.ylabel('sdss_zsp',fontsize=18)

plt.figure()
plt.plot(spec_overlap['z_est_bary'][blue_cam],deviations[blue_cam],'b.',markersize=18)
plt.plot(spec_overlap['z_est_bary'][red_cam],deviations[red_cam],'r.',markersize=18)
plt.title("{} median={} mean={}".format(target,meddev,meandev),fontsize=18)
plt.xlabel("Bary Redshift",fontsize=18)
plt.ylabel("c*dz/(1+zclust) [km/s]",fontsize=18)
xmin,xmax = plt.xlim()
plt.hlines(0.,xmin,xmax,'k','--',label='zero')
plt.legend(loc='best');

plt.figure(); plt.hist(deviations,bins=20)
plt.title("{} hist".format(target),fontsize=18)
plt.xlabel("This Work - SDSS",fontsize=18)
ymin,ymax = plt.ylim()
plt.vlines(meandev,ymin,ymax,colors='k',label='mean')
plt.vlines(meddev,ymin,ymax,colors='g',label='median')
plt.legend(loc='best')


plt.show()