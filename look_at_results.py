
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



prefix, target = '', 'A267'
# outfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'mtlz_{}_{}_full.fits'.format(prefix,target)))
# prefix, target = '_M2FS18', 'A20'
corval = 0.3

def compare_to_sdss(maskname,prefixname, corcut = 0.3):
    # outfile = os.path.abspath(os.path.join(os.curdir,'..','data','catalogs','merged_target_lists', 'mtlz{}_{}_full.fits'.format(prefixname,maskname)))
    outfile = os.path.abspath(os.path.join(os.curdir,'..','..','OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'mtlz{}_{}_full.fits'.format(prefixname,maskname)))

    complete_table = Table.read(outfile,format='fits')

    ra_clust = float(complete_table.meta['RA_TARG'])
    dec_clust= float(complete_table.meta['DEC_TARG'])
    z_clust  = float(complete_table.meta['Z_TARG'])
    kpc_p_amin = Planck13.kpc_comoving_per_arcmin(z_clust)
    cluster = SkyCoord(ra=ra_clust*u.deg,dec=dec_clust*u.deg)
    scalar_c = consts.c.to(u.km/u.s).value


    ################################################### A23 HACK ############################
    # reds = Table.read('../data/catalogs/merged_target_lists/A23_Abell1451_Valtchanov2002.fit')
    # # reds = reds[reds['Flag']==5]
    # red_locs = SkyCoord(reds['RAJ2000'],reds['DEJ2000'],unit=[u.hourangle,u.deg])
    #
    # my_red_locs = SkyCoord(complete_table['RA'],complete_table['DEC'],unit=[u.deg,u.deg])
    #
    # complete_table.add_column(Table.Column(data=np.zeros(len(complete_table)).astype(bool),name='Matched'))
    # complete_table.add_column(Table.Column(data=np.zeros(len(complete_table)),name='sdss_zsp'))
    #
    # for ii,loc in enumerate(my_red_locs):
    #     seps = loc.separation(red_locs)
    #     if np.any(seps < 1.5*u.arcsec):
    #         complete_table['Matched'][ii] = True
    #         complete_table['sdss_zsp'][ii] = reds['z'][np.argmin(seps)]
    #
    # measured_table = complete_table
    # spec_overlap = measured_table[measured_table['Matched']]
    ################################################### End A23 HACK ############################

    if 'SDSS_only' not in complete_table.colnames or 'sdss_zsp' not in complete_table.colnames:
        return
    measured_table = complete_table[np.bitwise_not(complete_table['SDSS_only'])]
    if len(measured_table)==0:
        return
    if type(measured_table['sdss_zsp']) is Table.MaskedColumn:
        spec_overlap = measured_table[np.logical_not(measured_table['sdss_zsp'].mask)]
    else:
        notnans = np.bitwise_not(np.isnan(measured_table['sdss_zsp']))
        if np.sum(notnans)>0:
            spec_overlap = measured_table[notnans]
        else:
            spec_overlap = None

    spec_overlap = spec_overlap[((spec_overlap['cor']>corcut))]

    if spec_overlap is None or len(spec_overlap)==0:
        print("There were no overlapping spectra and SDSS")
        return

    blue_cam = np.array([fib[0]=='b' for fib in spec_overlap['FIBNAME']])
    red_cam = np.bitwise_not(blue_cam)

    plt.figure()
    plt.title('Redshifts with cor > {}'.format(corcut),fontsize=14)
    plt.hist(spec_overlap['z_est_bary'],bins=300); plt.xlim(0.18,0.32)
    ymin,ymax = plt.ylim()
    plt.vlines(.23,ymin,ymax,'k','--')

    deviations = scalar_c * (spec_overlap['z_est_bary']-spec_overlap['sdss_zsp'])/(1+z_clust)
    #spec_overlap['velocity']
    #scalar_c*((spec_overlap['z_est_bary']/(1+spec_overlap['z_est_bary']))-(spec_overlap['Published_Vlos']/(1+spec_overlap['Published_Vlos'])))
    meandev,meddev = np.mean(deviations),np.median(deviations)



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


def plot_results(maskname,prefixname, corcut = 0.3):
    # outfile = os.path.abspath(os.path.join(os.curdir,'..','data','catalogs','merged_target_lists', 'mtlz_{}_{}_full.fits'.format(prefixname,maskname)))
    outfile = os.path.abspath(os.path.join(os.curdir,'..','..','OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'mtlz{}_{}_full.fits'.format(prefixname,maskname)))

    complete_table = Table.read(outfile, format='fits')

    ra_clust = float(complete_table.meta['RA_TARG'])
    dec_clust = float(complete_table.meta['DEC_TARG'])
    z_clust = float(complete_table.meta['Z_TARG'])
    kpc_p_amin = Planck13.kpc_comoving_per_arcmin(z_clust)
    cluster = SkyCoord(ra=ra_clust * u.deg, dec=dec_clust * u.deg)
    scalar_c = consts.c.to(u.km / u.s).value

    measured_table = complete_table[np.bitwise_not(complete_table['SDSS_only'])]

    spec_overlap = measured_table  # [np.logical_not(measured_table['sdss_zsp'].mask)]
    spec_overlap = spec_overlap[((spec_overlap['cor'] > corcut))]

    blue_cam = np.array([fib[0] == 'b' for fib in spec_overlap['FIBNAME']])
    red_cam = np.bitwise_not(blue_cam)

    plt.figure()
    plt.title('Redshifts with cor > {}'.format(corcut), fontsize=14)
    plt.hist(spec_overlap['z_est_bary'], bins=300);
    plt.xlim(0.18, 0.32)
    ymin, ymax = plt.ylim()
    plt.vlines(.23, ymin, ymax, 'k', '--')

    plt.figure()
    plt.plot(measured_table['Proj_R_Comoving_Mpc'], measured_table['velocity'], 'b.')
    # plt.plot(measured_table['Proj_R_Comoving_Mpc'],-1*measured_table['velocity'],'b.')
    plt.xlabel("Projected Radius [Comoving Mpc]")
    plt.ylabel('dv [km/s]')
    xmin, xmax = plt.xlim()
    ymin, ymax = -5000, 5000
    plt.ylim(ymin, ymax)
    plt.hlines(0., xmin, xmax, 'k', '--', label='zero')

    plt.figure()
    plt.hist(measured_table['z_est_bary'], bins=300);
    plt.title("{} hist".format(target), fontsize=18)
    plt.xlabel("Bary Redshift", fontsize=18)
    ymin, ymax = plt.ylim()
    plt.vlines(z_clust, ymin, ymax, 'k', '--')

    plt.show()



if __name__ == '__main__':
    compare_to_sdss(maskname=target,prefixname=prefix, corcut = corval)
    plot_results(maskname=target,prefixname=prefix, corcut = corval)