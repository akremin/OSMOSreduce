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



mask = 'A04'#'A02'
z_clust = 0.2198#0.252
kpc_p_amin = Planck13.kpc_comoving_per_arcmin(z_clust)



#apperature, redshift_est, cor, template
#ID,FIBNAME,sdss_SDSS12,RA,DEC,sdss_zsp,sdss_zph,sdss_rmag,MAG
zfit_path = os.path.abspath(os.path.join(os.curdir,'..', '..','OneDrive - umich.edu','Research','M2FSReductions',mask,'zfits'))
zfits_template = '{cam}_zfits_{maskname}_combined_1d_bcwfs.fits'
zfit_file = os.path.join( zfit_path, zfits_template.format(cam='r',maskname=mask) )
zfit_file1 = os.path.join( zfit_path, zfits_template.format(cam='b',maskname=mask) )

mtlzfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'mtlz_M2FS16_{}_full.csv'.format(mask)))
outfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'allzs_plus_mtlz_M2FS16_{}_full.csv'.format(mask)))

tabler = Table.read(zfit_file,format='fits')
tableb = Table.read(zfit_file1,format='fits')



# headerr = fits.getheader(zfit_file,ext=1)
# headerb = fits.getheader(zfit_file1,ext=1)
mtlz = Table.read(mtlzfile,format='ascii.csv')
tabler.rename_column('apperature','FIBNAME')
tableb.rename_column('apperature','FIBNAME')

fibermap = {}
for key,val in dict(tabler.meta).items():
    if key[:5] == 'FIBER':
        fibermap['{}{}'.format('r',key[5:])] = val.strip(' \t')
for key, val in dict(tableb.meta).items():
    if key[:5] == 'FIBER':
        fibermap['{}{}'.format('b', key[5:])] = val.strip(' \t')
for ii in range(len(mtlz)):
    id = mtlz['ID'][ii]
    fbnm = mtlz['FIBNAME'][ii]
    if fbnm not in fibermap.keys():
        print("{} not in fibermap!".format(fbnm))
    elif fibermap[fbnm].upper() != id:
        print(ii,fbnm,fibermap[fbnm],id)
for t in range(1,9):
    for f in range(1,17):
        testkey = 'FIBER{:d}{:02d}'.format(t,f)
        replacekey = 'r{:d}{:02d}'.format(t, f)
        if testkey in tabler.meta.keys():
            tabler.meta[replacekey] = tabler.meta[testkey]
            tabler.meta.pop(testkey)
for t in range(1,9):
    for f in range(1,17):
        testkey = 'FIBER{:d}{:02d}'.format(t,f)
        replacekey = 'b{:d}{:02d}'.format(t, f)
        if testkey in tableb.meta.keys():
            tableb.meta[replacekey] = tableb.meta[testkey]
            tableb.meta.pop(testkey)

combined_table = vstack([tabler,tableb])
# combined_table = tabler.copy()

full_table = join(combined_table,mtlz,'FIBNAME',join_type='inner')


## Add additional information
time = Time(full_table.meta['MJD'],format='mjd')
location = EarthLocation(lon=full_table.meta['SITELONG']*u.deg, lat=full_table.meta['SITELAT']*u.deg,\
                         height=full_table.meta['SITEALT']*u.meter)
cord = SkyCoord(ra=full_table.meta['RA-D']*u.deg,dec=full_table.meta['DEC-D']*u.deg)
hc_cor = cord.radial_velocity_correction(kind='barycentric',obstime=time,location=location)
dz = hc_cor/consts.c

full_table.add_column(Table.Column(data=full_table['redshift_est']+dz,name='z_est_bary'))


cluster = cord
all_coords = SkyCoord(ra=full_table['RA']*u.deg,dec=full_table['DEC']*u.deg)
seps = cluster.separation(all_coords)
full_table.add_column(Table.Column(data=seps.to(u.arcsec).value,name='Proj_R_asec'))
full_table.add_column(Table.Column(data=(kpc_p_amin*seps).to(u.Mpc).value,name='Proj_R_Comoving_Mpc'))


corcut = 0.4
spec_overlap = full_table[np.logical_not(full_table['sdss_zsp'].mask)]
spec_overlap = spec_overlap[((spec_overlap['cor']>corcut))]
deviations = spec_overlap['z_est_bary']-spec_overlap['sdss_zsp']
meandev,meddev = np.mean(deviations),np.median(deviations)

dvs = consts.c.to(u.km/u.s).value*(z_clust-full_table['z_est_bary'])/(1.+z_clust)
full_table.add_column(Table.Column(data=dvs,name='velocity'))


blue_cam = np.array([fib[0]=='b' for fib in spec_overlap['FIBNAME']])
red_cam = np.bitwise_not(blue_cam)

plt.figure()
plt.hist(full_table['z_est_bary'],bins=300);
plt.title("A02 hist",fontsize=18)
plt.xlabel("Bary Redshift",fontsize=18)
ymin,ymax = plt.ylim()
plt.vlines(.252,ymin,ymax,'k','--')


plt.figure()
plt.plot(spec_overlap['z_est_bary'],spec_overlap['sdss_zsp'],'b.')
plt.plot([0,1],[0,1],'k--')
plt.title("A02",fontsize=18)
plt.xlabel('z_est_bary',fontsize=18)
plt.ylabel('sdss_zsp',fontsize=18)

plt.figure()
plt.plot(spec_overlap['z_est_bary'][blue_cam],deviations[blue_cam]*consts.c.to(u.km/u.s).value,'b.',markersize=18)
plt.plot(spec_overlap['z_est_bary'][red_cam],deviations[red_cam]*consts.c.to(u.km/u.s).value,'r.',markersize=18)
plt.title("A02 median={} mean={}".format(meddev,meandev),fontsize=18)
plt.xlabel("Bary Redshift",fontsize=18)
plt.ylabel("c*dz/(1+zclust)",fontsize=18)
xmin,xmax = plt.xlim()
plt.hlines(0.,xmin,xmax,'k','--',label='zero')
plt.legend(loc='best');

plt.figure(); plt.hist(deviations,bins=20)
plt.title("A02 hist",fontsize=18)
plt.xlabel("This Work - SDSS",fontsize=18)
plt.vlines(meandev,0,4,colors='k',label='mean')
plt.vlines(meddev,0,4,colors='g',label='median')
plt.legend(loc='best')




vizier_catalogs = ['sdss12']
radius = 5*u.Mpc / kpc_p_amin
Vizier.ROW_LIMIT = -1
result = Vizier.query_region(cluster, radius=radius, catalog=vizier_catalogs)
res_tab = result[0]

if np.all(res_tab['zsp'].mask):
    if np.all(res_tab['zph'].mask):
        sdss_archive_table = res_tab
    else:
        cut_tab = res_tab[np.where(~res_tab['zph'].mask)]
        sdss_archive_table = cut_tab[np.where(cut_tab['zph'] > -99)]
else:
    sdss_archive_table = res_tab[np.where(~res_tab['zsp'].mask)]



for col in sdss_archive_table.colnames:
    sdss_archive_table.rename_column(col, 'sdss_' + col)

for ii in np.arange(len(sdss_archive_table))[::-1]:
    if sdss_archive_table['sdss_SDSS12'][ii] in full_table['sdss_SDSS12']:
        print("Removing: {}".format(sdss_archive_table['sdss_SDSS12'][ii]))
        sdss_archive_table.remove_row(ii)

sdss_archive_table.add_column(Table.Column(data=sdss_archive_table['sdss_RA_ICRS'],name='RA'))
sdss_archive_table.add_column(Table.Column(data=sdss_archive_table['sdss_DE_ICRS'],name='DEC'))
sdss_archive_table.add_column(Table.Column(data=['T']*len(sdss_archive_table),name='TYPE'))
sdss_archive_table.add_column(Table.Column(data=[2000.0]*len(sdss_archive_table),name='EPOCH'))

all_sdss_coords = SkyCoord(ra=sdss_archive_table['sdss_RA_ICRS'],dec=sdss_archive_table['sdss_DE_ICRS'])
seps = cluster.separation(all_sdss_coords)
sdss_archive_table.add_column(Table.Column(data=seps.to(u.arcsec).value,name='Proj_R_asec'))
sdss_archive_table.add_column(Table.Column(data=(kpc_p_amin*seps).to(u.Mpc).value,name='Proj_R_Comoving_Mpc'))
sdss_archive_table.add_column(Table.Column(data=sdss_archive_table['sdss_zsp'],name='z_est_bary'))
dvs = consts.c.to(u.km/u.s).value*(z_clust-sdss_archive_table['z_est_bary'])/(1.+z_clust)
sdss_archive_table.add_column(Table.Column(data=dvs,name='velocity'))
#
#
full_table.add_column(Table.Column(data=[False]*len(full_table),name='SDSS_only'))
sdss_archive_table.add_column(Table.Column(data=[True]*len(sdss_archive_table),name='SDSS_only'))


sdss_archive_table.convert_bytestring_to_unicode()
convert = []
for row in sdss_archive_table['sdss_q_mode']:
    convert.append(float(row.strip(' ')=='+'))

new_sdssq_col = Table.Column(data=convert,name='sdss_q_mode')
sdss_archive_table.replace_column('sdss_q_mode',new_sdssq_col)

mega_table = vstack([full_table,sdss_archive_table])

mega_table.write(outfile.replace('.csv','.fits'),format='fits')

mega_table.meta['comments'] = []
mega_table.write(outfile,format='ascii.csv')


plt.show()