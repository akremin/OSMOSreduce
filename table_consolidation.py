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


#apperature, redshift_est, cor, template
#ID,FIBNAME,sdss_SDSS12,RA,DEC,sdss_zsp,sdss_zph,sdss_rmag,MAG
zfit_path = os.path.abspath(os.path.join(os.curdir,'..', '..','OneDrive - umich.edu','Research','M2FSReductions','A02','zfits'))
zfits_template = '{cam}_zfits_{maskname}_combined_1d_bcwfs.fits'
zfit_file = os.path.join( zfit_path, zfits_template.format(cam='r',maskname='A02') )
zfit_file1 = os.path.join( zfit_path, zfits_template.format(cam='b',maskname='A02') )

mtlzfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'mtlz_M2FS16_A02_full.csv'))
outfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'mtlz_M2FS16_A02_full.csv'.replace('mtlz','allzs_plus_mtlz')))

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
hc_cor = cord.radial_velocity_correction(kind='heliocentric',obstime=time,location=location)
dz = hc_cor/consts.c

full_table.add_column(Table.Column(data=full_table['redshift_est']+dz,name='z_est_helio'))



all_coords = SkyCoord(ra=full_table['RA']*u.deg,dec=full_table['DEC']*u.deg)
seps = cluster.separation(all_coords)
full_table.add_column(Table.Column(data=seps.to(u.arcsec).value,name='Proj_R_asec'))
full_table.add_column(Table.Column(data=(kpc_p_amin*seps).to(u.Mpc).value,name='Proj_R_Comoving_Mpc'))

spec_overlap = full_table[np.logical_not(full_table['sdss_zsp'].mask)]
spec_overlap = spec_overlap[((spec_overlap['cor']>0.6))]


deviations = spec_overlap['z_est_helio']-spec_overlap['sdss_zsp']
full_table.add_column(Table.Column(data=full_table['redshift_est']+dz-np.median(deviations),name='z_est_helio_debiased'))
dvs = (3.0e5)*(z_clust-full_table['z_est_helio_debiased'])/(1.+z_clust)
full_table.add_column(Table.Column(data=dvs,name='velocity'))

for cname in ['sdss_q_mode', 'sdss_m_SDSS12']:
    convert = []
    for val in full_table[cname]:
        if val:
            convert.append(1)
        else:
            convert.append(0)

    full_table.remove_column(cname)
    full_table.add_column(Table.Column(data=convert, name=cname))

convert = []
for val in full_table['template']:
    if val:
        convert.append(str(val))
    else:
        convert.append('')

full_table.remove_column('template')
full_table.add_column(Table.Column(data=convert, name='template'))



vizier_catalogs = ['sdss12']
radius = 5*u.Mpc / kpc_p_amin
Vizier.ROW_LIMIT = -1
result = Vizier.query_region(cluster, radius=radius, catalog=vizier_catalogs)
res_tab = result[0]

if np.all(res_tab['zsp'].mask):
    if np.all(res_tab['zph'].mask):
        out_tab = res_tab
    else:
        cut_tab = res_tab[np.where(~res_tab['zph'].mask)]
        out_tab = cut_tab[np.where(cut_tab['zph'] > -99)]
else:
    out_tab = res_tab[np.where(~res_tab['zsp'].mask)]

#out_tab = out_tab[((out_tab['zsp']>0.24)&(out_tab['zsp']<0.26))]

for cname in ['q_mode','m_SDSS12']:
    convert = []
    for val in out_tab[cname]:
        if val:
            convert.append(1)
        else:
            convert.append(0)

    out_tab.remove_column(cname)
    out_tab.add_column(Table.Column(data=convert,name=cname))




for col in out_tab.colnames:
    out_tab.rename_column(col, 'sdss_' + col)

for ii in np.arange(len(out_tab))[::-1]:
    if out_tab['sdss_SDSS12'][ii] in full_table['sdss_SDSS12']:
        print("Removing: {}".format(out_tab['sdss_SDSS12'][ii]))
        out_tab.remove_row(ii)

out_tab.add_column(Table.Column(data=out_tab['sdss_RA_ICRS'],name='RA'))
out_tab.add_column(Table.Column(data=out_tab['sdss_DE_ICRS'],name='DEC'))
out_tab.add_column(Table.Column(data=['T']*len(out_tab),name='TYPE'))
out_tab.add_column(Table.Column(data=[2000.0]*len(out_tab),name='EPOCH'))

all_sdss_coords = SkyCoord(ra=out_tab['sdss_RA_ICRS'],dec=out_tab['sdss_DE_ICRS'])
seps = cluster.separation(all_sdss_coords)
out_tab.add_column(Table.Column(data=seps.to(u.arcsec).value,name='Proj_R_asec'))
out_tab.add_column(Table.Column(data=(kpc_p_amin*seps).to(u.Mpc).value,name='Proj_R_Comoving_Mpc'))
out_tab.add_column(Table.Column(data=out_tab['sdss_zsp'],name='z_est_helio_debiased'))
dvs = (3.0e5)*(z_clust-out_tab['z_est_helio_debiased'])/(1.+z_clust)
out_tab.add_column(Table.Column(data=dvs,name='velocity'))



mega_table = vstack([full_table,out_tab])
for col in mega_table.columns:
    if type(mega_table[col].fill_value) in [str,np.str_,np.bytes_]:
        continue
    if np.isnan(mega_table[col].fill_value):
        mega_table[col].fill_value = -999

mega_table.write(outfile.replace('csv','fits'),format='fits',overwrite=True)

# filled = mega_table.filled()
# filled.write(outfile,format='ascii.csv',overwrite=True)



# plt.figure(); plt.hist(good_vals['z_est_helio'],bins=300); plt.xlim(0.22,0.32);
# ymin,ymax = plt.ylim()
# plt.vlines(.252,ymin,ymax,'k','--')
# plt.figure(); plt.plot(spec_overlap['z_est_helio'],spec_overlap['sdss_zsp'],'b.'); plt.plot([0,1],[0,1],'k--');
# deviations = spec_overlap['z_est_helio']-spec_overlap['sdss_zsp']
# meandev,meddev = np.mean(deviations),np.median(deviations)
#
# plt.figure(); plt.plot(spec_overlap['z_est_helio'],deviations,'b.');
# xmin,xmax = plt.xlim()
# plt.hlines(0.,xmin,xmax,'k','--',label='zero')
# plt.hlines(meddev,xmin,xmax,'g','--',label='median')
# plt.hlines(meandev,xmin,xmax,'c','--',label='mean')
# plt.legend(loc='best');
#
# plt.figure(); plt.hist(deviations,bins=20);
# plt.vlines(meandev,0,4,colors='k',label='mean');
# plt.vlines(meddev,0,4,colors='g',label='median');
# plt.legend(loc='best');
#
plt.figure()
plt.plot(mega_table['Proj_R_Comoving_Mpc'],mega_table['velocity'],'b.')
plt.xlabel("Projected Radius [Comoving Mpc]")
plt.ylabel('dv [km/s]')
xmin,xmax = plt.xlim()
ymin,ymax = -5000,5000
plt.ylim(ymin,ymax)
plt.hlines(0.,xmin,xmax,'k','--',label='zero')
plt.show()
# #plt.legend(loc='best');
#
#
# plt.show()


# for row in spec_overlap:
#     bests = np.argsort(np.abs(good_vals['redshift_est']-row['sdss_zsp']))
#     ap = (row['FIBNAME'])[0]
#     all_aps = np.asarray([(good_vals['FIBNAME'][itterrow])[0] for itterrow in bests])
#     first_match = np.where(all_aps==ap)[0][0]
#     best = bests[first_match]
#     if np.abs(row['sdss_zsp']-good_vals['redshift_est'][best])<0.001:
#         print('true: ',row['sdss_rmag'],row['sdss_zsp'],row['FIBNAME'],'\tmatch: ',good_vals['cor'][best],good_vals['redshift_est'][best],good_vals['FIBNAME'][best],'\t',row['sdss_zsp']-good_vals['redshift_est'][best])