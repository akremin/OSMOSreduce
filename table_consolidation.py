import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, vstack, join
from astropy.io import fits


#apperature, redshift_est, cor, template
#ID,FIBNAME,sdss_SDSS12,RA,DEC,sdss_zsp,sdss_zph,sdss_rmag,MAG
zfit_path = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','A02','zfits'))
zfits_template = 'zfit_{cam}_{maskname}.fits'
zfit_file = os.path.join( zfit_path, zfits_template.format(cam='r',maskname='A02') )
zfit_file1 = os.path.join( zfit_path, zfits_template.format(cam='b',maskname='A02') )

mtlzfile = os.path.abspath(os.path.join(os.curdir,'..','..', 'OneDrive - umich.edu','Research','M2FSReductions','catalogs','merged_target_lists', 'mtlz_M2FS16_A02_selected.csv'))


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

full_table = join(combined_table,mtlz,'FIBNAME',join_type='inner')
spec_overlap = full_table[np.logical_not(full_table['sdss_zsp'].mask)]
#good_vals[['FIBNAME','redshift_est','sdss_zsp']]
good_vals = full_table[((full_table['cor']>0.4))]
#plt.figure(); plt.hist(good_vals['redshift_est'],bins=100); plt.xlim(0.22,0.32); plt.show()
for row in spec_overlap:
    bests = np.argsort(np.abs(good_vals['redshift_est']-row['sdss_zsp']))
    ap = (row['FIBNAME'])[0]
    all_aps = np.asarray([(good_vals['FIBNAME'][itterrow])[0] for itterrow in bests])
    first_match = np.where(all_aps==ap)[0][0]
    best = bests[first_match]
    if np.abs(row['sdss_zsp']-good_vals['redshift_est'][best])<0.001:
        print('true: ',row['sdss_rmag'],row['sdss_zsp'],row['FIBNAME'],'\tmatch: ',good_vals['cor'][best],good_vals['redshift_est'][best],good_vals['FIBNAME'][best],'\t',row['sdss_zsp']-good_vals['redshift_est'][best])