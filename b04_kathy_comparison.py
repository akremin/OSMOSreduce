from astropy.table import Table, join, vstack
import numpy as np
import astropy.constants as c
import astropy.units as u
cval = c.c.to(u.km/u.s).value

from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import os
import astropy.coordinates as ac

correlation_cut = 0.3   # must be strictly greater than this
overlap_criteria = 0.2*u.arcsec   # must be strictly less than this
flag_limit = 2   # must be strictly greater than this

main_dir = os.path.abspath('../data/B04_combined')

b04a = Table.read(os.path.join(main_dir,'mtlz_M2FS16_B04a_full.csv'),format='ascii.csv')
b04b = Table.read(os.path.join(main_dir,'mtlz_M2FS16_B04b_full.csv'),format='ascii.csv')

kathy = Table.read(os.path.join(main_dir,'Speczs_B04_AbellS1063_aj439002t2.csv'),header_start=4,format='ascii.csv')
kathy.rename_column('Redshift','z')



## Cut out objects with no redshifts
kathy = kathy[np.bitwise_not(kathy['z'].mask)]

## Make quality cuts for all catalogs
kathy = kathy[kathy['Error'] < 100] # in km/s

## Make quality cut on m2fs as well
b04a = b04a[b04a['cor'] > correlation_cut]
b04b = b04b[b04b['cor'] > correlation_cut]


## In the Kathy paper the member info is in the notes
memcol = np.array([int(('member' in str(row).lower())) for row in kathy['Notes']])
kathy.add_column(Table.Column(data=memcol,name="Member"))


kathy.add_column(Table.Column(data=kathy['ID'],name="CrossRefID",dtype='S8'))


## Generate conistent RA and DEC names:
kathy.rename_column('R.A. (2000)','RA')
kathy.rename_column('Decl. (2000)','Dec')


kcoords = SkyCoord(ra=kathy['RA'],dec=kathy['Dec'],unit=(u.hourangle,u.degree))

newname = []
for row in b04a:
    newname.append("a_"+str(row['FIBNAME']))
b04a.remove_column('FIBNAME')
b04a.add_column(Table.Column(data=newname,name='FIBERNAME'))
newname = []
for row in b04b:
    newname.append("b_"+str(row['FIBNAME']))
b04b.remove_column('FIBNAME')
b04b.add_column(Table.Column(data=newname,name='FIBERNAME'))


combined_m2fs = vstack([b04a,b04b])

kcoords = SkyCoord(ra=kathy['RA'],dec=kathy['Dec'],unit=(u.hourangle,u.degree))
m2fs_coords = SkyCoord(ra=combined_m2fs['RA'],dec=combined_m2fs['DEC'],unit=(u.deg,u.deg))

ref_inds, seps, dists = ac.match_coordinates_sky(m2fs_coords, kcoords)
m2fs_match_inds = np.where(seps<overlap_criteria)[0]
ref_match_inds = ref_inds[m2fs_match_inds]


m2fs_zs = []
ref_zs = []

for mind,rind in zip(m2fs_match_inds,ref_match_inds):
    ref_zs.append(kathy['z'][rind])
    m2fs_zs.append(combined_m2fs['z_est_bary'][mind])
    print("Fiber: {}    M2FS: {},    catalog: {}".format(combined_m2fs['FIBERNAME'][mind],combined_m2fs['z_est_bary'][mind],kathy['z'][rind]))


# m2fs_zs = []
# ref_zs = []
# for z in kathy['z']:
#     if np.any(np.abs(z-combined_m2fs['z_est_bary'])< 100/cval):
#         ref_zs.append(z)
#         m2fs_zs.append(combined_m2fs['z_est_bary'][np.argmin(np.abs(z-combined_m2fs['z_est_bary']))])


plt.figure()
plt.plot(np.array(ref_zs),cval*(np.array(m2fs_zs)-np.array(ref_zs)),'.')
plt.show()