from astropy.table import Table, join, vstack
import numpy as np
import astropy.constants as c
import astropy.units as u
cval = c.c.to(u.km/u.s).value

from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import os
import astropy.coordinates as ac

correlation_cut = 0.3 # must be strictly greater than this
overlap_criteria = 0.2*u.arcsec # must be strictly less than this
flag_limit = 2 ## must be strictly greater than this

main_dir = os.path.abspath('../data/B04_combined')

b04a = Table.read(os.path.join(main_dir,'mtlz_M2FS16_B04a_full.csv'),format='ascii.csv')
b04b = Table.read(os.path.join(main_dir,'mtlz_M2FS16_B04b_full.csv'),format='ascii.csv')

muse1 = Table.read(os.path.join(main_dir,'MUSE_B04_AbellS1063_speczs.csv'),header_start=5,format='ascii.csv')
muse2 = Table.read(os.path.join(main_dir,'MUSE_B04_AbellS1063_speczs_2.csv'),header_start=4,format='ascii.csv')
muse3 = Table.read(os.path.join(main_dir,'MUSE_B04_AbellS1063_speczs_3.csv'),header_start=4,format='ascii.csv')

## Muse 2 and 3 paper use 1 to mean to good quality flag, while muse 1 paper uses 4 to be the best, and 1 is uncertain.
## Change Muse 2 and 3 so they are consistent with Muse 1 convention
# muse1['QF'] = 5 - muse1['QF']
muse2['QF'] = 5 - muse2['QF']
muse3['QF'] = 5 - muse3['QF']


## Cut out objects with no redshifts
muse1 = muse1[np.bitwise_not(muse1['z'].mask)]
muse2 = muse2[np.bitwise_not(muse2['z'].mask)]
muse3 = muse3[np.bitwise_not(muse3['z'].mask)]

muse1 = muse1[muse1['z'] > 0.0001]
muse2 = muse2[muse2['z'] > 0.0001]
muse3 = muse3[muse3['z'] > 0.0001]

## Make quality cuts for all catalogs
muse1 = muse1[((muse1['QF']>flag_limit)&(muse1['QF']<9))]
muse2 = muse2[((muse2['QF']>flag_limit)&(muse2['QF']<9))]
muse3 = muse3[((muse3['QF']>flag_limit)&(muse3['QF']<9))]


## Make quality cut on m2fs as well
b04a = b04a[b04a['cor'] > correlation_cut]
b04b = b04b[b04b['cor'] > correlation_cut]

## Muse 2 and 3 are from the same paper. Muse 2 was a table of cluster members, Muse 3 was a table of nonmembers
muse2.add_column(Table.Column(data=np.ones(len(muse2)),name="Member"))
muse3.add_column(Table.Column(data=np.zeros(len(muse3)),name="Member"))


muse1.add_column(Table.Column(data=muse1['ID'],name="CrossRefID",dtype='S8'))
muse2.add_column(Table.Column(data=muse2['ID'],name="CrossRefID",dtype='S8'))
muse3.add_column(Table.Column(data=muse3['ID'],name="CrossRefID",dtype='S8'))


## Generate conistent RA and DEC names:
# m1 has correct labels
muse2.rename_column('RA (J2000)','RA')
muse2.rename_column('Dec (J2000)','Dec')
muse3.rename_column('RA (J2000)','RA')
muse3.rename_column('Dec (J2000)','Dec')

muse2 = join(muse2,muse3,keys=["CrossRefID","RA","Dec","z","QF"],join_type='outer')

m1coords = SkyCoord(ra=muse1['RA'],dec=muse1['Dec'],unit=(u.degree,u.degree))
m2coords = SkyCoord(ra=muse2['RA'],dec=muse2['Dec'],unit=(u.hourangle,u.degree))

muse1.add_column(Table.Column(data=m1coords.icrs.ra.deg,name="CrossRefRA"))
muse2.add_column(Table.Column(data=m2coords.icrs.ra.deg,name="CrossRefRA"))

muse1.add_column(Table.Column(data=m1coords.icrs.dec.deg,name="CrossRefDEC"))
muse2.add_column(Table.Column(data=m2coords.icrs.dec.deg,name="CrossRefDEC"))

match_1to2_ind,sep12, dist = ac.match_coordinates_sky(m1coords,m2coords)

goodseps12 = (sep12 < overlap_criteria)

inds12 = np.where(goodseps12)[0]

indsm12 = match_1to2_ind[goodseps12]

print("first muse")
for (ind_muse,ind_kath) in zip(indsm12,inds12):
    print("Matched these two with seps: {},  zs: {}  and {}".format(sep12[ind_muse].to(u.arcsec),muse2['z'][ind_muse],muse1['z'][ind_kath]))
    if muse2['z'][ind_muse] > 0.0001:
        muse1['CrossRefID'][ind_kath] = muse2['CrossRefID'][ind_muse]
        muse1['CrossRefRA'][ind_kath] = muse2['CrossRefRA'][ind_muse]
        muse1['CrossRefDEC'][ind_kath] = muse2['CrossRefDEC'][ind_muse]

merged = join(left=muse1,right=muse2,keys=['CrossRefID','CrossRefRA','CrossRefDEC'],join_type='outer',table_names=['m1','m23'])


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

mcoords = SkyCoord(ra=merged['CrossRefRA'],dec=merged['CrossRefDEC'],unit=(u.degree,u.degree))
m2fs_coords = SkyCoord(ra=combined_m2fs['RA'],dec=combined_m2fs['DEC'],unit=(u.deg,u.deg))

ref_inds, seps, dists = ac.match_coordinates_sky(m2fs_coords, mcoords)
m2fs_match_inds = np.where(seps<overlap_criteria)[0]
ref_match_inds = ref_inds[m2fs_match_inds]


m2fs_zs = []
ref_zs = []

for mind,rind in zip(m2fs_match_inds,ref_match_inds):
    ref_zs.append(merged['z_m1'][rind])
    m2fs_zs.append(combined_m2fs['z_est_bary'][mind])
    print("Fiber: {}    M2FS: {},    catalog: {}".format(combined_m2fs['FIBERNAME'][mind],combined_m2fs['z_est_bary'][mind],merged['z_m1'][rind]))


plt.figure()
plt.plot(np.array(ref_zs),cval*(np.array(m2fs_zs)-np.array(ref_zs)),'.')
plt.show()