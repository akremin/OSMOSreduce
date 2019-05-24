from astropy.table import Table, join, vstack
import numpy as np
import astropy.constants as c
import astropy.units as u
cval = c.c.to(u.km/u.s).value

from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
import os
import astropy.coordinates as ac

correlation_cut = 0.2 # must be strictly greater than this
overlap_criteria = 0.2*u.arcsec # must be strictly less than this
flag_limit = 2 ## must be strictly greater than this

main_dir = os.path.abspath('../../M2FSdata/B04_combined')

b04a = Table.read(os.path.join(main_dir,'mtlz_M2FS16_B04a_full.csv'),format='ascii.csv')
b04b = Table.read(os.path.join(main_dir,'mtlz_M2FS16_B04b_full.csv'),format='ascii.csv')

kathy = Table.read(os.path.join(main_dir,'Speczs_B04_AbellS1063_aj439002t2.csv'),header_start=4,format='ascii.csv')
kathy.rename_column('Redshift','z')
muse1 = Table.read(os.path.join(main_dir,'MUSE_B04_AbellS1063_speczs.csv'),header_start=5,format='ascii.csv')
muse2 = Table.read(os.path.join(main_dir,'MUSE_B04_AbellS1063_speczs_2.csv'),header_start=4,format='ascii.csv')
muse3 = Table.read(os.path.join(main_dir,'MUSE_B04_AbellS1063_speczs_3.csv'),header_start=4,format='ascii.csv')

## Muse 2 and 3 paper use 1 to mean to good quality flag, while muse 1 paper uses 4 to be the best, and 1 is uncertain.
## Change Muse 2 and 3 so they are consistent with Muse 1 convention
muse2['QF'] = 5 - muse2['QF']
muse3['QF'] = 5 - muse3['QF']

## Cut out objects with no redshifts
kathy = kathy[np.bitwise_not(kathy['z'].mask)]
muse1 = muse1[np.bitwise_not(muse1['z'].mask)]
muse2 = muse2[np.bitwise_not(muse2['z'].mask)]
muse3 = muse3[np.bitwise_not(muse3['z'].mask)]

muse1 = muse1[muse1['z'] > 0.0001]
muse2 = muse2[muse2['z'] > 0.0001]
muse3 = muse3[muse3['z'] > 0.0001]

## Make quality cuts for all catalogs
kathy = kathy[kathy['Error'] < 100] # in km/s

muse1 = muse1[((muse1['QF']>flag_limit)&(muse1['QF']<9))]
muse2 = muse2[((muse2['QF']>flag_limit)&(muse2['QF']<9))]
muse3 = muse3[((muse3['QF']>flag_limit)&(muse3['QF']<9))]

## Make quality cut on m2fs as well
b04a = b04a[b04a['cor'] > correlation_cut]
b04b = b04b[b04b['cor'] > correlation_cut]

## Muse 2 and 3 are from the same paper. Muse 2 was a table of cluster members, Muse 3 was a table of nonmembers
muse2.add_column(Table.Column(data=np.ones(len(muse2)),name="Member"))
muse3.add_column(Table.Column(data=np.zeros(len(muse3)),name="Member"))

## In the Kathy paper the member info is in the notes
memcol = np.array([int(('member' in str(row).lower())) for row in kathy['Notes']])
kathy.add_column(Table.Column(data=memcol,name="Member"))


kathy.add_column(Table.Column(data=kathy['ID'],name="CrossRefID",dtype='S8'))
muse1.add_column(Table.Column(data=muse1['ID'],name="CrossRefID",dtype='S8'))
muse2.add_column(Table.Column(data=muse2['ID'],name="CrossRefID",dtype='S8'))
muse3.add_column(Table.Column(data=muse3['ID'],name="CrossRefID",dtype='S8'))


## Generate conistent RA and DEC names:
kathy.rename_column('R.A. (2000)','RA')
kathy.rename_column('Decl. (2000)','Dec')
# m1 has correct labels
muse2.rename_column('RA (J2000)','RA')
muse2.rename_column('Dec (J2000)','Dec')
muse3.rename_column('RA (J2000)','RA')
muse3.rename_column('Dec (J2000)','Dec')


kcoords = SkyCoord(ra=kathy['RA'],dec=kathy['Dec'],unit=(u.hourangle,u.degree))
m1coords = SkyCoord(ra=muse1['RA'],dec=muse1['Dec'],unit=(u.degree,u.degree))
m2coords = SkyCoord(ra=muse2['RA'],dec=muse2['Dec'],unit=(u.hourangle,u.degree))
m3coords = SkyCoord(ra=muse3['RA'],dec=muse3['Dec'],unit=(u.hourangle,u.degree))

kathy.add_column(Table.Column(data=kcoords.icrs.ra.deg,name="CrossRefRA"))
muse1.add_column(Table.Column(data=m1coords.icrs.ra.deg,name="CrossRefRA"))
muse2.add_column(Table.Column(data=m2coords.icrs.ra.deg,name="CrossRefRA"))
muse3.add_column(Table.Column(data=m3coords.icrs.ra.deg,name="CrossRefRA"))

kathy.add_column(Table.Column(data=kcoords.icrs.dec.deg,name="CrossRefDEC"))
muse1.add_column(Table.Column(data=m1coords.icrs.dec.deg,name="CrossRefDEC"))
muse2.add_column(Table.Column(data=m2coords.icrs.dec.deg,name="CrossRefDEC"))
muse3.add_column(Table.Column(data=m3coords.icrs.dec.deg,name="CrossRefDEC"))

match_1tok_ind,sep1, dist = ac.match_coordinates_sky(m1coords,kcoords)
# match_2tok_ind,sep2, dist = ac.match_coordinates_sky(m2coords,kcoords)
# match_3tok_ind,sep3, dist = ac.match_coordinates_sky(m3coords,kcoords)

goodseps1 = (sep1 < overlap_criteria)
# goodseps2 = (sep2 < overlap_criteria)
# goodseps3 = (sep3 < overlap_criteria)

inds1 = np.where(goodseps1)[0]
# inds2 = np.where(goodseps2)[0]
# inds3 = np.where(goodseps3)[0]

indsk1 = match_1tok_ind[goodseps1]
# indsk2 = match_2tok_ind[goodseps2]
# indsk3 = match_3tok_ind[goodseps3]

matched_t1 = muse1[inds1]
matched_k1 = kathy[indsk1]

# matched_t2 = muse2[inds2]
# matched_k2 = kathy[indsk2]
#
# matched_t3 = muse3[inds3]
# matched_k3 = kathy[indsk3]
# ii=1
# for muse_tab,ktab in [(matched_t1,matched_k1),(matched_t2,matched_k2),(matched_t3,matched_k3)]:
#     print("Cat: {}".format(ii))
#
#     for (row_muse,row_kath) in zip(muse_tab,ktab):
#         print("Muse: {}  Kathy: {}  Difference: {}".format(row_muse['z'],row_kath['z'],row_muse['z']-row_kath['z']))
#         if ii > 1:
#             print("---> Muse quoted prev: {}  refid: {}   kathy: {}  id: {}".format(row_muse['z prev'], row_muse['IDref'], row_kath['Redshift'], row_kath['ID']))
#     ii += 1


print("first muse")
for (ind_muse,ind_kath) in zip(inds1,indsk1):
    print("Matched these two with seps: {},  zs: {}  and {}".format(sep1[ind_muse].to(u.arcsec),muse1['z'][ind_muse],kathy['z'][ind_kath]))
    if muse1['z'][ind_muse] > 0.0001:
        kathy['CrossRefID'][ind_kath] = muse1['CrossRefID'][ind_muse]
        kathy['CrossRefRA'][ind_kath] = muse1['CrossRefRA'][ind_muse]
        kathy['CrossRefDEC'][ind_kath] = muse1['CrossRefDEC'][ind_muse]


print("On to second muse")
for ind_muse in range(len(muse2)):
    if not muse2['IDref'].mask[ind_muse]:
        if muse2['z'][ind_muse] > 0.0001:
            matchname = muse2['IDref'][ind_muse]
            inds_kath = np.where(kathy['ID'] == int(matchname))[0]
            if len(inds_kath) == 0:
                continue
            else:
                ind_kath = inds_kath[0]
            print("Matched {} with {}, zs: {}  and {}".format(matchname, kathy['ID'][ind_kath], muse2['z prev'][ind_muse],
                                                            kathy['z'][ind_kath]))
            kathy['CrossRefID'][ind_kath] = muse2['CrossRefID'][ind_muse]
            kathy['CrossRefRA'][ind_kath] = muse2['CrossRefRA'][ind_muse]
            kathy['CrossRefDEC'][ind_kath] = muse2['CrossRefDEC'][ind_muse]

print("On to third muse")
for ind_muse in range(len(muse3)):
    if not muse3['IDref'].mask[ind_muse]:
        if muse3['z'][ind_muse] > 0.0001:
            if '(3)' in muse3['IDref'][ind_muse]:
                matchname = str(muse3['IDref'][ind_muse]).split(' ')[0]
                inds_kath = np.where(kathy['ID']==int(matchname))[0]
                if len(inds_kath) == 0:
                    continue
                else:
                    ind_kath = inds_kath[0]
                print("Matched {} with {}, zs: {}  and {}".format(matchname,kathy['ID'][ind_kath],muse3['z prev'][ind_muse],kathy['z'][ind_kath]))
                kathy['CrossRefID'][ind_kath] = muse3['CrossRefID'][ind_muse]
                kathy['CrossRefRA'][ind_kath] = muse3['CrossRefRA'][ind_muse]
                kathy['CrossRefDEC'][ind_kath] = muse3['CrossRefDEC'][ind_muse]




merged = join(left=kathy,right=muse1,keys=['CrossRefID','CrossRefRA','CrossRefDEC'],join_type='outer',table_names=['K','m1'])
merged = join(left=merged,right=muse2,keys=['CrossRefID','CrossRefRA','CrossRefDEC'],join_type='outer',table_names=['C','m2'])
merged = join(left=merged,right=muse3,keys=['CrossRefID','CrossRefRA','CrossRefDEC'],join_type='outer',table_names=['C','m3'])

merged.write(os.path.join(main_dir,'merged_muses_and_aj439002.csv'),overwrite=True)


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


m2fs_coords = SkyCoord(ra=combined_m2fs['RA'],dec=combined_m2fs['DEC'],unit=(u.deg,u.deg))
ref_coords = SkyCoord(ra=merged['CrossRefRA'],dec=merged['CrossRefDEC'],unit=(u.deg,u.deg))

ref_inds, seps, dists = ac.match_coordinates_sky(m2fs_coords, ref_coords)
m2fs_match_inds = np.where(seps<overlap_criteria)[0]
ref_match_inds = ref_inds[m2fs_match_inds]


m2fs_zs = [[],[],[],[]]
ref_zs = [[],[],[],[]]
print("                      K  C  M1  V  M3  ")
for mind,rind in zip(m2fs_match_inds,ref_match_inds):
    merged_row = merged[rind]
    print("M2FS: {},    catalog: {} {} {} {} {}".format(combined_m2fs['z_est_helio'][mind],float(merged_row['z_K']),float(merged_row['z_C']),float(merged_row['z_m1']),float(merged_row['z VIMOS']),float(merged_row['z_m3'])))
    zposs = [float(merged_row['z_K']),float(merged_row['z_C']),float(merged_row['z_m1']),float(merged_row['z VIMOS']),float(merged_row['z_m3'])]
    print(m2fs_coords[mind], ref_coords[rind])
    for ii in range(len(ref_zs)):
        if not np.isnan(zposs[ii]):
            ref_zs[ii].append(zposs[ii])
            m2fs_zs[ii].append(combined_m2fs['z_est_helio'][mind])



plt.figure()
for ii in range(len(ref_zs)):
    plt.plot(np.array(m2fs_zs[ii]),cval*(np.array(m2fs_zs[ii])-np.array(ref_zs[ii])),'.',label=str(ii))

plt.legend(loc='best')
plt.show()