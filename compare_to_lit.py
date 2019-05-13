

from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.coordinates import SkyCoord
import astropy.units as u
import astropy.constants as c
from matplotlib.tri import Triangulation
from mpl_toolkits import mplot3d

speed_of_light = c.c.to(u.km/u.s).value
catalog_dir = os.path.abspath("../../OneDrive - umich.edu/Research/M2FSReductions/catalogs")
corcut = 0.3
litname = 'HeCS_table2_ApJ-767-15.fit'
m2fsname = 'mtlz_A267_full.fits'

m2fs_ra_name, m2fs_dec_name, m2fs_z_name = 'RA','DEC','z_est_bary'
lit_ra_name, lit_dec_name, lit_v_name = '_RAJ2000','_DEJ2000','cz'

merged = Table.read(os.path.join(catalog_dir,'merged_target_lists',m2fsname),format='fits')
merged = merged[merged['cor'] > corcut]

litt = Table.read(os.path.join(catalog_dir,'literature',litname),format='fits')

merged_skies = SkyCoord(merged[m2fs_ra_name],merged[m2fs_dec_name],unit=(u.deg,u.deg))
litt_skies = SkyCoord(litt[lit_ra_name],litt[lit_dec_name],unit=(u.deg,u.deg))


litt_ind, seps, dist = merged_skies.match_to_catalog_sky(litt_skies)
matched_merged = merged[seps < 0.2*u.arcsec]
matched_litt = litt[(litt_ind[seps < 0.2*u.arcsec])]

litt_reds = matched_litt[lit_v_name]/speed_of_light
merg_reds = matched_merged['z_est_bary']

dzs = (merg_reds-litt_reds)
dvs = dzs*speed_of_light

plt.figure()
plt.plot(litt_reds,dvs,'b.')

plt.figure()
plt.hist(dvs[dvs<600],bins=30)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = merged[m2fs_ra_name]*np.cos(np.deg2rad(merged[m2fs_dec_name]))
ys = merged[m2fs_z_name]
zs = merged[m2fs_dec_name]
ax.scatter3D(xs, ys, zs, cmap='Greens')
ax.set_xlabel('RA')
ax.set_ylabel('z')
ax.set_zlabel('DEC')


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
xs = litt[lit_ra_name]*np.cos(np.deg2rad(litt[lit_dec_name]))
ys = litt[lit_v_name]
zs = litt[lit_dec_name]
ax.scatter3D(xs, ys, zs, cmap='Greens')
ax.set_xlabel('RA')
ax.set_ylabel('z')
ax.set_zlabel('DEC')


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xs = merged[m2fs_ra_name]*np.cos(np.deg2rad(merged[m2fs_dec_name]))
# zs = merged[m2fs_z_name]
# ys = merged[m2fs_dec_name]
# ax.plot_trisurf(xs, ys, zs,
#                 cmap='viridis', edgecolor='none')
# ax.set_xlabel('RA')
# ax.set_ylabel('DEC')
# ax.set_zlabel('z')
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xs = litt[lit_ra_name]*np.cos(np.deg2rad(litt[lit_dec_name]))
# zs = (litt[lit_v_name]/speed_of_light - 0.35)
# ys = litt[lit_dec_name]
# ax.plot_trisurf(xs, ys, zs,
#                 cmap='viridis', edgecolor='none')
# ax.set_xlabel('RA')
# ax.set_ylabel('DEC')
# ax.set_zlabel('z')



plt.show()