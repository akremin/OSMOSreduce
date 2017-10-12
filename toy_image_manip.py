# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:26:11 2017

@author: kremin
"""

from astropy.io import fits
from astropy.wcs import WCS

from astropy import units as u
from ccdproc import CCDData, Combiner, combine
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


from ccdproc import wcs_project

ccds = []
headers = []
for i in range(1,5):
    filename = "/u/home/kremin/value_storage/Data/M2FS/m2fsdata/sep2017/r1962c{}.fits".format(i)

    hdu = fits.open(filename)[0]
    wcs = WCS(hdu.header) 
    ccd_dat = CCDData(hdu.data, unit=u.adu, wcs=wcs)
    header = hdu.header
    xcent, ycent = header['CHOFFX'],header['CHOFFY']
    (y1,y2),(x1,x2) = [[int(x) for x in va.split(':')] for va in header['DATASEC'].strip('[]').split(',')]
    print(x1,x2,y1,y2)
    if i==1:
        final = np.ndarray(shape=(y2*2,x2*2))
    if xcent < 0:
        ix1 = x1-1
        ix2 = x2
        xs = np.arange(ix1,ix2)
    else:
        ix1 = x2
        ix2 = 2*x2
        xs = np.arange(ix1,ix2)[::-1]
    if ycent < 0:
        iy1 = y1-1
        iy2 = y2
        ys = np.arange(iy1,iy2)
    else:
        iy1 = y2
        iy2 = 2*y2
        ys = np.arange(iy1,iy2)[::-1]
    x,y = np.meshgrid(xs,ys)
    final[y,x] = hdu.data[y1-1:y2,x1-1:x2]
    #for i,iy in enumerate(ys):
    #    for j,jx in enumerate(xs):
    #        final[iy,jx] = hdu.data[i,j]
    ccds.append(ccd_dat)
    headers.append(header)

#for i in range(1,5):
#    filename = "/u/home/kremin/value_storage/Data/M2FS/m2fsdata/sep2017/r1962c{}.fits".format(i)
#
#    hdu = fits.open(filename)[0]
#    wcs = WCS(hdu.header) 
#    ccd_dat = CCDData(hdu.data, unit=u.adu, wcs=wcs)
#    header = hdu.header
#    xcent, ycent = header['CHOFFX'],header['CHOFFY']
#    y1,y2,x1,x2 = 0, header['NAXIS2'], 0, header['NAXIS1']
#    print(x1,x2,y1,y2)
#    if i==1:
#        final = np.ndarray(shape=(y2*2,x2*2))
#    if xcent < 0:
#        ix1 = x1
#        ix2 = x2
#        xs = np.arange(ix1,ix2)
#    else:
#        ix1 = x2
#        ix2 = 2*x2
#        xs = np.arange(ix1,ix2)[::-1]
#    if ycent < 0:
#        iy1 = y1
#        iy2 = y2
#        ys = np.arange(iy1,iy2)
#    else:
#        iy1 = y2
#        iy2 = 2*y2
#        ys = np.arange(iy1,iy2)[::-1]
#    x,y = np.meshgrid(xs,ys)
#    final[y,x] = hdu.data #[y1:y2,x1:x2]
#    #for i,iy in enumerate(ys):
#    #    for j,jx in enumerate(xs):
#    #        final[iy,jx] = hdu.data[i,j]
#    ccds.append(ccd_dat)
#    headers.append(header)
    
    
#print(headers[0].keys)
#combiner = Combiner(ccds)
fig = plt.figure()
fin2 = final.copy()
fin2[10:,10:] = final[:-10,:-10]
ims = [[plt.imshow(final, animated=True)],[plt.imshow(fin2, animated=True)]]
ani = animation.ArtistAnimation(fig, ims, interval=1500, blit=True)#,repeat_delay=1000)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.tight_layout()
# ani.save('dynamic_images.mp4')

plt.show()
#stacked_image = combiner.average_combine()
#combined = combine(ccds)
#fig = plt.figure()
#fig.add_subplot(111)#, projection=wcs)
#plt.imshow(final, origin='lower', cmap=plt.cm.viridis)
#plt.xlabel('RA')
#plt.ylabel('Dec')