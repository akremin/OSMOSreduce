# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:26:11 2017

@author: kremin
"""

from astropy.io import fits
from astropy.wcs import WCS

from astropy import units as u
#from ccdproc import CCDData, Combiner, combine
import pdb
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


#from ccdproc import wcs_project
allxs,allys = [],[]
ccds = []
names = []
headers = []
version = 'v3'

filenumber = '1680'#'1962'
if version == 'v3':
    for i in range(1,5):
        #filename = "/u/home/kremin/value_storage/Data/M2FS/m2fsdata/sep2017/r1962c{}.fits".format(i)
        filename = "./r{}c{}.fits".format(filenumber,i)

        hdu = fits.open(filename)[0]
        wcs = WCS(hdu.header)
        #ccd_dat = CCDData(hdu.data, unit=u.adu, wcs=wcs)
        header = hdu.header
        # rows = x = naxis1               columns = y = naxis2
        xbin,ybin = [int(x) for x in header['BINNING'].split('x')]
        xcent, ycent = header['CHOFFX']/(xbin*header['SCALE']),header['CHOFFY']/(ybin*header['SCALE'])
        (x1,x2),(y1,y2) = [[int(x) for x in va.split(':')] for va in header['DATASEC'].strip('[]').split(',')]
        print((x1,x2,y1,y2))
        xtrans = np.floor(x2 + xcent - np.abs(xcent))
        ytrans = np.floor(y2 + ycent - np.abs(ycent))
        if i==1:
            final_image = np.ndarray(shape=(x2*2,y2*2))
        if xcent < 0:
            ix1 = x1-1
            ix2 = x2
            xs = np.arange(ix1,ix2).astype(int)
        else:
            ix1 = xtrans-1
            ix2 = xtrans+x2-1
            xs = np.arange(ix1,ix2).astype(int)#[::-1].astype(int)
        if ycent < 0:
            iy1 = y1-1
            iy2 = y2
            ys = np.arange(iy1,iy2).astype(int)
        else:
            iy1 = ytrans-1
            iy2 = ytrans+y2-1
            ys = np.arange(iy1,iy2).astype(int)#[::-1].astype(int)
        y,x = np.meshgrid(ys,xs)
        print((xtrans,ytrans))
        print((x.shape,y.shape,hdu.data.shape,hdu.data[x1-1:x2,y1-1:y2].shape,final_image.shape))
        #final_image[x,y] = hdu.data[x1-1:x2,y1-1:y2]
        #for i,iy in enumerate(ys):
        #    for j,jx in enumerate(xs):
        #        final[iy,jx] = hdu.data[i,j]
        ccds.append(hdu.data[x1-1:x2,y1-1:y2])
        allxs.append(x)
        allys.append(y)
        names.append(i)
        headers.append(header)
elif version == 'v2':
    for i in range(1,5):
        filename = "/u/home/kremin/value_storage/Data/M2FS/m2fsdata/sep2017/rc{}.fits".format(i)

        hdu = fits.open(filename)[0]
        wcs = WCS(hdu.header)
        #ccd = CCDData(hdu.data, unit=u.adu, wcs=wcs)
        ccd_dat = hdu.data
        header = hdu.header
        # rows = x = naxis1               columns = y = naxis2
        xcent, ycent = header['CHOFFX']/header['SCALE'],header['CHOFFY']/header['SCALE']
        (x1, x2),(y1,y2) = [[int(x) for x in va.split(':')] for va in header['DATASEC'].strip('[]').split(',')]
        x1 -= 1
        y1 -= 1
        print((x1,x2,y1,y2))
        #xbin,ybin = [int(x) for x in header['BINNING'].split('x')]
        xbin, ybin = 1, 1
        xcent,ycent = xcent/xbin, ycent/ybin
        y1, y2 = y1/ybin,y2/ybin
        x1, x2 = x1/xbin,x2/xbin
        if i == 1:
            final_image = np.zeros(shape=(2*(y2-y1+1),2*(x2-x1+1)))
        pixtransform_x = int(0.5*(np.abs(xcent) - xcent))
        pixtransform_y = int(0.5*(np.abs(ycent) - ycent))
        print((pixtransform_x,pixtransform_y))
        #for x in np.arange(x1,x2)[::-1]:
        #    for y in np.arange(y1,y2)[::-1]:
        #        final_image[y+pixtransform_y,x+pixtransform_x] = ccd_dat[y,x]

        #for i,iy in enumerate(ys):
        #    for j,jx in enumerate(xs):
        #        final[iy,jx] = hdu.data[i,j]
        ccds.append(ccd_dat)
        headers.append(header)
elif version == 'v1':
    for i in range(1,5):
       filename = "/u/home/kremin/value_storage/Data/M2FS/m2fsdata/sep2017/r1962c{}.fits".format(i)

       hdu = fits.open(filename)[0]
       wcs = WCS(hdu.header)
       #ccd_dat = CCDData(hdu.data, unit=u.adu, wcs=wcs)
       header = hdu.header
       xcent, ycent = header['CHOFFX'],header['CHOFFY']
       y1,y2,x1,x2 = 0, header['NAXIS2'], 0, header['NAXIS1']
       print((x1,x2,y1,y2))
       if i==1:
           final_image = np.ndarray(shape=(y2*2,x2*2))
       if xcent < 0:
           ix1 = x1
           ix2 = x2
           xs = np.arange(ix1,ix2)
       else:
           ix1 = x2
           ix2 = 2*x2
           xs = np.arange(ix1,ix2)[::-1]
       if ycent < 0:
           iy1 = y1
           iy2 = y2
           ys = np.arange(iy1,iy2)
       else:
           iy1 = y2
           iy2 = 2*y2
           ys = np.arange(iy1,iy2)[::-1]
       x,y = np.meshgrid(xs,ys)
       #final_image[y,x] = hdu.data #[y1:y2,x1:x2]
       #for i,iy in enumerate(ys):
       #    for j,jx in enumerate(xs):
       #        final[iy,jx] = hdu.data[i,j]
       ccds.append(hdu.data)
       headers.append(header)

image1 = ccds[0]
image2 = ccds[1]
image3 = ccds[2]
image4 = ccds[3]
#for i in range(4):
i=0
xs = allxs[i]
ys = allys[i]
for orient in ['ul']:
    if orient == 'l':
        pic1 = np.fliplr(image1)
    elif orient == 'u':
        pic1 = np.flipud(image1)
    elif orient == 'none':
        pic1 = image1
    elif orient == 'ul':
        pic1 = np.flipud(np.fliplr(image1))
    else:
        print("That didn't work")
    final_image[xs,ys] = pic1
    it2s = list(range(4))
    it2s.remove(i)

    #for ii in it2s:
    ii = 3
    orient2 = 'u'
    x2s = allxs[ii]
    y2s = allys[ii]
    #for orient2 in ['u', 'l','none', 'ul']:
    if orient2 == 'l':
        pic2 = np.fliplr(image2)
    elif orient2 == 'u':
        pic2 = np.flipud(image2)
    elif orient2 == 'none':
        pic2 = image2
    elif orient2 == 'ul':
        pic2 = np.flipud(np.fliplr(image2))
    else:
        print("That didn't work")
    final_image[x2s,y2s] = pic2

    it3s = list(range(4))
    it3s.remove(i)
    it3s.remove(ii)
    iii = 2
    orient3 = 'none'
    it4 = 1
    orient4 = 'l'
    #for iii in it3s:
    x3s = allxs[iii]
    y3s = allys[iii]
#    for orient3 in ['l', 'u', 'none', 'ul']:
    if orient3 == 'l':
        pic3 = np.fliplr(image3)
    elif orient3 == 'u':
        pic3 = np.flipud(image3)
    elif orient3 == 'none':
        pic3 = image3
    elif orient3 == 'ul':
        pic3 = np.flipud(np.fliplr(image3))
    else:
        print("That didn't work")
    final_image[x3s, y3s] = pic3
    #it4 = range(4)
    #it4.remove(i)
    #it4.remove(ii)
    #it4.remove(iii)
    #it4 = it4[0]
    x4s,y4s = allxs[it4],allys[it4]
    #for orient4 in ['l', 'u', 'none', 'ul']:
    if orient4 == 'l':
        pic4 = np.fliplr(image4)
    elif orient4 == 'u':
        pic4 = np.flipud(image4)
    elif orient4 == 'none':
        pic4 = image4
    elif orient4 == 'ul':
        pic4 = np.flipud(np.fliplr(image4))
    else:
        print("That didn't work")
    final_image[x4s, y4s] = pic4
    plt.figure()
    plt.imshow(final_image)
    plt.show()
    #plt.savefig('./outputs/{}-{}_{}-{}_{}-{}_{}-{}.png'.format(i,orient,ii,orient2,iii,orient3,it4,orient4))
    #plt.close()



#print(headers[0].keys)
#combiner = Combiner(ccds)
# fig = plt.figure()
# fin2 = final_image.copy()
# fin2[10:,10:] = final_image[:-10,:-10]
# ims = [[plt.imshow(final_image, animated=True)],[plt.imshow(fin2, animated=True)]]
# ani = animation.ArtistAnimation(fig, ims, interval=1500, blit=True)#,repeat_delay=1000)
# figManager = plt.get_current_fig_manager()
# figManager.window.showMaximized()
# plt.tight_layout()
# ani.save('dynamic_images.mp4')

plt.show()
#stacked_image = combiner.average_combine()
#combined = combine(ccds)
#fig = plt.figure()
#fig.add_subplot(111)#, projection=wcs)
#plt.imshow(final, origin='lower', cmap=plt.cm.viridis)
#plt.xlabel('RA')
#plt.ylabel('Dec')