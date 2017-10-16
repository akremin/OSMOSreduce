# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:26:11 2017

@author: kremin
"""

from astropy.io import fits
from astropy.wcs import WCS

from astropy import units as u
import pdb
import numpy as np
import matplotlib.pyplot as plt
import os
import re


version = 'v3'
transforms = {1: 'ul', 2: 'u', 3: 'none', 4: 'l'}
camloc = {1:'br',2:'ur',3:'ul',4:'bl'}

#datadir = './'
datadir = '/u/home/kremin/value_storage/Data/M2FS/m2fsdata/sep2017/'

prog = re.compile('[br][0-9]{4}c1.fits')
for fil in os.listdir(datadir ):
    result = prog.match(fil)
    if result is None:
        continue
    fullfilename = os.path.join(datadir,fil)
    if fits.getheader(fullfilename)['BINNING'] != '2x2':
        continue
    camera = fil[0]
    filenumber = fil.split('c')[0].lstrip('br')
    if os.path.exists(os.path.join(datadir,"{}{}_c.fits".format(camera,filenumber))):
        continue
    try:
        for i in range(1,5):
            #filename = "/u/home/kremin/value_storage/Data/M2FS/m2fsdata/sep2017/r1962c{}.fits".format(i)
            filename = "{}{}c{}.fits".format(camera,filenumber,i)
            fullpath = os.path.join(datadir,filename)
            hdu = fits.open(fullpath)[0]
            wcs = WCS(hdu.header)
            header = hdu.header
            (y1, y2), (x1, x2) = [[int(x) for x in va.split(':')] for va in header['DATASEC'].strip('[]').split(',')]
            print(x1,x2,y1,y2)
            curtrans = transforms[i]
            curloc = camloc[i]
            curccd = hdu.data[x1 - 1:x2, y1 - 1:y2]

            if curtrans == 'l':
                pic = np.fliplr(curccd)
            elif curtrans == 'u':
                pic = np.flipud(curccd)
            elif curtrans == 'none':
                pic = curccd
            elif curtrans == 'ul':
                pic = np.flipud(np.fliplr(curccd))
            else:
                print("That didn't work")

            if curloc == 'br':
                #all positive
                xs = np.arange(x2,2*x2).astype(int)
                ys = np.arange(y2, 2 * y2).astype(int)
            elif curloc == 'bl':
                # negative in x pos in y
                xs = np.arange(x1-1, x2).astype(int)
                ys = np.arange(y2, 2 * y2).astype(int)
            elif curloc == 'ul':
                # neg in y and x
                xs = np.arange(x1-1, x2).astype(int)
                ys = np.arange(y1-1, y2).astype(int)
            elif curloc == 'ur':
                # neg in y  pos in x
                xs = np.arange(x2, 2 * x2).astype(int)
                ys = np.arange(y1-1, y2).astype(int)
            else:
                print("That didn't work")
            if i==1:
                final_image = np.ndarray(shape=(x2*2,y2*2))
                outheader = header

            y,x = np.meshgrid(ys,xs)
            print(x.shape,y.shape,hdu.data.shape,pic.shape,final_image.shape)
            final_image[x,y] = pic

        outheader.add_history('Cropped and merged by proc_ccd, version {}'.format(version))
        outfile = fits.HDUList(fits.PrimaryHDU(data=final_image,header=outheader))
        outfile.writeto(os.path.join(datadir,"{}{}_c.fits".format(camera,filenumber)),overwrite=True)
    except:
        print("Something went wrong with {}. Continuing".format(filename))

    #fig = plt.figure()
    #plt.imshow(final_image, animated=True)
    #plt.tight_layout()
    #plt.show()