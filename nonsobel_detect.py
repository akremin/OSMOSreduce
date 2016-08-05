# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:05:38 2016

@author: kremin
"""

import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import argrelextrema

ypixmaxss = {}
ypixminss = {}
for color in ['b','r']:
    for cam in np.arange(4)+1:
        dict_key = color+str(cam)
        data_dir = '/u/home/kremin/value_storage/m2fsdata_jun2016'
        data_array = fits.open(data_dir +'/A02/'+color+'0582c'+str(cam)+'_b.fits')[0].data
        for i in range(583,592):
            data_array += fits.open(data_dir +('/A02/%s%04dc%d_b.fits' % (color,i,cam)))[0].data
        #data_array = data_array - np.mean(data_array)
        #im = scipy.misc.imread(data_dir +'/A02/b0633c3_b.fits',mode='I')

        #avd = ndimage.median_filter(dy-dx,2)
        #mag = numpy.hypot(dx, dy)  # magnitude
        mag = 255*(data_array)/np.max(np.abs(data_array))
        mag[np.abs(mag)<50] = 0
        plt.figure()
        plt.imshow(mag,vmin=-255,vmax=255)
        

        sumd = np.sum(mag,axis=1)
        #asumd = ndimage.median_filter(sumd,2)
        #asumd -= 56
        #asumd[asumd<0] = 0.
        
        ypixmaxss[dict_key] = argrelextrema(sumd, np.greater)
        ypixminss[dict_key] = argrelextrema(sumd, np.less)
        plt.figure()
        plt.plot(range(len(sumd)),sumd)
        plt.title(dict_key)