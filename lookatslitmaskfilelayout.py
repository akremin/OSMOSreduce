# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 11:33:36 2017

@author: kremin
"""

import numpy as np
import matplotlib.pyplot as plt
import os
from astropy.io import fits


xs1 = []
ys1 = []
xs2 = []
ys2 = []


img1 = fits.open('/nfs/kremin/Data/goodman_jan17/Kremin10/mask1/data_products/comp/Kremin10_comp.aligned.combined.fits')[0].data
with open('/nfs/kremin/Data/goodman_jan17/Kremin10/maskfiles/Kremin10_Mask1.txt','r') as fil:
    for line in fil:
        #if line[:2] == 'G0':
        #    spl = line.split(' ')
        #    x10 = float(spl[1][1:])
        #    y10 = float(spl[2][1:])
        if line[:2] == 'G1':
            throw1,throw2,x,y = line.split(' ')
            xs1.append(float(x[1:]))
            ys1.append(float(y[1:]))
            
img2 = fits.open('/nfs/kremin/Data/goodman_jan17/Kremin10/mask2/data_products/comp/Kremin10_2comp.aligned.combined.fits')[0].data
with open('/nfs/kremin/Data/goodman_jan17/Kremin10/maskfiles/Kremin10_Mask2.txt','r') as fil:
    for line in fil:
        #if line[:2] == 'G0':
        #    spl = line.split(' ')
        #    x20 = float(spl[1][1:])
        #    y20 = float(spl[2][1:])
        if line[:2] == 'G1':
            throw1,throw2,x,y = line.split(' ')
            xs2.append(float(x[1:]))
            ys2.append(float(y[1:]))            

xs1 = np.asarray(xs1)-np.mean([min(xs1),max(xs1)])
ys1 = np.asarray(ys1)-np.mean([min(ys1),max(ys1)])
xs2 = np.asarray(xs2)-np.mean([min(xs2),max(xs2)])
ys2 = np.asarray(ys2)-np.mean([min(ys2),max(ys2)])
y,x = img1.shape
scalefac = 10.42
plt.figure()
plt.imshow(img1)
plt.plot(int(x/2)+scalefac*ys1,int(y/2)+scalefac*xs1-7,'y.')
#plt.gca().set_aspect('equal', adjustable='box')
#plt.plot(x10,y10,'r*')
#plt.xlim(min(xs1),max(xs1))
#plt.ylim(min(xs1),max(xs1))

plt.figure()
plt.imshow(img2)
plt.plot(int(x/2)+scalefac*ys2,int(y/2)+scalefac*xs2-7,'y.')
#plt.gca().set_aspect('equal', adjustable='box')
#plt.plot(x20,y20,'r*')
#plt.xlim(min(xs2),max(xs2))
#plt.ylim(min(xs2),max(xs2))


plt.show()