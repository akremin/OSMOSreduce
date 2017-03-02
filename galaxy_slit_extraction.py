#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 22:33:50 2017

@author: kremin
"""
#import numpy as np
#import matplotlib.pyplot as plt
#from slit_find import _gaus
#from scipy.optimize import curve_fit
#binnedx = 2070    # 4064    # this is in binned pixels
#binnedy = 1256    # this is in binned pixels
#binxpix_mid = int(binnedx/2)
#binypix_mid = int(binnedy/2)

#def maketheseplots(d2_spectra_s,slit_width):
gal_guess = np.arange(0,slit_width,1)[np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1)== \
                                    np.max(np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1))][0]
#_gaus(x,amp,sigma,x0,background)
normd_media_narrowcross = np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1)

popt_g,pcov_g = curve_fit(_gaus,np.arange(0,slit_width,1),normd_media_narrowcross,p0=[1,4.0,gal_guess,np.min(normd_media_narrowcross)],maxfev = 10000)
gal_amp,gal_wid,gal_pos,sky_val = popt_g
#gal_wid = popt_g[1]#4.0
#if gal_wid > 5: gal_wid=5
upper_gal = gal_pos + gal_wid*2.0
lower_gal = gal_pos - gal_wid*2.0
if upper_gal >= slit_width: upper_gal = (slit_width-2)
if lower_gal <= 0: lower_gal = 2
raw_gal = d2_spectra_s.T[lower_gal:upper_gal,:]
sky = np.append(d2_spectra_s.T[:lower_gal,:],d2_spectra_s.T[upper_gal:,:],axis=0)
sky_sub = np.zeros(raw_gal.shape) + np.median(sky,axis=0)
sky_sub_tot = np.zeros(d2_spectra_s.T.shape) + np.median(sky,axis=0)
plt.figure()
plt.imshow(np.log(d2_spectra_s.T),aspect=35,origin='lower')#
plt.axhline(lower_gal,color='k',ls='--')
plt.axhline(upper_gal,color='k',ls='--')
plt.xlim(0,binnedx)
plt.show()
plt.figure()
plt.plot(np.arange(0,slit_width,1),_gaus(np.arange(0,slit_width,1),*popt_g))
plt.plot(np.arange(0,slit_width,1),np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1))
plt.show()

print 'gal dim:',raw_gal.shape
print 'sky dim:',sky.shape
plt.figure()
plt.imshow(np.log(d2_spectra_s.T-sky_sub_tot),aspect=35,origin='lower')#aspect=35,
plt.show()
plt.figure()
plt.plot(np.arange(raw_gal.shape[1]),np.sum(raw_gal-sky_sub,axis=0),'b-')
plt.plot(np.arange(raw_gal.shape[1]),np.median(sky,axis=0),'r-')
plt.show()

plt.figure(); 
plt.subplot(311); 
plt.imshow(np.log(d2_spectra_s.T),aspect=35,origin='lower');
plt.axhline(lower_gal,color='k',ls='--'); 
plt.axhline(upper_gal,color='k',ls='--'); 
plt.subplot(312); 
plt.imshow(np.log(sky_sub_tot),aspect=35,origin='lower'); 
plt.subplot(313); 
plt.imshow(np.log(d2_spectra_s.T-sky_sub_tot),aspect=35,origin='lower'); 
plt.axhline(lower_gal,color='k',ls='--'); 
plt.axhline(upper_gal,color='k',ls='--'); 
plt.show()