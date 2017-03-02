#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 23:10:11 2017

@author: kremin
"""

import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from slit_find import _gaus
with open('../../SOAR_data/Kremin10/Kremin10_reduced_spectra.pkl','rb') as pklfil:
    specdict = pkl.load(pklfil)
dict1 = specdict['0']
arcspec1 = dict1['arc_spec']
galspec1 = dict1['gal_spec']
scispec1 = dict1['science_spec']
#gcut1,gcut2 = dict1['gal_cuts']
#gcut1 = 6
#gcut2 = 22
#slit_width = 26
lower_gal = 3
upper_gal = 19
slit_width = 23
d2_spectra_s = (scispec1.T)[:,3:]
d2_spectra_a = (arcspec1.T)[:,3:]
raw_gal = d2_spectra_s.T[lower_gal:upper_gal,:]
sky = np.append(d2_spectra_s.T[:lower_gal,:],d2_spectra_s.T[upper_gal:,:],axis=0)
sky_sub = np.zeros(raw_gal.shape) + np.median(sky,axis=0)
sky_sub_tot = np.zeros(d2_spectra_s.T.shape) + np.median(sky,axis=0)

ncols = d2_spectra_s.shape[0]
naivegalflux = np.zeros(ncols)
naiveskyflux = np.zeros(ncols)
fitgalflux = np.zeros(ncols)
fitskyflux = np.zeros(ncols)
normdspect = d2_spectra_s/np.max(d2_spectra_s)
totalflux = np.sum(d2_spectra_s,axis=1)

gal_guess = np.arange(0,slit_width,1)[np.median(normdspect,axis=0)== \
                                    np.max(np.median(normdspect,axis=0))][0]
gal_amp,gal_pos,gal_wid,sky_val = 1,4.0,gal_guess,np.min(d2_spectra_s[0,:])
yvals = np.arange(0,slit_width,1)
for i in np.arange(ncols):
    try:
        popt_g,pcov_g = curve_fit(_gaus,yvals,d2_spectra_s[i,:],p0=[1,4.0,gal_guess,np.min(d2_spectra_s[i,:])],maxfev = 100000)
        gal_amp,gal_wid,gal_pos,sky_val = popt_g
    except:
        # if something breaks, implicitly use the previous iterations fit values for this index
        print i
    print popt_g
    naiveskyflux[i] = slit_width*sky_val
    naivegalflux[i] = totalflux[i] - naiveskyflux[i]
    fitgalflux[i] = np.sum(_gaus(yvals,gal_amp,gal_wid,gal_pos,0.))
    fitskyflux[i] = totalflux[i] - fitgalflux[i]
    
plt.figure()
plt.subplot(211)
plt.title('Sky  b = fitted  r = naive')
plt.plot(np.arange(ncols),fitskyflux,'b-')
plt.plot(np.arange(ncols),naiveskyflux,'r-')
plt.subplot(212)
plt.title('Galaxies  b = fitted  r = naive')
plt.plot(np.arange(ncols),fitgalflux,'b-')
plt.plot(np.arange(ncols),naivegalflux,'r-')
plt.show()
plt.figure()
plt.plot(np.arange(ncols),fitgalflux,'b-',alpha=0.4,label='Fitted')
plt.plot(np.arange(ncols),fitskyflux,'b-.',alpha=0.4)
plt.plot(np.arange(ncols),naiveskyflux,'r-',alpha=0.4,label='Naive')
plt.plot(np.arange(ncols),naivegalflux,'r-.',alpha=0.4)
plt.plot(np.arange(ncols),np.sum(raw_gal-sky_sub,axis=0),'g-',alpha=0.4,label='Dans')
plt.plot(np.arange(ncols),np.median(sky,axis=0),'g-.',alpha=0.4)
plt.legend(loc='best')
plt.show()