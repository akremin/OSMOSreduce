#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  1 23:10:11 2017

@author: kremin
"""

start_up=True
if start_up:
    import pickle as pkl
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit
    import os
    if 'HOSTNAME' in list(os.environ.keys()) and os.environ['HOSTNAME'] == 'umdes7.physics.lsa.umich.edu':
        data_dir =  'goodman_jan17'# 
    else:
        data_dir = 'SOAR_data'

    with open('../../'+data_dir+'/Kremin10/Kremin10_reduced_spectra.pkl','rb') as pklfil:  #SOAR_data/Kremin10/Kremin10_reduced_spectra.pkl'
        specdict = pkl.load(pklfil)
    dict1 = specdict['0']
    arcspec1 = dict1['arc_spec']
    galspec1 = dict1['gal_spec']
    scispec1 = dict1['science_spec']
    gcut1,gcut2 = dict1['gal_cuts']
    gcut1 = 6
    gcut2 = 22
    slit_width = 26
    lower_gal = 3
    upper_gal = 19
    slit_width = 23
    d2_spectra_s = (scispec1.T)[:,3:]
    d2_spectra_a = (arcspec1.T)[:,3:]
    raw_gal = d2_spectra_s.T[lower_gal:upper_gal,:]
    sky = np.append(d2_spectra_s.T[:lower_gal,:],d2_spectra_s.T[upper_gal:,:],axis=0)
    sky_sub = np.zeros(raw_gal.shape) + np.median(sky,axis=0)
    sky_sub_tot = np.zeros(d2_spectra_s.T.shape) + np.median(sky,axis=0)
    lowerpix = 500
    upperpix = -20

def _fullquadfit(dx,a,b,c):
    '''define quadratic galaxy fitting function'''
    return a*dx*dx + b*dx + c

def _gaus(x,amp,sigma,x0,background):
    if amp <= 0: amp = np.inf
    # sig = 4.0
    return amp*np.exp(-(x-x0)**2/(2*sigma**2)) + background

def _constrained_gaus(dx_ov_sig,amp,background):
    if amp <= 0: amp = np.inf
    # sig = 4.0
    return amp*np.exp(-0.5*dx_ov_sig*dx_ov_sig) + background


ncols = d2_spectra_s.shape[0]

normdspect = d2_spectra_s/np.max(d2_spectra_s)
gal_guess = np.arange(0,slit_width,1)[np.median(normdspect,axis=0)== \
                                    np.max(np.median(normdspect,axis=0))][0]
gal_amp,gal_pos,gal_wid,sky_val = 1,4.0,gal_guess,np.min(d2_spectra_s[0,:])
yvals = np.arange(0,slit_width,1)
cut_xvals = np.arange(lowerpix,ncols+upperpix)
cutncols = len(cut_xvals)
galamps = np.zeros(cutncols)
galposs = np.zeros(cutncols)
galwids = np.zeros(cutncols)
skyvals = np.zeros(cutncols)
badpos = []
badwid = []
for i,col in enumerate(cut_xvals):
    try:
        popt_g,pcov_g = curve_fit(_gaus,yvals,d2_spectra_s[col,:],p0=[1,4.0,gal_guess,np.min(d2_spectra_s[col,:])],maxfev = 100000)
        popt_g[1] = np.abs(popt_g[1])
        #gal_amp,sky_val = popt_g[0],popt_g[3]
        if popt_g[2] < 2*slit_width and popt_g[2] > -slit_width:     #slit_width  0
            gal_pos = popt_g[2]
        else:
            badpos.append(col)
        # else use value from previous index
        if popt_g[1] < 2*slit_width:
            gal_wid = popt_g[1]
        else:
            badwid.append(col)
        # else use value from previous index
    except:
        # if something breaks, implicitly use the previous iterations fit values for this index
        print(i)
    #print popt_g
    galamps[i],galwids[i],galposs[i],skyvals[i] = gal_amp,gal_wid,gal_pos,sky_val


galwids_fitparams,pcov = curve_fit(_fullquadfit,cut_xvals,galwids,p0=[1e-4,1e-4,1e-4],maxfev = 100000)
#fitd_galposs = _fullquadfit(cut_xvals,*galposs_fitparams)
cutxmask = np.ones(len(cut_xvals)).astype(bool)

for i in range(10):
    tempfitd_galwids = _fullquadfit(cut_xvals[cutxmask],*galwids_fitparams)
    dgalwids = tempfitd_galwids-galwids[cutxmask]
    deviants_mask = np.where(np.abs(dgalwids)>3*np.std(dgalwids))
    cutxmask[deviants_mask] = False   
    galwids_fitparams,pcov = curve_fit(_fullquadfit,cut_xvals[cutxmask],galwids[cutxmask],p0=[1e-4,1e-4,1e-4],maxfev = 100000)
    

galposs_fitparams,pcov = curve_fit(_fullquadfit,cut_xvals[cutxmask],galposs[cutxmask],p0=[1e-4,1e-4,1e-4],maxfev = 100000)
xvals = np.arange(ncols)
fitd_galwids = _fullquadfit(xvals,*galwids_fitparams)
fitd_galposs = _fullquadfit(xvals,*galposs_fitparams)

naivegalflux = np.zeros(ncols)
naiveskyflux = np.zeros(ncols)
fitgalflux = np.zeros(ncols)
fitskyflux = np.zeros(ncols)
fitgalamps = np.zeros(ncols)
fitskyamps = np.zeros(ncols)
totalflux = np.sum(d2_spectra_s,axis=1)
bad_xvals = cut_xvals[~cutxmask]

for i in xvals:
    try:
        dy_over_sigmas = (yvals-fitd_galposs[i])/fitd_galwids[i]
        popt_cg,pcov_cg = curve_fit(_constrained_gaus,dy_over_sigmas,d2_spectra_s[i,:],p0=[1,np.min(d2_spectra_s[i,:])],maxfev = 100000)
        gal_amp,sky_val = popt_cg
    except:
        # if something breaks, implicitly use the previous iterations fit values for this index
        print(i)
    print(popt_cg)
    fitgalamps[i] = gal_amp
    fitskyamps[i] = sky_val
    naiveskyflux[i] = slit_width*sky_val
    naivegalflux[i] = totalflux[i] - naiveskyflux[i]
    constraind_guasfit = _constrained_gaus(dy_over_sigmas,gal_amp,0.)
    fitgalflux[i] = np.sum(constraind_guasfit)
    fitskyflux[i] = totalflux[i] - fitgalflux[i]
    if i in bad_xvals:
        plt.figure()
        plt.plot(yvals,constraind_guasfit,'b-')
        plt.plot(yvals,constraind_guasfit+sky_val,'y-')
        plt.plot(yvals,d2_spectra_s[i,:],'g-')
        plt.title('Column i='+str(i)+' bad fit')
        plt.show()
        

tempfitd_galwids = _fullquadfit(cut_xvals[cutxmask],*galwids_fitparams)
dgalwids = tempfitd_galwids-galwids[cutxmask]
deviants_mask = np.where(np.abs(dgalwids)>3*np.std(dgalwids))
cutxmask[deviants_mask] = False 
good_xvals = cut_xvals[cutxmask]
fullxmask = good_xvals # since xvals is just 0 to ncols, values are same as index
#xvalmask = np.zeros(ncols).astype(bool)
#good_xvals = xvals[goodcol_inds]



plt.figure()
plt.subplot(211)
plt.title('Sky  b = fitted  r = naive')
plt.plot(xvals,fitskyflux,'b-')
plt.plot(xvals,naiveskyflux,'r-')
plt.subplot(212)
plt.title('Galaxies  b = fitted  r = naive')
plt.plot(good_xvals,fitgalflux[fullxmask],'b-')
plt.plot(good_xvals,naivegalflux[fullxmask],'r-')
plt.show()
plt.figure()
plt.plot(good_xvals,fitgalflux[fullxmask],'b-',alpha=0.4,label='Fitted')
plt.plot(xvals,fitskyflux,'b-.',alpha=0.4)
plt.plot(xvals,naiveskyflux,'r-.',alpha=0.4,label='Naive')
plt.plot(good_xvals,naivegalflux[fullxmask],'r-',alpha=0.4)
plt.plot(xvals,np.sum(raw_gal-sky_sub,axis=0),'g-',alpha=0.4,label='Dans')
plt.plot(xvals,np.median(sky,axis=0)*slit_width,'g-.',alpha=0.4)
plt.legend(loc='best')
plt.show()
plt.figure()
plt.plot(xvals,fitd_galposs);
plt.plot(good_xvals,galposs[cutxmask])
plt.plot(cut_xvals,galposs)
plt.show()
plt.figure()
plt.plot(xvals,fitd_galwids)
plt.plot(good_xvals,galwids[cutxmask])
plt.plot(cut_xvals,galwids)
plt.show()

fitd_galposs
fitd_galwids
fitgalamps
fitskyamps

