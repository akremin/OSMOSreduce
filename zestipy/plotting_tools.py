# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:13:31 2015

@author: kremin
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import scipy.signal as signal
import pdb

def plot_skylines(axi,red_est):
    '''Take an axis object and plot the location of skylines
    as well as emission and absorption lines at a redshift of red_est.'''
    # Edited to include sdss lines
    # Converting to values used in code     SDSS Wave /1.00027 = these waves
    # Red = HK
    # Purple = OII, Halpha
    # Black = sky
    # Blue = Emission
    # Orange = Absorption\
    emission_lines = np.array([2798.4,4101.8,4340.5,4861.4,4959.,5006.9,6548.1,
                      6583.5,6716.5,6730.9])*(1.+red_est)

    sky_lines = [5577.,5893.,6300.,7244.,7913.7,8344.6,8827.1]        

    absorption_lines = np.array([4304.4,5175.3,5894.,8498.1,8542.1,8662.2])*(1.+red_est)

    pspecs = [] #np.ndarray((4+len(emission_lines)+len(absorption_lines),))
    pspecs.append(axi.axvline(3726.1*(1.+red_est),ls='--',alpha=0.7,c='purple'))
    pspecs.append(axi.axvline(6562.8*(1.+red_est),ls='--',alpha=0.7,c='purple'))
    pspecs.append(axi.axvline(3933.7*(1.+red_est),ls='--',alpha=0.7,c='red'))
    pspecs.append(axi.axvline(3968.5*(1.+red_est),ls='--',alpha=0.7,c='red'))
    
    for vlin in emission_lines:
        pspecs.append(axi.axvline(vlin,ls='--',alpha=0.7,c='blue'))

    for vlin in sky_lines:    
        axi.axvline(vlin,ls='-.',alpha=0.7,c='black')

    for vlin in absorption_lines:
        pspecs.append(axi.axvline(vlin,ls='--',alpha=0.7,c='orange'))

    return pspecs
    
    
    
def summary_plot(waves, flux, templ_waves, template,zest,z_test,corrs,plt_name,frame_name,mock_photoz=None):
    '''Display the spectrum and reference lines for the best fitting redshift.'''
    cont_subd_flux = flux - signal.medfilt(flux,171)
    cont_subd_temp_flux = template - signal.medfilt(template,171)
    cont_subd_flux = cont_subd_flux/np.std(cont_subd_flux)
    cont_subd_temp_flux = cont_subd_temp_flux/np.std(cont_subd_temp_flux)
    temp_shifted_waves = templ_waves*(1+zest)
    plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])
    plt.tight_layout()
    ax = plt.subplot(gs[0])
    plt.subplots_adjust(bottom=0.1)
    #pdb.set_trace()
    alp = 0.5
    ax.plot(waves,flux,label='Target {}'.format(frame_name),alpha=alp+0.1)
    tl,th = np.nanquantile(template,[0.2,0.8])
    fl,fh = np.nanquantile(flux,[0.2,0.8])
    modtemplate = 0.75*(fh-fl)*template/(th-tl)
    if np.nanmax(modtemplate)>np.nanmax(flux):
        modtemplate = modtemplate*0.5
    ax.plot(temp_shifted_waves,modtemplate,alpha=alp,label='SDSS Template')
    ax.set_xlim(waves[0],waves[-1])
    last_ind = np.max(np.where(temp_shifted_waves<waves[-1]))
    shortnd_temp_flux = cont_subd_temp_flux[:last_ind]
    if len(shortnd_temp_flux)>0:
        ax.set_ylim(np.min([np.nanmin(flux),np.nanmin(modtemplate)]),\
                 np.max([np.nanmax(flux),np.nanmax(modtemplate)]))
    else:
       ax.set_ylim(np.nanmin(flux),np.nanmax(flux))

    plot_skylines(ax,zest)

    ax.set_xlabel('Wavelength (A)',fontsize=12)
    ax.set_ylabel('Flux [Arbitrary]',fontsize=12)
    ax.legend(loc='best')
    title = 'Target {}'.format(frame_name)
    if mock_photoz:
        title += " photoz=%0.3f" % mock_photoz
    ax.set_title(title,fontsize=16)

    ax2 = plt.subplot(gs[1])
    plt.subplots_adjust(bottom=0.1)
    #pdb.set_trace()
    alp = 0.5
    ax2.plot(waves,cont_subd_flux,label='Target {}'.format(frame_name),alpha=alp+0.1)
    ax2.plot(temp_shifted_waves,(cont_subd_temp_flux),alpha=alp,label='SDSS Template')
    ax2.set_xlim(waves[0],waves[-1])
    last_ind = np.max(np.where(temp_shifted_waves<waves[-1]))
    shortnd_temp_flux = cont_subd_temp_flux[:last_ind]
    if len(shortnd_temp_flux)>0:
        ax2.set_ylim(np.min([cont_subd_flux.min(),shortnd_temp_flux.min()]),\
                 np.nanmax([np.nanmax(cont_subd_flux),np.nanmax(shortnd_temp_flux)]))
    else: 
       ax2.set_ylim(cont_subd_flux.min(),cont_subd_flux.max())

    plot_skylines(ax2,zest)

    ax2.set_xlabel('Wavelength (A)',fontsize=12)
    ax2.set_ylabel('Cont. Subd. Flux/std(Flux)',fontsize=12)
    # ax2.legend(loc='best')
    # title = 'Frame %s' % frame_name
    if mock_photoz:
        title += " photoz=%0.3f" % mock_photoz
    # ax2.set_title(title)
    
    ax3 = plt.subplot(gs[2])
    plt.subplots_adjust(bottom=0.1)
    ax3.plot(z_test,corrs,'b')

    ax3.axvline(zest,color='k',ls='--',label=r'$z_{est}  $'+' =\t{:0.5f}\n'.format(zest)+r'$r_{p,max}$'+' =\t{:0.3f}'.format(np.nanmax(corrs)))
    ax3.legend(loc='best')
    ax3.set_xlabel('Redshift',fontsize=12)
    ax3.set_ylabel('Correlation',fontsize=12)
    plt.savefig(plt_name,dpi=600,overwrite=True)
    plt.close()