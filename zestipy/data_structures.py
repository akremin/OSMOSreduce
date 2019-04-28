# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:02:56 2015

@author: kremin
"""

import numpy as np
import scipy.signal as signal
import os
from astropy.io import fits as pyfits
from astropy.table import Table


class waveform:
    def __init__(self,wave,flux,name,mask=None):
        if mask is None or len(mask)!= flux.size:
            self.mask = np.zeros(flux.size).astype(bool)
        else:
            self.mask = mask
        self.flux = flux
        self.wave = wave
        self.masked_flux = flux[np.bitwise_not(self.mask)]
        self.masked_wave = wave[np.bitwise_not(self.mask)]
        self.continuum_subtracted_flux = self.__continuum_subtract()
        self.min_lam = np.min(wave)
        self.max_lam = np.max(wave)
        self.name = name


    def __continuum_subtract(self):
        return self.masked_flux - signal.medfilt(self.masked_flux,171)




def smooth_waveform(in_waveform):      
    smoothing_number = 10
    smoothed_flux = signal.convolve(in_waveform.flux,signal.boxcar(smoothing_number),'same')
    cut_smoothed_flux = smoothed_flux[int(smoothing_number/2):-int(smoothing_number/2)]
    cut_wave = in_waveform.wave[int(smoothing_number/2):-int(smoothing_number/2)]
    # Mask negative values and skip spectra if something goes wrong               
    if len(smoothed_flux)==0:
        print("The length of smoothed_flux was 0 so skipping\n")
        return in_waveform
    return waveform(cut_wave,cut_smoothed_flux,in_waveform.name)

def mask_inf_regions(in_waveform):
    '''Mask Infinite values from the array'''
    flux = in_waveform.flux
    wave = in_waveform.wave
    mask = np.isinf(flux)
    flux[mask] = 0
    return waveform(wave=wave,flux=flux,name=in_waveform.name)
    
def mask_neg_regions(in_waveform):
    '''Mask Negative values from the array (fluxes are positive definite).'''
    in_flux = in_waveform.flux
    in_wave = in_waveform.wave
    mask2 = np.where(in_flux>=0)[0]
    new_flux = in_flux[mask2]
    new_wave = in_wave[mask2]
    return waveform(wave=new_wave,flux=new_flux,name=in_waveform.name) 
     

        
def load_sdss_templatefiles(path_to_files='.',filenames=['spDR2-023.fit']):
    if type(filenames)==str:
        filenames = [filenames]
        
    waveforms = []
    for template_file in filenames:
        try:
            early_type = pyfits.open(os.path.join(path_to_files,template_file))
        except IOError:
            print("There is no '%s' in that path" % (template_file))
            print("path given =", path_to_files)
            raise IOError
            
        # Declare the array for the template flux(es)  
        coeff0 = float(early_type[0].header['COEFF0'])
        coeff1 = float(early_type[0].header['COEFF1'])
        final_z = float(early_type[0].header['Z'])
        flux=early_type[0].data[0]

        wave=10**(coeff0 + coeff1*np.arange(0.,flux.size,1.))
        wave = wave/(1.0+final_z)

        name = (template_file.split('.fit')[0]).replace('spDR2-','')
        early_type.close()
        waveforms.append(waveform(wave=wave,flux=flux,name=name))
    return waveforms
        
def generate_listof_waveforms(waves,fluxes,names):
    waveforms = []
    for i in np.arange(names.size):
        wave = waves[i]
        flux = fluxes[i]
        name = names[i]
        waveforms.append(waveform(wave=wave,flux=flux,name=name))
    return waveforms
        
        
        
        
        
        
class trivial_class:
    def __init__(self):
        pass
        
        
class redshift_data:
    def __init__(self,redshift_est=0.,cor=0.,ztest=np.zeros((1,)),corr_val=np.zeros((1,)),template=trivial_class(),qualityval=None):
        self.best_zest = redshift_est
        self.max_cor = cor
        self.ztest_vals = ztest
        self.corr_vals = corr_val
        self.template_name = template
        if qualityval != None:
            self.qualityval = qualityval
        
        
    
