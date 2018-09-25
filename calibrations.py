#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:36:09 2017

@author: kremin
"""
import numpy as np
from astropy.table import Table
from scipy.optimize import curve_fit
##  a zoom in window
##  mutlicursor
## And checkboxes
## and radio buttons
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider, Button
from scipy.signal import medfilt
from scipy import signal
from scipy.interpolate import interp1d
from linebrowser import LineBrowser

import numpy as np
import os
import re
from interactive_plot import interactive_plot,pix_to_wave
from scipy.signal import argrelmax


def air_to_vacuum(airwl, nouvconv=True):
    """
    Returns vacuum wavelength of the provided air wavelength array or scalar.
    Good to ~ .0005 angstroms.

    If nouvconv is True, does nothing for air wavelength < 2000 angstroms.

    Input must be in angstroms.

    Adapted from idlutils airtovac.pro, based on the IAU standard
    for conversion in Morton (1991 Ap.J. Suppl. 77, 119)
    """
    airwl = np.array(airwl, copy=False, dtype=float, ndmin=1)
    isscal = airwl.shape == tuple()
    if isscal:
        airwl = airwl.ravel()

    # wavenumber squared
    sig2 = (1e4 / airwl) ** 2

    convfact = 1. + 6.4328e-5 + 2.94981e-2 / (146. - sig2) + 2.5540e-4 / (41. - sig2)
    newwl = airwl.copy()
    if nouvconv:
        convmask = newwl >= 2000
        newwl[convmask] *= convfact[convmask]
    else:
        newwl[:] *= convfact
    return newwl[0] if isscal else newwl



def load_calibration_lines_dict(cal_lamp):
    linelistdict = {}
    print(('Using calibration lamps: ', cal_lamp))
    possibilities = ['Xe', 'Ar', 'HgNe', 'Hg', 'Ne', 'ThAr', 'Th']
    for lamp in possibilities:
        if lamp in cal_lamp:
            wm, fm = np.loadtxt('./lamp_linelists/osmos_{}.dat'.format(lamp), usecols=(0, 2), unpack=True)
            wm_vac = air_to_vacuum(wm)
            ## sort lines by wavelength
            sortd = np.argsort(wm_vac)
            srt_wm_vac, srt_fm = wm[sortd], fm[sortd]
            linelistdict[lamp] = (srt_wm_vac, srt_fm)

    cal_states = {
        'Xe': ('Xe' in cal_lamp), 'Ar': ('Ar' in cal_lamp), \
        'HgNe': ('HgNe' in cal_lamp), 'Ne': ('Ne' in cal_lamp)
    }

    return linelistdict, cal_states


def load_calibration_lines_salt_dict(cal_lamp,wavemincut=4000,wavemaxcut=10000):
    linelistdict = {}
    print(('Using calibration lamps: ', cal_lamp))
    possibilities = ['Xe','Ar','HgNe','HgAr','NeAr','Hg','Ne','ThAr','Th']
    for lamp in possibilities:
        if lamp in cal_lamp:
            print(lamp)
            tab = Table.read('./lamp_linelists/salt/{}.txt'.format(lamp), \
                               format='ascii.csv',)
            fm = tab['Intensity'].data
            wm_vac = air_to_vacuum(tab['Wavelength'].data)

            ## sort lines by wavelength
            sortd = np.argsort(wm_vac)
            srt_wm_vac, srt_fm = wm_vac[sortd], fm[sortd]
            good_waves = np.where((srt_wm_vac>=wavemincut)&(srt_wm_vac<=wavemaxcut))[0]
            out_wm_vac,out_fm_vac = srt_wm_vac[good_waves], srt_fm[good_waves]
            linelistdict[lamp] = (out_wm_vac,out_fm_vac)

    return linelistdict

def load_calibration_lines_NIST_dict(cal_lamp,wavemincut=4000,wavemaxcut=10000):
    lineuncertainty = 0.002
    linelistdict = {}
    print(('Using calibration lamps: ', cal_lamp))
    possibilities = ['Ar','He','Hg','Ne','ThAr','Th','Xe']
    for lamp in cal_lamp:
        if lamp in possibilities:
            print(lamp)
            if lamp is 'ThAr':
                tab = Table.read('./lamp_linelists/NIST/{}.txt'.format(lamp),
                                 format='ascii.csv', header_start=8, data_start=9)
                ## Quality Control
                tab = tab[tab['Obs_Unc'] <= lineuncertainty]
                ## Remove non-ThAr lines
                #tab = tab[tab['Name'] != 'NoID']
                tab = tab[tab['Name'] != 'Unkwn']
                ## Get wavelength and frequency information
                fm = tab.filled()['Rel_Intensity'].data
                wm_vac = tab.filled()['Obs_Wave'].data
            else:
                tab = Table.read('./lamp_linelists/NIST/{}.txt'.format(lamp),\
                                 format='ascii.csv', header_start=5, data_start=6)
                ## Quality Control the lines
                #tab = tab[tab['Flag'].mask]
                names = np.unique(tab['Name'])
                if 'I' in names[0]:
                    selection = names[0].split('I')[0]
                else:
                    selection = names[0].split('I')[0]
                name = selection+'I'
                tab = tab[tab['Name']==name]
                tab = tab[np.bitwise_not(tab['Rel_Intensity'].mask)]
                tab = tab[np.bitwise_not(tab['Ritz_Wave'].mask)]
                tab['Obs_Unc'].fill_value = 999.
                tab['Ritz_Unc'].fill_value = 999.
                tab['Obs-Ritz'].fill_value = 999.

                for col in ['Obs_Unc','Ritz_Unc','Obs-Ritz']:
                    tab = tab[(tab[col].filled().data.astype(float)<=lineuncertainty)]

                if np.all(tab['Calibd_Intensity'].mask):
                    fm = tab['Rel_Intensity'].filled().data
                else:
                    tab['Calib_Conf'].fill_value = 'E-'
                    tab['Calibd_Intensity'].fill_value = -999
                    calibd = tab[np.bitwise_not(tab['Calibd_Intensity'].mask)].filled()
                    calibs, rel = [],[]
                    for grade in ['A','AA','A+','A-','B+','BB','B']:
                        boolean = np.where(calibd['Calib_Conf']==grade)[0]
                        calibs.extend(calibd['Calibd_Intensity'][boolean].data.tolist())
                        rel.extend(calibd['Rel_Intensity'][boolean].data.tolist())
                    if len(calibs)>0:
                        ratios = np.array(calibs).astype(float)/np.array(rel).astype(float)
                        fm = tab.filled()['Rel_Intensity'].data.astype(float)*np.median(ratios)
                    else:
                        fm = tab['Rel_Intensity'].filled().data
                ## NIST values from 2000A to 10000A are in air wavelenghts
                ## This function only converts wavelengths in that range to vacuum
                ## assumes the rest are already in vacuum
                waves_nm = tab.filled()['Obs_Wave'].data
                waves_ang = 10*waves_nm
                wm_vac = air_to_vacuum(waves_ang)

            ## sort lines by wavelength
            sortd = np.argsort(wm_vac)
            srt_wm_vac, srt_fm = wm_vac[sortd], fm[sortd]
            good_waves = np.where((srt_wm_vac>=wavemincut)&(srt_wm_vac<=wavemaxcut))[0]
            out_wm_vac,out_fm_vac = srt_wm_vac[good_waves], srt_fm[good_waves]
            linelistdict[lamp] = (out_wm_vac,out_fm_vac)

    return linelistdict


if __name__ == '__main__':
    cal_lamp = ['Hg', 'Argon', 'Neon','Xenon']
    linelistdict, cal_states = load_calibration_lines_salt_dict(cal_lamp)