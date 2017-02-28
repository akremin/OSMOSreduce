#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:36:09 2017

@author: kremin
"""
import numpy as np
from testopt import air_to_vacuum

def load_calibration_lines(cal_lamp):
    wm = []
    fm = []
    
    print(('Using calibration lamps: ', cal_lamp))
    
    if 'Xenon' in cal_lamp:
        wm_Xe,fm_Xe = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
        wm_Xe = air_to_vacuum(wm_Xe)
        wm.extend(wm_Xe)
        fm.extend(fm_Xe)
    if 'Argon' in cal_lamp:
        wm_Ar,fm_Ar = np.loadtxt('osmos_Argon.dat',usecols=(0,2),unpack=True)
        wm_Ar = air_to_vacuum(wm_Ar)
        wm.extend(wm_Ar)
        fm.extend(fm_Ar)
    if 'HgNe' in cal_lamp:
        wm_HgNe,fm_HgNe = np.loadtxt('osmos_HgNe.dat',usecols=(0,2),unpack=True)
        wm_HgNe = air_to_vacuum(wm_HgNe)
        wm.extend(wm_HgNe)
        fm.extend(fm_HgNe)
    if 'Neon' in cal_lamp:
        wm_Ne,fm_Ne = np.loadtxt('osmos_Ne.dat',usecols=(0,2),unpack=True)
        wm_Ne = air_to_vacuum(wm_Ne)
        wm.extend(wm_Ne)
        fm.extend(fm_Ne)
    
    fm = np.array(fm)[np.argsort(wm)]
    wm = np.array(wm)[np.argsort(wm)]
    
    
    cal_states = {'Xe':('Xenon' in cal_lamp),'Ar':('Argon' in cal_lamp),\
    'HgNe':('HgNe' in cal_lamp),'Ne':('Neon' in cal_lamp)}
    
    return wm,fm,cal_states