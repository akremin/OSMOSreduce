#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 17:36:09 2017

@author: kremin
"""
import numpy as np
from testopt import air_to_vacuum
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


def save_calib_dict(calib_coef_dict,fittype,cam,config,filenum,timestamp):
    calib_tab = Table(calib_coef_dict)
    calib_tab.write('calibration_{}_{}_{}_{}_{}.fits'.format(fittype,cam,config,filenum,timestamp),format='fits',overwrite=True)

def load_calib_dict(fittype,cam,config,filenum,timestamp):
    calib_tab = Table.read('calibration_{}_{}_{}_{}_{}.fits'.format(fittype,cam,config,filenum,timestamp),format='fits')
    return calib_tab

def locate_calib_dict(fileloc,fittype, camera, config, filenum):
    calib_coef_table = None
    match_str = 'calibration_{}_{}_{}_{}_'.format(fittype, camera, config, filenum)
    match_str += r'(\d{5}).fits'
    files = os.listdir(fileloc)
    matches = []
    for fil in files:
        srch_res = re.search(match_str, fil)
        if srch_res:
            matches.append(int(srch_res.group(1)))
        else:
            continue

    if len(matches) > 0:
        print(matches)
        newest = max(matches)
        calib_coef_table = load_calib_dict(fittype, camera, config, filenum, newest)
    return calib_coef_table





def load_calibration_lines(cal_lamp):
    wm = []
    fm = []
    
    print(('Using calibration lamps: ', cal_lamp))
    
    if 'Xenon' in cal_lamp:
        wm_Xe,fm_Xe = np.loadtxt('./lamp_linelists/osmos_Xenon.dat',usecols=(0,2),unpack=True)
        wm_Xe = air_to_vacuum(wm_Xe)
        wm.extend(wm_Xe)
        fm.extend(fm_Xe)
    if 'Argon' in cal_lamp:
        wm_Ar,fm_Ar = np.loadtxt('./lamp_linelists/osmos_Argon.dat',usecols=(0,2),unpack=True)
        wm_Ar = air_to_vacuum(wm_Ar)
        wm.extend(wm_Ar)
        fm.extend(fm_Ar)
    if 'HgNe' in cal_lamp:
        wm_HgNe,fm_HgNe = np.loadtxt('./lamp_linelists/osmos_HgNe.dat',usecols=(0,2),unpack=True)
        wm_HgNe = air_to_vacuum(wm_HgNe)
        wm.extend(wm_HgNe)
        fm.extend(fm_HgNe)
    if 'Neon' in cal_lamp:
        wm_Ne,fm_Ne = np.loadtxt('./lamp_linelists/osmos_Neon.dat',usecols=(0,2),unpack=True)
        wm_Ne = air_to_vacuum(wm_Ne)
        wm.extend(wm_Ne)
        fm.extend(fm_Ne)
    
    fm = np.array(fm)[np.argsort(wm)]
    wm = np.array(wm)[np.argsort(wm)]

    cal_states = {'Xe':('Xenon' in cal_lamp),'Ar':('Argon' in cal_lamp),\
    'HgNe':('HgNe' in cal_lamp),'Ne':('Neon' in cal_lamp)}
    
    return wm,fm,cal_states


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



def load_calibration_lines_crc_dict(cal_lamp,wavemincut=4000,wavemaxcut=10000):
    linelistdict = {}
    print(('Using calibration lamps: ', cal_lamp))
    possibilities = ['Xe', 'Ar', 'HgNe', 'Hg', 'Ne', 'ThAr', 'Th']
    for lamp in possibilities:
        if lamp in cal_lamp:
            print(lamp)
            tab = Table.read('./lamp_linelists/linelists_crc/{}.txt'.format(lamp), \
                               format='ascii.csv')
            fm = tab['Intensity'].data
            wm_vac = air_to_vacuum(tab['Wavelength Ang'].data)

            ## sort lines by wavelength
            sortd = np.argsort(wm_vac)
            srt_wm_vac, srt_fm = wm_vac[sortd], fm[sortd]
            good_waves = np.where((srt_wm_vac>=wavemincut)&(srt_wm_vac<=wavemaxcut))[0]
            out_wm_vac,out_fm_vac = srt_wm_vac[good_waves], srt_fm[good_waves]
            linelistdict[lamp] = (out_wm_vac,out_fm_vac)

    return linelistdict



# def load_calibration_lines_nist_dict(cal_lamp,wavemincut=4000,wavemaxcut=10000):
#     linelistdict = {}
#     print(('Using calibration lamps: ', cal_lamp))
#     possibilities = ['Xe','Ar','HgNe','Hg','Ne','ThAr','Th']
#     for lamp in possibilities:
#         if lamp in cal_lamp:
#             print(lamp)
#             tab = Table.read('./lamp_linelists/others/nist_github_lines/{}_NIST_air.txt'.format(lamp), \
#                                format='ascii.basic',data_start=0,colnames=('Wavelength Ang','Intensity'))
#             fm = tab['Intensity'].data
#             wm_vac = air_to_vacuum(tab['Wavelength Ang'].data)
#             ## sort lines by wavelength
#             sortd = np.argsort(wm_vac)
#             srt_wm_vac, srt_fm = wm_vac[sortd], fm[sortd]
#             good_waves = np.where((srt_wm_vac>=wavemincut)&(srt_wm_vac<=wavemaxcut))[0]
#             out_wm_vac,out_fm_vac = srt_wm_vac[good_waves], srt_fm[good_waves]
#             linelistdict[lamp] = (out_wm_vac,out_fm_vac)
#
#     return linelistdict, cal_states



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





def aperature_number_pixoffset(fibnum):
    if type(fibnum) is str:
        strpd_fibnum = fibnum.strip('rb')
        if strpd_fibnum.isnumeric():
            tet = np.float64(strpd_fibnum[0]) - 1.
            fib = np.float64(strpd_fibnum[1:]) - 1.
        else:
            return 0.
    elif np.isscalar(fibnum):
        tet = fibnum // 16
        fib = fibnum % 16

    c1 = 1.023
    c2 = 54.058
    c3 = -6.962
    c4 = 1.985
    c5 = -0.5560
    return c1 + c2 * tet + c3 * tet * tet + c4 * fib + c5 * tet * fib


def aperature_pixoffset_between2(fibnum):
    if type(fibnum) is str:
        strpd_fibnum = fibnum.strip('rb')
        if strpd_fibnum.isnumeric():
            tet = np.float64(strpd_fibnum[0]) - 1.
            fib = np.float64(strpd_fibnum[1:]) - 1.
        else:
            return 0.
        fibnum = 16*tet+fib
    elif not np.isscalar(fibnum):
        return 0.
    else:
        pass

    if fibnum == 0:
        return 0.
    else:
        return aperature_number_pixoffset(fibnum)-aperature_number_pixoffset(fibnum-1)


from interactive_plot import interactive_plot,pix_to_wave
from scipy.signal import argrelmax
def top_peak_wavelengths(pixels,spectra,coefs):
    top_pixels = top_peak_pixels(pixels, spectra)
    max_flux_wavelengths = pix_to_wave(top_pixels,coefs)
    return max_flux_wavelengths

def top_peak_pixels(pixels,spectra):
    max_locs = argrelmax(spectra)[0]
    max_vals = spectra[max_locs]
    sorted_max_val_inds = np.argsort(max_vals).astype(int)
    top_max_val_inds = sorted_max_val_inds[-2:]
    top_max_locs = max_locs[top_max_val_inds]
    max_flux_pixels = pixels[top_max_locs]
    return np.sort(max_flux_pixels)


def get_highestflux_waves(complinelistdict):
    fms, wms = [], []
    for cwm, cfm in complinelistdict.values():
        fms.extend(cfm)
        wms.extend(cwm)
    fms, wms = np.asarray(fms), np.asarray(wms)
    flux_sorter = np.argsort(fms)
    top_inds = flux_sorter[int(0.75 * len(fms)):]
    fsorted_top_flux = fms[top_inds]
    fsorted_top_wave = wms[top_inds]
    wave_sorter = np.argsort(fsorted_top_wave)

    wsorted_top_flux = fsorted_top_flux[wave_sorter]
    wsorted_top_wave = fsorted_top_wave[wave_sorter]

    return wsorted_top_wave,wsorted_top_flux


def update_default_dict(default_dict,fiber_identifier,histories, \
                        pixels, comp_spec,matched_peak_waves,\
                        do_history=False,first_iteration=True):
    ## Update historical default
    if do_history:
        default_dict['from history'] = histories[fiber_identifier]

    ## Change offset of the basic default
    adef, bdef, cdef = default_dict['default']
    default_dict['default'] = (adef + aperature_number_pixoffset(fiber_identifier), bdef, cdef)

    ## Guess next position from the previous one and predictive offset function
    apred, bpred, cpred = default_dict['predicted from prev spec']
    expected_difference = aperature_pixoffset_between2(fiber_identifier)
    apred += expected_difference
    default_dict['predicted from prev spec'] = (apred, bpred, cpred)

    ## Use largest peaks to guess the constant and linear terms
    if not first_iteration:
        top_pixel_peaks = top_peak_pixels(pixels, comp_spec)
        ## naive linear fit without quadratic terms
        # dpix = top_pixel_peaks[1:]-top_pixel_peaks[:-1]
        # dlam = matched_peak_waves[1:]-matched_peak_waves[:-1]
        # bcor = np.median(dlam/dpix)
        # acor = np.median(matched_peak_waves-(bcor*top_pixel_peaks))
        ## using least squares curve_fit
        # linear = quad_to_linear(cpred)
        # (acor,bcor), cov = curve_fit(linear,top_pixel_peaks,matched_peak_waves,p0=(apred,bpred))
        ## Only update if the fit was reasonable  (numbers are arbitrary but reasonable vals)

        ## Fit to line but including the predicted quadratic term
        dpix = top_pixel_peaks[1] - top_pixel_peaks[0]
        dlam = matched_peak_waves[1] - matched_peak_waves[0]
        mean_pix = np.mean(top_pixel_peaks)
        bcor = (dlam / dpix) - 2 * cpred * mean_pix
        mean_wave = np.mean(matched_peak_waves)
        acor = mean_wave - (bcor * mean_pix) - (cpred * mean_pix * mean_pix)
        prev_acor, prev_bcor, prev_ccor = default_dict['cross correlation']
        if np.abs(prev_acor - acor) < 50 and np.abs(prev_bcor - bcor) < 0.2:
            default_dict['cross correlation'] = (acor, bcor, cpred)
        else:
            default_dict['cross correlation'] = (apred, bpred, cpred)
    return default_dict


import pickle as pkl
def interactive_wavelength_fitting(first_comp,complinelistdict,default = None,histories=None,\
                                   steps = None, default_key = None, trust_initial = False):
    if default is None:
        zeroeth_offset = 4523.4
        default = (zeroeth_offset,1.0007,-1.6e-6)

    default_dict = {    'default': default,
                        'predicted from prev spec': default,
                        'cross correlation': default           }

    do_history = False
    if histories is not None:
        default_dict['from history'] = default
        do_history = True

    if steps is None:
        steps = (1, 0.01, 0.00001)

    if default_key is None:
        default_key = 'cross correlation'

    ## Find the highest flux wavelengths in the calibrations
    wsorted_top_wave, wsorted_top_flux = get_highestflux_waves(complinelistdict)
    ## Make sure the information is in astropy table format
    first_comp = Table(first_comp)
    ## Define loop params
    counter = 0
    first_iteration = True

    ## Initiate arrays/dicts for later appending inside loop (for keeping in scope)
    matched_peak_waves, matched_peak_flux = [], []
    matched_peak_index = []
    all_coefs = {}
    all_flags = {}

    ## Loop over fiber names (strings e.g. 'r101')
    for fiber_identifier in first_comp.colnames:
        counter += 1
        print(fiber_identifier)

        ## Get the spectra (column with fiber name as column name)
        comp_spec = np.asarray(first_comp[fiber_identifier])

        ## create pixel array for mapping to wavelength
        pixels = np.arange(len(comp_spec))

        ## Update the defaults using history or cross correlation if available,
        ## and also update with a fitted function for the offsets
        default_dict = update_default_dict(default_dict,fiber_identifier,histories, \
                                           pixels, comp_spec,matched_peak_waves,\
                                           do_history,first_iteration)

        ## Do an interactive second order fit to the spectra
        if trust_initial and counter != 1:
            good_spec = True
            out_coef = {}
            out_coef['a'],out_coef['b'],out_coef['c'] = default_dict[default_key]
            print("\t\tYou trusted {} which gave: a={} b={} c={}".format(default_key,*default_dict[default_key]))
        else:
            good_spec,out_coef = interactive_plot(pixels=pixels, spectra=comp_spec,\
                             linelistdict=complinelistdict, gal_identifier=fiber_identifier,\
                             default_dict=default_dict,steps=steps,default_key=default_key)

        ## If it's the first iteration, use the results to compute the largest
        ## flux lines and their true wavelength values
        ## these are used in all future iterations of this loop in the cross cor
        if first_iteration and good_spec:
            top_peak_waves = top_peak_wavelengths(pixels, comp_spec, out_coef)

            for peak in top_peak_waves:
                index = np.argmin(np.abs(wsorted_top_wave-peak))
                matched_peak_waves.append(wsorted_top_wave[index])
                matched_peak_flux.append(wsorted_top_flux[index])
                matched_peak_index.append(index)

            matched_peak_waves = np.asarray(matched_peak_waves)
            matched_peak_flux = np.asarray(matched_peak_flux)
            matched_peak_index = np.asarray(matched_peak_index)
            print("Returned waves: {}\nMatched_waves:{}\n".format(top_peak_waves,matched_peak_waves))

        ## Save the flag
        all_flags[fiber_identifier] = good_spec

        ## Save the coefficients if it's good
        if good_spec:
            default_dict['predicted from prev spec'] = (out_coef['a'],out_coef['b'],out_coef['c'])
            all_coefs[fiber_identifier] = [out_coef['a'],out_coef['b'],out_coef['c'],0.,0.,0.]
            first_iteration = False
        else:
            all_coefs[fiber_identifier] = [0.,0.,0.,0.,0.,0.]

        if counter == 999:
            counter = 0
            with open('_temp_wavecalib.pkl','wb') as temp_pkl:
                pkl.dump([all_coefs,all_flags],temp_pkl)
            print("Saving an incremental backup to _temp_wavecalib.pkl")
            cont = str(input("\n\n\tDo you want to continue? (y or n)\t\t"))
            if cont.lower() == 'n':
                break

    return Table(all_coefs)



def wavelength_fitting(comp, linelistdict, coef_table, select_lines = False, bounds=None):

    if select_lines:
        app_specific_linelists = {}
        wm, fm = [], []
        for key,(keys_wm,keys_fm) in linelistdict.items():
            if key in['ThAr','Th']:
                # wm_thar,fm_thar = np.asarray(keys_wm), np.asarray(keys_fm)
                # sorted = np.argsort(fm_thar)
                # wm_thar_fsort,fm_thar_fsort = wm_thar[sorted], fm_thar[sorted]
                # cutoff = len(wm_thar_fsort)//2
                # wm_thar_fsortcut = wm_thar_fsort[cutoff:]
                # fm_thar_fsortcut = fm_thar_fsort[cutoff:]
                # wm.extend(wm_thar_fsortcut.tolist())
                # fm.extend(fm_thar_fsortcut.tolist())
                wm.extend(keys_wm)
                fm.extend(keys_fm)
            else:
                wm.extend(keys_wm)
                fm.extend(keys_fm)

        wm,fm = np.asarray(wm),np.asarray(fm)
        ordered = np.argsort(wm)
        wm = wm[ordered]
        fm = fm[ordered]

        all_wm,all_fm = wm.copy(),fm.copy()
        app_specific_linelists['all'] = (all_wm,all_fm)
    else:
        all_wm, all_fm = linelistdict['all']

    comp = Table(comp)
    counter = 0
    all_coefs = {}
    all_covs = {}

    for fiber in comp.colnames:
        counter += 1
        f_x = comp[fiber].data
        coefs = coef_table[fiber]
        iteration_wm,iteration_fm = [],[]
        if select_lines:
            iteration_wm,iteration_fm = wm.copy(),fm.copy()
        else:
            iteration_wm,iteration_fm = linelistdict[fiber]

        browser = LineBrowser(iteration_wm,iteration_fm, f_x, coefs,all_wm, bounds=bounds)
        browser.plot()
        params,covs = browser.fit()

        print(fiber,*params)
        all_coefs[fiber] = params
        all_covs[fiber] = covs

        savename = '{}'.format(fiber)
        browser.create_saveplot(params,covs, savename)

        if select_lines:
            app_specific_linelists[fiber] = (browser.wm, browser.fm)
            init_deleted_wm = np.asarray(browser.last['wm'])
            init_deleted_fm = np.asarray(browser.last['fm'])
            wm_sorter = np.argsort(init_deleted_wm)
            deleted_wm_srt, deleted_fm_srt = init_deleted_wm[wm_sorter], init_deleted_fm[wm_sorter]
            del init_deleted_fm, init_deleted_wm, wm_sorter
            mask_wm_nearedge = ((deleted_wm_srt>(browser.xspectra[0]+4)) & (deleted_wm_srt<(browser.xspectra[-1]-4)))
            deleted_wm = deleted_wm_srt[mask_wm_nearedge]
            deleted_fm = deleted_fm_srt[mask_wm_nearedge]
            del deleted_fm_srt, deleted_wm_srt, mask_wm_nearedge
            bool_mask = np.ones(shape=len(wm),dtype=bool)
            for w,f in zip(deleted_wm,deleted_fm):
                loc = wm.searchsorted(w)
                if fm[loc] == f:
                    bool_mask[loc] = False
            wm,fm = wm[bool_mask],fm[bool_mask]

        #wave, Flux, fifth, fourth, cube, quad, stretch, shift = wavecalibrate(p_x, f_x, 1679.1503, 0.7122818, 2778.431)
        plt.close()
        del browser
        if counter == 66:
            counter = 0
            if select_lines:
                with open('_temp_fine_wavecalib.pkl','wb') as temp_pkl:
                    pkl.dump([all_coefs,all_covs,app_specific_linelists],temp_pkl)
            else:
                with open('_temp_fine_wavecalib.pkl', 'wb') as temp_pkl:
                    pkl.dump([all_coefs, all_covs], temp_pkl)
            print("Saving an incremental backup to _temp_fine_wavecalib.pkl")
            cont = str(input("\n\n\tDo you want to continue? (y or n)\t\t"))
            if cont.lower() == 'n':
                break

    if select_lines:
        return Table(all_coefs), all_covs, app_specific_linelists
    else:
        return Table(all_coefs), all_covs

if __name__ == '__main__':
    cal_lamp = ['Hg', 'Argon', 'Neon','Xenon']
    linelistdict, cal_states = load_calibration_lines_salt_dict(cal_lamp)