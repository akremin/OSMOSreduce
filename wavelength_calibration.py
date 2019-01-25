
import numpy as np
from astropy.table import Table
import pickle as pkl
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

deltat = np.datetime64('now' ,'m').astype(int ) -np.datetime64('2018-06-01T00:00' ,'m').astype(int)
print(deltat)



# if do_step['wavecalib']:
#     load_fromfile_if_possible = False
#     timestamp = np.datetime64('now', 'm').astype(int) - np.datetime64('2018-06-01T00:00', 'm').astype(int)
#
#     from wavelength_calibration_funcs import calibrate_pixels2wavelengths
#     from calibrations import wavelength_fitting, interactive_wavelength_fitting
#
#     from calibrations import save_calib_dict ,locate_calib_dict
#
#     # bounds = None
#     bounds = ([-1e5 ,0.96 ,-1e-4 ,-1e-6 ,-1e-8 ,-1e-8] ,[1e5 ,1.2 ,1e-4 ,1e-6 ,1e-8 ,1e-8])
#     complinelistdict = load_calibration(cal_lamp, wavemincut=4500, wavemaxcut=6600)
#     tharlinelistdict = load_calibration(thar_lamp, wavemincut=4500, wavemaxcut=6600)
#
#     calib_coefs = {}
#     calib_coefs['comp'] = {key :{} for key in dict_of_hdus['comp'][setup_info['cameras'][0]].keys()}
#     calib_coefs['thar'] = {key :{} for key in dict_of_hdus['thar'][setup_info['cameras'][0]].keys()}
#     calib_coefs['interactive'] = {key :{} for key in setup_info['cameras']}
#
#     for camera in setup_info['cameras']:
#         comp_filenums = list(dict_of_hdus['comp'][camera].keys())
#         thar_filenums = list(dict_of_hdus['thar'][camera].keys())
#
#         ## Interactive
#         fil = comp_filenums[0]
#         if load_fromfile_if_possible:
#             coef_table = locate_calib_dict('./', 'interactive' ,camera ,config ,fil)
#             if coef_table is None:
#                 load_fromfile_if_possible = False
#             else:
#                 calib_coef_table = coef_table
#
#         coarse_comp = (dict_of_hdus['comp'][camera][fil]).data
#
#         if not load_fromfile_if_possible:
#             calib_coef_table = interactive_wavelength_fitting(coarse_comp ,complinelistdict, \
#                                                               default = (4522.6 ,1.0007 ,-1.6e-6), \
#                                                               trust_initial = True)
#             calib_coefs['interactive'][camera] = calib_coef_table
#             save_calib_dict(calib_coef_table ,'interactive' ,camera ,config ,fil ,timestamp)
#
#         calib_coefs['interactive'][camera] = calib_coef_table
#
#         ## First pointed fit
#         if load_fromfile_if_possible:
#             coef_table = locate_calib_dict('./', 'compfit' ,camera ,config ,comp_filenums[0])
#             if coef_table is None:
#                 load_fromfile_if_possible = False
#             else:
#                 calib_coef_table = coef_table
#
#         if not load_fromfile_if_possible:
#             calib_coef_table, covs, selected_complinelists = wavelength_fitting(coarse_comp, complinelistdict, \
#                                                                                 calib_coef_table ,select_lines = True, bounds=bounds)
#             save_calib_dict(calib_coef_table, 'compfit', camera, config, comp_filenums[0], timestamp)
#
#         calib_coefs['comp'][comp_filenums[0]][camera] = calib_coef_table
#
#         ## Loop through pointed fits
#         # for filenum in comp_filenums[1:]:
#         #     if load_fromfile_if_possible:
#         #         coef_table = locate_calib_dict('./', 'compfit', camera, config, filenum)
#         #         if coef_table is None:
#         #             load_fromfile_if_possible = False
#         #         else:
#         #             calib_coef_table = coef_table
#         #
#         #     if not load_fromfile_if_possible:
#         #         comp = dict_of_hdus['comp'][camera][filenum].data
#         #         calib_coef_table, covs = wavelength_fitting(comp, selected_complinelists, calib_coef_table)
#         #         save_calib_dict(calib_coef_table, 'compfit', camera, config, filenum, timestamp)
#         #
#         #     calib_coefs['comp'][filenum][camera] = calib_coef_table
#
#         if load_fromfile_if_possible:
#             coef_table = locate_calib_dict('./', 'tharfit', camera, config, thar_filenums[0])
#             if coef_table is None:
#                 load_fromfile_if_possible = False
#             else:
#                 calib_coef_table = coef_table
#
#         if not load_fromfile_if_possible:
#             first_thar = (dict_of_hdus['thar'][camera][thar_filenums[0]]).data
#             calib_coef_table, covs, selected_tharlinelists = wavelength_fitting(first_thar, tharlinelistdict, \
#                                                                                 calib_coef_table ,select_lines = True)
#             save_calib_dict(calib_coef_table, 'tharfit', thar_filenums[0], camera, config, timestamp)
#
#         calib_coefs['thar'][thar_filenums[0]][camera] = calib_coef_table
#
#         for filenum in thar_filenums[1:]:
#             if load_fromfile_if_possible:
#                 coef_table = locate_calib_dict('./', 'tharfit', camera, config, filenum)
#                 if coef_table is None:
#                     load_fromfile_if_possible = False
#                 else:
#                     calib_coef_table = coef_table
#
#             if not load_fromfile_if_possible:
#                 thar = dict_of_hdus['thar'][camera][filenum].data
#                 calib_coef_table, covs = wavelength_fitting(thar ,selected_tharlinelists, calib_coef_table)
#                 save_calib_dict(calib_coef_table, 'tharfit', camera, config, filenum, timestamp)
#
#             calib_coefs['thar'][filenum][camera] = calib_coef_table
#
#     with open('calib_coefs.pkl' ,'wb') as pklout:
#         pkl.dump(calib_coefs ,pklout)



def aperature_number_pixoffset(fibnum,camera='r'):
    if type(fibnum) is str:
        strpd_fibnum = fibnum.strip('rb')
        if strpd_fibnum.isnumeric():
            tet = np.int8(strpd_fibnum[0]) - 1.
            fib = np.int8(strpd_fibnum[1:]) - 1.
        else:
            return 0.
    elif np.isscalar(fibnum):
        tet = fibnum // 16
        fib = fibnum % 16

    if camera.lower() != 'r':
        orientation = 1.
    else:
        orientation = -1.
    c1, c2, c3, c4, c5 = 1.023, 54.058, -6.962, 1.985, -0.5560
    outval_mag = (c1) + (c2 * tet) + (c3 * tet * tet) + (c4 * fib) + (c5 * tet * fib)
    return orientation * outval_mag

def aperature_pixoffset_between2(fibnum,camera='r'):
    if type(fibnum) is str:
        strpd_fibnum = fibnum.strip('rb')
        if strpd_fibnum.isnumeric():
            tet = np.int8(strpd_fibnum[0]) - 1.
            fib = np.int8(strpd_fibnum[1:]) - 1.
        else:
            return 0.
        fibnum = 16*tet+fib
    elif not np.isscalar(fibnum):
        return 0.

    if fibnum == 0:
        return 0.
    else:
        return aperature_number_pixoffset(fibnum,camera)-aperature_number_pixoffset(fibnum-1,camera)


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
    for (cwm, cfm) in complinelistdict.values():
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


def update_default_dict(default_dict,fiber_identifier,default_vals, history_vals, \
                        pixels, comp_spec,matched_peak_waves,\
                        do_history=False,first_iteration=True):
    ## Change offset of the basic default
    adef,bdef,cdef,ddef,edef,fdef = default_vals[fiber_identifier]
    default_dict['default'] = (adef,bdef,cdef)

    ## Update historical default
    if do_history:
        if fiber_identifier in history_vals.colnames:
            default_dict['from history'] = history_vals[fiber_identifier]
        else:
            default_dict['from history'] = default_dict['default']

    ## Guess next position from the previous one and predictive offset function
    apred, bpred, cpred = default_dict['predicted from prev spec']
    #expected_difference = aperature_pixoffset_between2(fiber_identifier)
    #default_dict['predicted from prev spec'] = (apred+expected_difference, bpred, cpred)

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
            default_dict['predicted from prev spec'] = (apred + (acor-prev_acor), bpred, cpred)
        else:
            default_dict['cross correlation'] = (adef, bpred, cpred)
            default_dict['predicted from prev spec'] = (adef, bpred, cpred)
    return default_dict


def run_interactive_slider_calibration(self,coarse_comp, complinelistdict, default_vals=None,history_vals=None,\
                                   steps = None, default_key = None, trust_initial = False):

    init_default = (4523.4,1.0007,-1.6e-6)

    default_dict = {    'default': init_default,
                        'predicted from prev spec': init_default,
                        'cross correlation': init_default           }

    do_history = False
    if history_vals is not None:
        default_dict['from history'] = init_default
        do_history = True

    if steps is None:
        steps = (1, 0.01, 0.00001)

    if default_key is None:
        default_key = 'cross correlation'

    ## Find the highest flux wavelengths in the calibrations
    wsorted_top_wave, wsorted_top_flux = get_highestflux_waves(complinelistdict)
    ## Make sure the information is in astropy table format
    coarse_comp = Table(coarse_comp)
    ## Define loop params
    counter = 0
    first_iteration = True

    ## Initiate arrays/dicts for later appending inside loop (for keeping in scope)
    matched_peak_waves, matched_peak_flux = [], []
    matched_peak_index = []
    all_coefs = {}
    all_flags = {}

    ## Loop over fiber names (strings e.g. 'r101')
    ##hack!
    for fiber_identifier in coarse_comp.colnames: #['r101','r401','r801']:
        counter += 1
        print(fiber_identifier)

        ## Get the spectra (column with fiber name as column name)
        comp_spec = np.asarray(coarse_comp[fiber_identifier])

        ## create pixel array for mapping to wavelength
        pixels = np.arange(len(comp_spec))

        ## Update the defaults using history or cross correlation if available,
        ## and also update with a fitted function for the offsets
        default_dict = update_default_dict(default_dict,fiber_identifier,default_vals, history_vals, \
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



def wavelength_fitting_by_line_selection(self,comp, selectedlistdict, fulllinelist, coef_table, select_lines = False, bounds=None):
    if select_lines:
        wm, fm = [], []
        for key,(keys_wm,keys_fm) in selectedlistdict.items():
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

    comp = Table(comp)
    counter = 0
    app_specific_linelists = {}

    all_coefs = {}
    variances = {}
    app_fit_pix = {}
    app_fit_lambs = {}
    for fiber in comp.colnames: #['r101','r401','r801']:
        counter += 1
        f_x = comp[fiber].data
        coefs = coef_table[fiber]
        iteration_wm,iteration_fm = [],[]
        if select_lines:
            iteration_wm,iteration_fm = wm.copy(),fm.copy()
        else:
            iteration_wm,iteration_fm = selectedlistdict[fiber]

        browser = LineBrowser(iteration_wm,iteration_fm, f_x, coefs, fulllinelist, bounds=bounds)
        if np.any((np.asarray(browser.line_matches['lines'])-np.asarray(browser.line_matches['peaks_w']))>0.5):
            browser.plot()
        params,covs = browser.fit()

        print(fiber,*params)
        all_coefs[fiber] = params
        variances[fiber] = covs.diagonal()
        print(np.dot(variances[fiber],variances[fiber]))

        #savename = '{}'.format(fiber)
        #browser.create_saveplot(params,covs, savename)

        app_fit_pix[fiber] = browser.line_matches['peaks_p']
        app_fit_lambs[fiber] = browser.line_matches['lines']
        if select_lines:
            app_specific_linelists[fiber] = (browser.wm, browser.fm)
            init_deleted_wm = np.asarray(browser.last['wm'])
            init_deleted_fm = np.asarray(browser.last['fm'])
            wm_sorter = np.argsort(init_deleted_wm)
            deleted_wm_srt, deleted_fm_srt = init_deleted_wm[wm_sorter], init_deleted_fm[wm_sorter]
            del init_deleted_fm, init_deleted_wm, wm_sorter
            mask_wm_nearedge = ((deleted_wm_srt>(browser.xspectra[0]+10)) & (deleted_wm_srt<(browser.xspectra[-1]-10)))
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
                    pkl.dump([all_coefs,variances,app_specific_linelists],temp_pkl)
            else:
                with open('_temp_fine_wavecalib.pkl', 'wb') as temp_pkl:
                    pkl.dump([all_coefs, variances], temp_pkl)
            print("Saving an incremental backup to _temp_fine_wavecalib.pkl")
            cont = str(input("\n\n\tDo you want to continue? (y or n)\t\t"))
            if cont.lower() == 'n':
                break

    cont = input("\n\n\tDo you need to repeat any? (y or n)")
    if cont.lower() == 'y':
        fiber = input("\n\tName the fiber")
        print(fiber)
        cam = comp.colnames[0][0]
        while fiber != '':
            if cam not in fiber:
                fiber = cam + fiber
            f_x = comp[fiber].data
            coefs = coef_table[fiber]
            iteration_wm, iteration_fm = [], []
            if select_lines:
                iteration_wm, iteration_fm = wm.copy(), fm.copy()
            else:
                iteration_wm, iteration_fm = selectedlistdict[fiber]

            browser = LineBrowser(iteration_wm, iteration_fm, f_x, coefs, fulllinelist, bounds=bounds)
            browser.plot()
            params, covs = browser.fit()

            print(fiber, *params)
            all_coefs[fiber] = params
            variances[fiber] = covs.diagonal()
            print(np.dot(variances[fiber], variances[fiber]))

            if select_lines:
                app_specific_linelists[fiber] = (browser.wm, browser.fm)

            # wave, Flux, fifth, fourth, cube, quad, stretch, shift = wavecalibrate(p_x, f_x, 1679.1503, 0.7122818, 2778.431)
            plt.close()
            del browser
            fiber = input("\n\tName the fiber")

    if not select_lines:
        app_specific_linelists = None
    return Table(all_coefs), app_specific_linelists, app_fit_lambs, app_fit_pix, variances