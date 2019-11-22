

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack


from calibration_helper_funcs import pix_to_wave,\
    get_fiber_number, pix_to_wave_explicit_coefs2, get_meantime_and_date,\
    update_coeficients_deviations,vacuum_to_air

from linebrowser import LineBrowser
from collections import Counter


def auto_wavelength_fitting_by_lines_wrapper(input_dict):
    return auto_wavelength_fitting_by_lines(**input_dict)



def auto_wavelength_fitting_by_lines(comp, coarse_coefs, out_coefs, fulllinelist, linelistdict, mock_spec_w=None,mock_spec_f=None,\
                                     bounds=None, filenum='', save_plots = True,  savetemplate_funcs='{}{}{}{}{}'.format):
    print('in autocal')
    if 'ThAr' in linelistdict.keys():
        wm, fm = linelistdict['ThAr']
        app_specific = False
    else:
        app_specific = True

    comp = Table(comp)

    # out_coefs = {}
    variances = {}
    app_fit_pix = {}
    app_fit_lambs = {}
    outlinelist = {}
    hand_fit_subset = np.array(list(out_coefs.keys()))

    cam = comp.colnames[0][0]

    # numeric_hand_fit_names = get_fiber_number(hand_fit_subset,cam=cam)
    all_numerics = get_fiber_number(comp.colnames,cam=cam)

    coarse_coef_fits = Table(coarse_coefs)

    sorted = np.argsort(all_numerics)
    all_fibers = np.array(comp.colnames)[sorted]
    del sorted,all_numerics

    # ## go from outside in
    # if cam == 'r' and int(all_fibers[0][1]) > 3:
    #     all_fibers = all_fibers[::-1]
    # elif cam =='b' and int(all_fibers[0][1]) < 6:
    #     all_fibers = all_fibers[::-1]

    ## go from inside out
    if len(all_fibers) < 70:
        if cam == 'r' and int(all_fibers[0][1]) < 4:
            all_fibers = all_fibers[::-1]
        elif cam =='b' and int(all_fibers[0][1]) > 5:
            all_fibers = all_fibers[::-1]

    upper_limit_resid,upper_limit_resid_step = 0.36,0.04
    max_upper_limit = 0.74
    len_last_loop = 999
    bad_waves = []
    completed_fibers = []
    for itter in range(200):
        badfits,maxdevs = [],[]
        if len(all_fibers) == len_last_loop:
            if upper_limit_resid + upper_limit_resid_step > max_upper_limit:
                break

            print("\n\n\nFiber iteration didn't yield improvement, increasing limit from {:.02f} to {:.02f}\n\n\n".format(\
                upper_limit_resid,upper_limit_resid+upper_limit_resid_step))
            print("{} Fibers remain out of {}".format(len(all_fibers),len(comp.colnames)))
            upper_limit_resid += upper_limit_resid_step

        for fiber in all_fibers:
            if fiber in completed_fibers or fiber not in coarse_coef_fits.colnames:
                continue
            if fiber in outlinelist.keys():
                itterwm, itterfm = outlinelist[fiber]
            elif app_specific and fiber in linelistdict.keys():
                itterwm,itterfm = linelistdict[fiber]
            else:
                itterwm,itterfm = wm.copy(),fm.copy()
            coefs = np.asarray(coarse_coef_fits[fiber]).copy()
            f_x = comp[fiber].data

            if len(out_coefs) > 0:
                if fiber in out_coefs.keys():
                    adjusted_coefs_guess = np.array(out_coefs[fiber],copy=True)
                else:
                    adjusted_coefs_guess = coefs.copy() + update_coeficients_deviations(fiber,coarse_coefs,out_coefs)
            else:
                adjusted_coefs_guess = coefs.copy()

            browser = LineBrowser(itterwm,itterfm,fulllinelist,mock_spec_w,mock_spec_f, f_x, adjusted_coefs_guess, \
                                  bounds=None, edge_line_distance=(-20.0),initiate=False)

            params,covs, resid = browser.fit()

            if np.sqrt(resid) < upper_limit_resid:
                fitlamb = pix_to_wave(np.asarray(browser.line_matches['peaks_p']), params)
                dlamb = fitlamb - browser.line_matches['lines']
                print('\n\n', fiber, '{:.2f} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}'.format(*params))
                print("  ----> mean={:.06f}, median={:.06f}, std={:.06f}, sqrt(resid)={:.06f}".format(np.mean(dlamb),\
                                                                                                      np.median(dlamb),\
                                                                                                      np.std(dlamb),\
                                                                                                      np.sqrt(resid)))

                if save_plots:
                    template = savetemplate_funcs(cam=str(filenum) + '_', ap=fiber, imtype='calib', step='finalfit',
                                                  comment='auto')
                    browser.initiate_browser()
                    browser.create_saveplot(params, covs, template)

                out_coefs[fiber] = params
                variances[fiber] = covs.diagonal()
                outlinelist[fiber] = (itterwm,itterfm)
                app_fit_pix[fiber] = browser.line_matches['peaks_p']
                app_fit_lambs[fiber] = browser.line_matches['lines']
                # if np.sqrt(resid) < upper_limit_resid:
                # numeric_hand_fit_names = np.append(numeric_hand_fit_names,fibern)
                completed_fibers.append(fiber)
            else:
                badfits.append(fiber)
                fitlamb = pix_to_wave(np.asarray(browser.line_matches['peaks_p']), adjusted_coefs_guess)
                dlamb = fitlamb - browser.line_matches['lines']
                abs_dlamb = np.abs(dlamb)
                if np.any(abs_dlamb>2.) and len(browser.line_matches['lines']) > 11:
                    ninetieth_quant = np.quantile(abs_dlamb,q=0.9)
                    if ninetieth_quant < 1. and np.max(abs_dlamb) > 2*ninetieth_quant:
                        maxdev_line = browser.line_matches['lines'][np.argmax(abs_dlamb)]

                        print("\n\n\nDetermined that line={:.03f} is causing bad fit for fiber={}.".format( \
                                maxdev_line, fiber))
                        print(" Dev: {:.03f} vs Median: {:.03f} (in {} total lines)\n\n\n".format(np.max(abs_dlamb),\
                                                                                                  np.median(abs_dlamb),\
                                                                                                  len(browser.line_matches['lines'])))
                        print("Vac Wavelength: {:.03f}A   Air Wavelength: {:.03f}A".format(maxdev_line,vacuum_to_air(maxdev_line)))
                        bad_waves.append(maxdev_line)
                        wm_loc = np.argmin(np.abs(itterwm-maxdev_line))
                        wmlist, fmlist = itterwm.tolist(), itterfm.tolist()
                        wmlist.pop(wm_loc)
                        fmlist.pop(wm_loc)
                        outlinelist[fiber] = (np.asarray(wmlist), np.asarray(fmlist))
            plt.close()
            del browser

        len_last_loop = len(all_fibers)
        all_fibers = np.array(badfits)[::-1]
        if len(all_fibers) == 0:
            break

    if len(all_fibers)> 0:
        print("\n\t{} fibers fit.".format(len(app_fit_pix)))
        print("\tThe remaining {} cannot converge in less than rms={:.02f} automatically. You'll have to fit by hand\n".format(len(all_fibers),upper_limit_resid))
        print(all_fibers)
        for fiber in all_fibers:
            if fiber in outlinelist.keys():
                outlinelist.pop(fiber)
    if len(bad_waves)> 0:
        print("The most problematic wavelengths were:")
        count = Counter(bad_waves)
        for line, number in count.most_common():
            print("{} fibers had problems with:\t{:.03f} (Vac A)\t\t{:.03f} (Air A)".format(number,line,vacuum_to_air(line)))

    out_dict = {'calib coefs':out_coefs, 'fit variances':variances, 'wavelengths':app_fit_lambs, 'pixels':app_fit_pix, 'linelist':outlinelist}
    return out_dict, all_fibers









def wavelength_fitting_by_line_selection(comp, coarse_coef_fits, fulllinelist, selectedlistdict, mock_spec_w=None,mock_spec_f=None,  \
                                         bounds=None, filenum='', savetemplate_funcs='', save_plots=False, select_lines = False, \
                                         subset=[],completed_coefs = {}):
    iteration_wm, iteration_fm = None, None
    wm, fm = None, None
    if select_lines:
        wm, fm = [], []
        for key,(keys_wm,keys_fm) in selectedlistdict.items():
            if key in ['ThAr','Th']:
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

    out_coefs = {}
    variances = {}
    app_fit_pix = {}
    app_fit_lambs = {}
    cam = comp.colnames[0][0]

    if cam == 'r':
        extremes = ['r101','r816','r416','r501']
    else:
        extremes = ['b116', 'b801', 'b516','b401']

    extrema_fiber = False
    hand_fit_subset = np.array(subset)
    fibers = hand_fit_subset.copy()
    for itter in range(1000):
        for fiber in fibers:
            if fiber in extremes:
                extrema_fiber = True
            else:
                extrema_fiber = False
            counter += 1
            f_x = comp[fiber].data

            coefs = coarse_coef_fits[fiber]

            if select_lines or fiber not in selectedlistdict.keys():
                iteration_wm, iteration_fm = wm.copy(), fm.copy()
            else:
                iteration_wm, iteration_fm = selectedlistdict[fiber]

            updated_coefs = coefs.copy()
            if len(completed_coefs) > 0:
                updated_coefs += update_coeficients_deviations(fiber, coarse_coef_fits, completed_coefs)

            browser = LineBrowser(iteration_wm,iteration_fm, fulllinelist, mock_spec_w, mock_spec_f, f_x, updated_coefs, bounds=bounds, \
                                  edge_line_distance=20.0,fibname=fiber)
            if np.any((np.asarray(browser.line_matches['lines'])-np.asarray(browser.line_matches['peaks_w']))>0.3):
                browser.plot()
            params,covs,resid = browser.fit()

            print(fiber,*params)
            print(np.sqrt(resid))
            out_coefs[fiber] = params
            # completed_coefs[fiber] = params
            variances[fiber] = covs.diagonal()
            app_fit_pix[fiber] = browser.line_matches['peaks_p']
            app_fit_lambs[fiber] = browser.line_matches['lines']

            template = savetemplate_funcs(cam=str(filenum)+'_',ap=fiber,imtype='calib',step='finalfit',comment='byhand')
            if save_plots:
                browser.create_saveplot(params,covs, template)

            if select_lines and itter == 0:
                app_specific_linelists[fiber] = (browser.wm, browser.fm)
                init_deleted_wm = np.asarray(browser.last['wm'])
                init_deleted_fm = np.asarray(browser.last['fm'])
                wm_sorter = np.argsort(init_deleted_wm)
                deleted_wm_srt, deleted_fm_srt = init_deleted_wm[wm_sorter], init_deleted_fm[wm_sorter]
                del init_deleted_fm, init_deleted_wm, wm_sorter
                if extrema_fiber and fiber[1] in ['1','8']:
                    ## If outer extremes, these define the highest possible wavelengths viewed, so they define the upper bound
                    ## but not the lower bound
                    mask_wm_nearedge = (deleted_wm_srt>(browser.xspectra[0]+100.0))
                    deleted_wm = deleted_wm_srt[mask_wm_nearedge]
                    deleted_fm = deleted_fm_srt[mask_wm_nearedge]
                elif extrema_fiber and fiber[1] in ['4','5']:
                    ## If inner extremes, these define the lowest possible wavelengths viewed, so they define the lower bound
                    ## but not the upper bound
                    mask_wm_nearedge = (deleted_wm_srt<(browser.xspectra[-1]-100.0))
                    deleted_wm = deleted_wm_srt[mask_wm_nearedge]
                    deleted_fm = deleted_fm_srt[mask_wm_nearedge]
                else:
                    mask_wm_nearedge = ((deleted_wm_srt>(browser.xspectra[0]+100.0)) & (deleted_wm_srt<(browser.xspectra[-1]-100.0)))
                    deleted_wm = deleted_wm_srt[mask_wm_nearedge]
                    deleted_fm = deleted_fm_srt[mask_wm_nearedge]
                bool_mask = np.ones(shape=len(wm),dtype=bool)
                from calibration_helper_funcs import vacuum_to_air
                print(vacuum_to_air(deleted_wm))
                for w,f in zip(deleted_wm,deleted_fm):
                    loc = wm.searchsorted(w)
                    if fm[loc] == f:
                        bool_mask[loc] = False
                wm,fm = wm[bool_mask],fm[bool_mask]

            #wave, Flux, fifth, fourth, cube, quad, stretch, shift = wavecalibrate(p_x, f_x, 1679.1503, 0.7122818, 2778.431)
            plt.close()
            del browser

        fibers = []
        if len(hand_fit_subset) > 0:
            cont = 'q'
            while cont.lower() not in ['y','n']:
                cont = str(input("\n\n\tDo you need to repeat any? (y or n)")).strip(' \t\r\n')
            if cont.lower() == 'y':
                fiber = str(input("\n\tName the fiber")).strip(' \t\r\n')
                print("Received: '{}'".format(fiber))

                if fiber == '' or fiber is None:
                    continue
                elif fiber in ['b','d','x','c','n']:
                    break
                else:
                    if cam not in fiber:
                        fiber = cam + fiber
                    fibers = [fiber]
                    continue
            else:
                break
        else:
            break

    if not select_lines:
        app_specific_linelists = selectedlistdict
        if iteration_wm is not None:
            wm,fm = iteration_wm, iteration_fm

    out_dict = {'calib coefs':out_coefs, 'fit variances':variances, 'wavelengths':app_fit_lambs, 'pixels':app_fit_pix, 'linelist':app_specific_linelists}
    return out_dict, wm, fm