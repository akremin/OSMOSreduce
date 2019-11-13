

# mock_spec_w, mock_spec_f = create_simple_line_spectra('ThAr', {'ThAr': (example_wm, example_fm)}, wave_low, wave_high,
#                                                       clab_step=clab_step)

from calibration_helper_funcs import get_fiber_number, pix_to_wave, get_psf,\
create_simple_line_spectra, update_coeficients_deviations
from scipy.ndimage import gaussian_filter
from linebrowser import LineBrowser
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from collections import OrderedDict


def auto_wavelength_fitting_by_lines_wrapper(input_dict):
    return auto_wavelength_fitting_by_lines(**input_dict)



def auto_wavelength_fitting_by_lines(comp, coarse_coefs, out_coefs, fulllinelist, linelistdict, mock_spec_w=None,mock_spec_f=None,\
                                     bounds=None, filenum='', save_plots = True,  savetemplate_funcs='{}{}{}{}{}'.format):
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
    hand_fit_subset = np.array(list(coarse_coefs.keys()))

    cam = comp.colnames[0][0]

    numeric_hand_fit_names = get_fiber_number(hand_fit_subset,cam=cam)
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

    upper_limit_resid = 0.4
    for itter in range(20):
        badfits,maxdevs = [],[]
        upper_limit_resid += 0.01
        for fiber in all_fibers:
            if fiber in hand_fit_subset or fiber not in coarse_coefs.colnames:
                continue
            if app_specific and fiber in linelistdict.keys():
                wm,fm = linelistdict[fiber]

            coefs = np.asarray(coarse_coef_fits[fiber])
            f_x = comp[fiber].data

            fibern = get_fiber_number(fibername=fiber,cam=cam)

            adjusted_coefs_guess = coefs
            if len(hand_fit_subset) > 0 and len(out_coefs)>0:
                adjusted_coefs_guess += update_coeficients_deviations(fiber,coarse_coefs,out_coefs)
            # if len(hand_fit_subset)==0:
            #     adjusted_coefs_guess = coefs
            # elif len(hand_fit_subset)==1:
            #     nearest_fib = hand_fit_subset[0]
            #     diffs_fib1 = np.asarray(coarse_coef_fits[nearest_fib]) - np.asarray(coarse_coef_fits[nearest_fib])
            #     diffs_mean = diffs_fib1
            #     adjusted_coefs_guess = coefs + diffs_mean
            # else:
            #     nearest_fibs = hand_fit_subset[np.argsort(np.abs(fibern - numeric_hand_fit_names))]
            #     diffs_fib1 = np.asarray(coarse_coef_fits[nearest_fibs[0]]) - np.asarray(coarse_coef_fits[nearest_fibs[0]])
            #     diffs_mean = diffs_fib1
            #     adjusted_coefs_guess = coefs + diffs_mean

            browser = LineBrowser(wm,fm,fulllinelist,mock_spec_w,mock_spec_f, f_x, adjusted_coefs_guess, bounds=None, edge_line_distance=(-20.0),initiate=False)

            params,covs, resid = browser.fit()

            print('\n\n',fiber,'{:.2f} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}'.format(*params))
            fitlamb = pix_to_wave(np.asarray(browser.line_matches['peaks_p']), params)
            dlamb = fitlamb - browser.line_matches['lines']
            print("  ----> mean={}, median={}, std={}, sqrt(resid)={}".format(np.mean(dlamb),np.median(dlamb),np.std(dlamb),np.sqrt(resid)))

            if np.sqrt(resid) < upper_limit_resid:
                if save_plots:
                    template = savetemplate_funcs(cam=str(filenum) + '_', ap=fiber, imtype='calib', step='finalfit',
                                                  comment='auto')
                    browser.initiate_browser()
                    browser.create_saveplot(params, covs, template)

                out_coefs[fiber] = params
                variances[fiber] = covs.diagonal()
                outlinelist[fiber] = (wm, fm)
                app_fit_pix[fiber] = browser.line_matches['peaks_p']
                app_fit_lambs[fiber] = browser.line_matches['lines']
                # if np.sqrt(resid) < upper_limit_resid:
                numeric_hand_fit_names = np.append(numeric_hand_fit_names,fibern)
                hand_fit_subset = np.append(hand_fit_subset,fiber)
            else:
                badfits.append(fiber)
                if np.sqrt(resid) < 3.0:
                    guessed_waves = pix_to_wave(browser.line_matches['peaks_p'], adjusted_coefs_guess)
                    lines = browser.line_matches['lines']
                    dlines = np.array(lines) - np.array(guessed_waves)

                    sorted_args = np.argsort(np.abs(dlines))
                    sorted_dlamb = np.abs(dlines)[sorted_args]
                    if (sorted_dlamb[-1] > 10.0) and sorted_dlamb[-3] < 4.0:
                        maxdev_line = lines[sorted_args[-1]]

                        if app_specific and len(wm) > 11:
                            print(
                                "\n\n\nDetermined that line={} is causing bad fit for fiber={}. Dev:{}  vs 3rd:{}\n\n\n".format( \
                                    maxdev_line, fiber, sorted_dlamb[-1], sorted_dlamb[-3]))
                            wm_loc = np.where(wm == maxdev_line)[0][0]
                            wmlist, fmlist = wm.tolist(), fm.tolist()
                            wmlist.pop(wm_loc)
                            fmlist.pop(wm_loc)
                            linelistdict[fiber] = (np.asarray(wmlist), np.asarray(fmlist))
                        else:
                            maxdevs.append(maxdev_line)

            plt.close()
            del browser

        if (not app_specific) and (len(maxdevs) > (len(all_fibers)//4)) and (len(wm) > 11):
            count = Counter(maxdevs)
            line, num = count.most_common(1)[0]
            if (num >= (len(maxdevs)//3)):
                print(
                    "\n\n\nDetermined that line={} is causing bad fits, {} of {} had problems with it out of {} fibers\n\n\n".format( \
                        line, num, len(maxdevs), len(all_fibers)))

                wm_loc = np.where(wm == line)[0][0]
                wmlist,fmlist = wm.tolist(),fm.tolist()
                wmlist.pop(wm_loc)
                fmlist.pop(wm_loc)
                wm,fm = np.asarray(wmlist),np.asarray(fmlist)

        all_fibers = np.array(badfits)[::-1]

    out_dict = {'calib coefs':out_coefs, 'fit variances':variances, 'wavelengths':app_fit_lambs, 'pixels':app_fit_pix, 'linelist':outlinelist}
    return out_dict, all_fibers


def wavelength_fitting_by_line_selection(comp, coarse_coef_fits, fulllinelist, selectedlistdict, mock_spec_w=None,mock_spec_f=None,  \
                                         bounds=None, filenum='', savetemplate_funcs='', save_plots=False, select_lines = False, \
                                         subset=[],completed_coefs = {}):
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

    iteration_wm, iteration_fm = [],[]
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

            if select_lines:
                iteration_wm, iteration_fm = wm.copy(), fm.copy()
            else:
                iteration_wm, iteration_fm = selectedlistdict[fiber]

            updated_coefs = coefs
            if len(completed_coefs) > 0:
                updated_coefs += update_coeficients_deviations(fiber, coarse_coef_fits, completed_coefs)

            browser = LineBrowser(iteration_wm,iteration_fm, f_x, fulllinelist, mock_spec_w, mock_spec_f, updated_coefs, bounds=bounds, \
                                  edge_line_distance=10.0,fibname=fiber)
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
                    break
            else:
                break
        else:
            break

    if not select_lines:
        app_specific_linelists = selectedlistdict
        wm,fm = iteration_wm, iteration_fm

    out_dict = {'calib coefs':out_coefs, 'fit variances':variances, 'wavelengths':app_fit_lambs, 'pixels':app_fit_pix, 'linelist':app_specific_linelists}
    return out_dict, wm, fm