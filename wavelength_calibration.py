
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




def run_automated_calibration(coarse_comp, complinelistdict, last_obs=None, print_itters = True):
    precision = 1e-3
    convergence_criteria = 1.0e-5 # change in correlation value from itteration to itteration
    waves, fluxes = generate_synthetic_spectra(complinelistdict, compnames=['HgAr', 'NeAr'],precision=precision,maxheight=10000.)

    ## Make sure the information is in astropy table format
    coarse_comp = Table(coarse_comp)
    ## Define loop params
    counter = 0

    ## Initiate arrays/dicts for later appending inside loop (for keeping in scope)
    all_coefs = {}
    all_flags = {}

    ## Loop over fiber names (strings e.g. 'r101')
    ##hack!
    fibernames = coarse_comp.colnames
    for fiber_identifier in fibernames:#['r101','r408','r409','r608','r816']:
        counter += 1
        #print("\n\n", fiber_identifier)

        ## Get the spectra (column with fiber name as column name)
        comp_spec = np.asarray(coarse_comp[fiber_identifier])

        ## create pixel array for mapping to wavelength
        pixels = np.arange(len(comp_spec))

        pix1 = pixels
        pix2 = pixels*pixels
        subset = np.arange(0, len(pixels), 2).astype(int)
        subset_comp = comp_spec[subset]
        subpix1 = pix1[subset]

        abest, bbest, cbest, corrbest = 0., 0., 0., 0.
        alow, ahigh = 3000, 8000

        if last_obs is None or fiber_identifier not in last_obs.keys():
            if counter == 1:
                avals = (alow, ahigh+1, 1)
                bvals = (0.96,1.04,0.01)
                cvals = (0., 1., 1.)
                if print_itters:
                    print("\nItter 1 results, (fixing c to 0.):")
                abest, bbest, cbest, corrbest = fit_using_crosscorr(pixels=subpix1, raw_spec=subset_comp,
                                                                      comp_highres_fluxes=fluxes, \
                                                                      avals=avals, bvals=bvals, cvals=cvals, \
                                                                      calib_wave_start=waves[0],
                                                                      flux_wave_precision=precision,\
                                                                      print_itters=print_itters)
            else:
                last_fiber = fibernames[counter-2]
                [trasha, bbest, cbest, trash1, trash2, trash3] = all_coefs[last_fiber]
                astep,bstep,cstep = 1,1,1
                avals = (alow,   ahigh+astep,  astep)
                bvals = (bbest , bbest+bstep , bstep)
                cvals = (cbest , cbest+cstep , cstep)
                if print_itters:
                    print("\nItter 1 results, (fixing b and c to past vals):")
                abest, trashb, trashc, corrbest = fit_using_crosscorr(pixels=subpix1, raw_spec=subset_comp,
                                                                    comp_highres_fluxes=fluxes, \
                                                                    avals=avals, bvals=bvals, cvals=cvals, \
                                                                    calib_wave_start=waves[0],
                                                                    flux_wave_precision=precision,\
                                                                      print_itters=print_itters)
        else:
            [abest, bbest, cbest, trash1, trash2, trash3] = last_obs[fiber_identifier]
            if print_itters:
                print("\nItter 1 results:")
                print("--> Using previous obs value of:   a={:.2f}, b={:.5f}, c={:.2e}".format(abest, bbest, cbest))

        if print_itters:
            print("\nItter 2 results:")
        astep,bstep,cstep = 1, 1.0e-3, 4.0e-7
        awidth, bwidth, cwidth = 20, 0.02, 4.0e-6
        avals = ( abest-awidth, abest+awidth+astep, astep )
        bvals = ( bbest-bwidth, bbest+bwidth+bstep, bstep )
        cvals = ( cbest-cwidth, cbest+cwidth+cstep, cstep )
        abest, bbest, cbest, corrbest = fit_using_crosscorr(pixels=subpix1, raw_spec=subset_comp, comp_highres_fluxes=fluxes, \
                                                            avals=avals, bvals=bvals, cvals=cvals, \
                                                            calib_wave_start=waves[0], flux_wave_precision=precision,\
                                                                      print_itters=print_itters)

        itter = 0
        dcorr = 1.
        while dcorr > convergence_criteria:
            itter += 1
            if print_itters:
                print("\nItter {:d} results:".format(itter+2))
            last_corrbest = corrbest
            incremental_res_div = 2.
            astep, bstep, cstep = astep/incremental_res_div, bstep/incremental_res_div, cstep/incremental_res_div
            awidth,bwidth,cwidth = awidth/incremental_res_div,bwidth/incremental_res_div,cwidth/incremental_res_div
            avals = ( abest-awidth, abest+awidth+astep, astep )
            bvals = ( bbest-bwidth, bbest+bwidth+bstep, bstep )
            cvals = ( cbest-cwidth, cbest+cwidth+cstep, cstep )
            abest_itt, bbest_itt, cbest_itt, corrbest = fit_using_crosscorr(pixels=pixels, raw_spec=comp_spec, comp_highres_fluxes=fluxes, \
                                                                avals=avals, bvals=bvals, cvals=cvals, \
                                                                calib_wave_start=waves[0], flux_wave_precision=precision,\
                                                                      print_itters=print_itters)
            if corrbest > last_corrbest:
                abest,bbest,cbest = abest_itt, bbest_itt, cbest_itt

            dcorr = np.abs(corrbest-last_corrbest)

        print("\n\n", fiber_identifier)
        print("--> Results:   a={:.2f}, b={:.5f}, c={:.2e}".format(abest, bbest, cbest))

        all_coefs[fiber_identifier] = [abest, bbest, cbest, 0., 0., 0.]
        all_flags[fiber_identifier] = corrbest

    return Table(all_coefs)


def fit_using_crosscorr(pixels, raw_spec, comp_highres_fluxes, avals, bvals, cvals, calib_wave_start, flux_wave_precision,print_itters):
    alow, ahigh, astep = avals
    blow, bhigh, bstep = bvals
    clow, chigh, cstep = cvals

    pix1 = pixels
    pix2 = pixels*pixels
    prec_multiplier = int(1/flux_wave_precision)
    if print_itters:
        print("--> Looking for best fit within:   a=({:.2f}, {:.2f})  b=({:.5f}, {:.5f})  c=({:.2e}, {:.2e})  with steps=({:.2f}, {:.1e}, {:.1e})".format(alow, ahigh-astep,\
                                                                                                      blow, bhigh-bstep,\
                                                                                                      clow, chigh-cstep,\
                                                                                                      astep, bstep,cstep))

    aitterbest, bitterbest, citterbest,corrbest = 0., 0., 0.,0.
    for b in np.arange(blow,bhigh,bstep):
        pixb = b * pix1
        for c in np.arange(clow,chigh,cstep):
            pixbc = pixb + (c * pix2)
            pixinds = (pixbc * prec_multiplier).astype(int)
            for a in np.arange(alow,ahigh,astep):
                indoffset = int((a - calib_wave_start) * prec_multiplier)
                synthwaveinds = pixinds + indoffset
                cut_comp_spec = raw_spec
                if synthwaveinds[-40] < 0. or synthwaveinds[40] >= len(comp_highres_fluxes):
                    continue
                elif synthwaveinds[0] < 0. or synthwaveinds[-1] >= len(comp_highres_fluxes):
                    waverestrict_cut = np.argwhere(((synthwaveinds >= 0) & (synthwaveinds < len(comp_highres_fluxes))))[0]
                    synthwaveinds = synthwaveinds[waverestrict_cut]
                    cut_comp_spec = raw_spec[waverestrict_cut]
                synthflux = comp_highres_fluxes[synthwaveinds]
                #corr, pval = pearsonr(synthflux, cut_comp_spec)
                corrs = np.correlate(synthflux, cut_comp_spec)
                corr = np.sqrt(np.dot(corrs,corrs))
                if corr > corrbest:
                    aitterbest, bitterbest, citterbest, corrbest = a, b, c, corr

    if print_itters:
        if (aitterbest == alow) or (aitterbest == (ahigh-astep)):
            if ahigh != alow+astep:
                print("!--> Warning: best fit return a boundary of the search region: alow={:.2f}, ahigh={:.2f}, abest={:.2f}".format(alow,ahigh,aitterbest))
        if (bitterbest == blow) or (bitterbest == (bhigh-bstep)):
            if bhigh != blow+bstep:
                print("!--> Warning: best fit return a boundary of the search region: blow={:.5f}, bhigh={:.5f}, bbest={:.5f}".format(blow,bhigh,bitterbest))
        if (citterbest == clow) or (citterbest == (chigh-cstep)):
            if chigh != clow+cstep:
              print("!--> Warning: best fit return a boundary of the search region: clow={:.2e}, chigh={:.2e}, cbest={:.2e}".format(clow,chigh,citterbest))

        print("--> --> Best fit correlation value: {}    with fits a={:.2f}, b={:.5f}, c={:.2e}".format(corrbest,aitterbest, bitterbest, citterbest))

    return aitterbest, bitterbest, citterbest, corrbest

def gaussian(x0,height,xs):
    width = 0.01+np.log(height)/np.log(500.)
    twosig2 = 2.*width*width
    dx = xs-x0
    fluxes = height*np.exp(-(dx*dx)/twosig2)
    return fluxes

def generate_synthetic_spectra(compdict,compnames=[],precision=1.e-4,maxheight=1000.):
    import matplotlib.pyplot as plt
    heights,waves = [],[]

    for compname in compnames:
        itterwaves,itterheights = compdict[compname]
        normalized_height = np.asarray(itterheights).astype(np.float64)/np.max(itterheights)
        waves.extend(np.asarray(itterwaves.astype(np.float64)).tolist())
        heights.extend(normalized_height.tolist())

    wave_order = np.argsort(waves)
    heights = np.asarray(heights)[wave_order]
    waves = np.asarray(waves)[wave_order]

    wavelengths = np.arange(np.floor(waves.min()),np.ceil(waves.max()),precision).astype(np.float64)
    fluxes = np.zeros(len(wavelengths)).astype(np.float64)

    for center,height in zip(waves,heights):
        modheight = maxheight*height
        itterflux = gaussian(center,modheight,wavelengths)
        fluxes = fluxes + itterflux

    #plt.figure(); plt.plot(wavelengths,fluxes,'r-'); plt.plot(waves,maxheight*heights,'b.'); plt.show()
    return wavelengths,fluxes








def compare_outputs(raw_data,table1,table2):
    def waves(pixels, a, b, c):
        return a + (b * pixels) + (c * pixels * pixels)
    fib1s = set(table1.colnames)
    fib2s = set(table2.colnames)
    matches = fib1s.intersection(fib2s)

    for match in matches:
        pixels = np.arange(len(raw_data[match])).astype(np.float64)
        a1,b1,c1,d1,e1,f1 = table1[match]
        a2, b2, c2, d2, e2, f2 = table2[match]
        waves1 = waves(pixels, a1, b1, c1)
        waves2 = waves(pixels, a2, b2, c2)
        dwaves = waves1-waves2
        print("\n"+match)
        print("--> Max deviation: {}  mean: {}  median: {}".format(dwaves[np.argmax(np.abs(dwaves))], np.mean(np.abs(dwaves)), np.median(np.abs(dwaves))))
        plt.figure()
        plt.plot(pixels, dwaves, 'r-')
        plt.show()

def automated_calib_wrapper_script(input_dict):
    return run_automated_calibration(**input_dict)


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
    if default_vals is not None:
        adef,bdef,cdef,ddef,edef,fdef = default_vals[fiber_identifier]
        default_dict['default'] = (adef,bdef,cdef)
    else:
        adef, bdef, cdef, ddef, edef, fdef = 4523.4,1.0007,-1.6e-6,0.,0.,0.
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


def run_interactive_slider_calibration(coarse_comp, complinelistdict, default_vals=None,history_vals=None,\
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



def wavelength_fitting_by_line_selection(self, comp, selectedlistdict, fulllinelist, coef_table, select_lines = False, bounds=None):
    if select_lines:
        wm, fm = [], []
        for key,(keys_wm,keys_fm) in selectedlistdict.items():
            if key in['ThAr','Th']:
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

    def iterate_fib(fib):
        tetn = int(fib[1])
        fibn = int(fib[2:])
        if tetn == 8 and fibn >= 8:
            fibn -= 1
        elif tetn == 4 and fibn >= 8:
            fibn -= 1
        else:
            fibn += 1
            if fibn > 16:
                tetn += 1
                fibn = 1
        outfib = '{}{}{:02d}'.format(cam, tetn, fibn)
        return outfib

    def ensure_match(fib, allfibs, subset, cam):
        print(fib)
        outfib = fib
        if outfib not in allfibs:
            outfib = iterate_fib(outfib)
            outfib = ensure_match(outfib, allfibs, subset, cam)
        if outfib in subset:
            outfib = iterate_fib(outfib)
            outfib = ensure_match(outfib, allfibs, subset, cam)
        return outfib

    cam = comp.colnames[0][0]
    specific_set = [cam+'101',cam+'816',cam+'416',cam+'501']
    hand_fit_subset = []
    for i,fib in enumerate(specific_set):
        outfib = ensure_match(fib,comp.colnames,hand_fit_subset,cam)
        hand_fit_subset.append(outfib)
    seed = 10294
    np.random.seed(seed)
    randfibs = ['{:02d}'.format(x) for x in np.random.randint(1, 16, 4)]
    for tetn,fibn in zip([2,3,6,7],randfibs):
        fib = '{}{}{}'.format(cam,tetn,fibn)
        outfib = ensure_match(fib, comp.colnames, hand_fit_subset, cam)
        hand_fit_subset.append(outfib)

    #hand_fit_subset = [cam+'101',cam+'416',cam+'816']
    hand_fit_subset = np.asarray(hand_fit_subset)
    extrema_fiber = False
    for fiber in hand_fit_subset:#''r401','r801']:hand_fit_subset
        if fiber[1:] in ['101','816','501','416']:
            extrema_fiber = True
        else:
            extrema_fiber = False
        counter += 1
        f_x = comp[fiber].data
        coefs = coef_table[fiber]
        iteration_wm, iteration_fm = wm.copy(), fm.copy()

        if len(all_coefs.keys())>0:
            coef_devs = np.zeros(len(coefs)).astype(np.float64)
            for key,key_coefs in all_coefs.items():
                dev = np.asarray(key_coefs)-np.asarray(coef_table[key])
                coef_devs += dev
            coef_devs /= len(all_coefs.keys())

            updated_coefs = coefs+coef_devs
        else:
            updated_coefs = coefs

        browser = LineBrowser(iteration_wm,iteration_fm, f_x, updated_coefs, fulllinelist, bounds=bounds,edge_line_distance=10.0)
        if np.any((np.asarray(browser.line_matches['lines'])-np.asarray(browser.line_matches['peaks_w']))>0.5):
            browser.plot()
        params,covs = browser.fit()

        print(fiber,*params)
        all_coefs[fiber] = params
        variances[fiber] = covs.diagonal()
        print(np.sum(variances[fiber]))

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
            if extrema_fiber:
                deleted_wm,deleted_fm = deleted_wm_srt, deleted_fm_srt
            else:
                mask_wm_nearedge = ((deleted_wm_srt>(browser.xspectra[0]+10.0)) & (deleted_wm_srt<(browser.xspectra[-1]-10.0)))
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

            browser = LineBrowser(iteration_wm, iteration_fm, f_x, coefs, fulllinelist, bounds=bounds,edge_line_distance=-20.0)
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


    numeric_hand_fit_names = np.asarray([ 16*int(fiber[1])+int(fiber[2:]) for fiber in hand_fit_subset])

    last_fiber = cam+'101'

    all_fibers = np.sort(list(comp.colnames))
    for fiber in all_fibers:
        if fiber in hand_fit_subset:
            continue
        if fiber not in coef_table.colnames:
            continue
        coefs = np.asarray(coef_table[fiber])
        f_x = comp[fiber].data

        fibern = 16*int(fiber[1])+int(fiber[2:])

        nearest_fibs = hand_fit_subset[np.argsort(np.abs(fibern-numeric_hand_fit_names))[:2]]
        diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
        diffs_fib2 = np.asarray(all_coefs[nearest_fibs[1]]) - np.asarray(coef_table[nearest_fibs[1]])

        nearest_fib = np.asarray(all_coefs[last_fiber]) - np.asarray(coef_table[last_fiber])

        diffs_mean = (0.25*diffs_fib1)+(0.25*diffs_fib2)+(0.5*nearest_fib)

        adjusted_coefs_guess = coefs+diffs_mean
        browser = LineBrowser(wm,fm, f_x, adjusted_coefs_guess, fulllinelist, bounds=None, edge_line_distance=-20.0)

        params,covs = browser.fit()

        plt.close()
        browser.create_saveplot(params, covs, 'fiberfits/{}'.format(fiber))
        print('\n\n',fiber,*params)
        all_coefs[fiber] = params
        variances[fiber] = covs.diagonal()
        normd_vars = variances[fiber]/(params*params)
        print(np.sqrt(np.sum(normd_vars)))
        print(np.sqrt(normd_vars))

        #savename = '{}'.format(fiber)
        #browser.create_saveplot(params,covs, savename)

        app_fit_pix[fiber] = browser.line_matches['peaks_p']
        app_fit_lambs[fiber] = browser.line_matches['lines']
        del browser
        last_fiber = fiber

    if not select_lines:
        app_specific_linelists = None
    return all_coefs, app_specific_linelists, app_fit_lambs, app_fit_pix, variances