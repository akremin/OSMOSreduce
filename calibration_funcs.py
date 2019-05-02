
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
##  a zoom in window
##  mutlicursor
## And checkboxes
## and radio buttons
from matplotlib.widgets import MultiCursor
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider, Button

from astropy.table import Table

from scipy.signal import medfilt
from scipy.signal import find_peaks
import numpy as np
from scipy.signal import argrelmax

deltat = np.datetime64('now' ,'m').astype(int) -np.datetime64('2018-06-01T00:00' ,'m').astype(int)
print(deltat)



import numpy as np

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

def run_automated_calibration(coarse_comp, complinelistdict, last_obs=None, print_itters = True, only_use_peaks = True):
    precision = 1e-4
    convergence_criteria = 1.0e-5 # change in correlation value from itteration to itteration
    waves, fluxes = generate_synthetic_spectra(complinelistdict, compnames=['HgAr', 'NeAr','Xe'],precision=precision,\
                                               maxheight=10000.,minwave=3400,maxwave=7500)

    ## Make sure the information is in astropy table format
    coarse_comp = Table(coarse_comp)
    # for fib in ['r209','r711','r808']:
    #     if fib in coarse_comp.colnames:
    #         coarse_comp.remove_column(fib)
    ## Define loop params
    counter = 0

    ## Initiate arrays/dicts for later appending inside loop (for keeping in scope)
    all_coefs = {}
    all_flags = {}

    ## Loop over fiber names (strings e.g. 'r101')
    ##hack!
    cam = coarse_comp.colnames[0][0]

    if cam =='b':
        numerics = np.asarray([(16 * (9 - int(fiber[1]))) + int(fiber[2:]) for fiber in coarse_comp.colnames])
    else:
        numerics = np.asarray([(16 * int(fiber[1])) + int(fiber[2:]) for fiber in coarse_comp.colnames])

    sorted = np.argsort(numerics)
    fibernames = np.array(coarse_comp.colnames)[sorted]

    if cam == 'r' and int(fibernames[0][1]) > 3:
        fibernames = fibernames[::-1]
    elif cam =='b' and int(fibernames[0][1]) < 6:
        fibernames = fibernames[::-1]

    burnin = 6;
    fibernames = np.concatenate([fibernames[:burnin], fibernames[1:burnin - 1][::-1], fibernames])

    for counter,fiber_identifier in enumerate(fibernames):#['r101','r408','r409','r608','r816']:
        #print("\n\n", fiber_identifier)

        ## Get the spectra (column with fiber name as column name)
        comp_spec = np.asarray(coarse_comp[fiber_identifier])

        if only_use_peaks:
            ## Find just the peaks in the calibration spectrum
            c_peak_inds, c_peak_props = find_peaks(comp_spec, height=(400, None), width=(0.1, 20), \
                                                   threshold=(None, None),
                                                   prominence=(200, None), wlen=101)

            ## create pixel array for mapping to wavelength
            c_peak_inds = np.asarray(c_peak_inds)
            np.random.seed(int(np.sum(comp_spec)))
            randoms = np.random.randint(low=0,high=len(comp_spec),size=len(comp_spec)//4)
            #randoms = np.arange(len(comp_spec))[::2]
            pix1 = np.concatenate((c_peak_inds,c_peak_inds-1,c_peak_inds+1,c_peak_inds-2,c_peak_inds+2,randoms))
            pix1 = np.unique(np.sort(pix1))
            pix1 = pix1[pix1<len(comp_spec)]
            comp_spec = comp_spec[pix1]
            #comp_spec[pix1<400] *= 100.
        else:
            pix1 = np.arange(len(comp_spec))

        abest, bbest, cbest, corrbest = 0., 0., 0., 0.
        alow, ahigh = 3000, 8000
        awidth, bwidth, cwidth = 12, 0.005, 5.0e-6

        if last_obs is None or fiber_identifier not in last_obs.keys():
            if counter == 0:
                avals = (alow, ahigh+1, 1)
                bvals = (1.0,1.01,0.1)
                cvals = (0, 1.0, 2)
                if print_itters:
                    print("\nItter 1 results, (fixing c to 0.):")
                abest, trashb, trashc, corrbest = fit_using_crosscorr(pixels=pix1, raw_spec=comp_spec,
                                                                      comp_highres_fluxes=fluxes, \
                                                                      avals=avals, bvals=bvals, cvals=cvals, \
                                                                      calib_wave_start=waves[0],
                                                                      flux_wave_precision=precision,\
                                                                      print_itters=print_itters)


                avals = (abest-40, abest+41, 0.5)
                bvals = (0.96,1.04,0.005)
                cvals = (-1.0e-5, 1.0e-5, 1.0e-6)
                if print_itters:
                    print("\nItter 1 results, (fixing c to 0.):")
                abest, bbest, cbest, corrbest = fit_using_crosscorr(pixels=pix1, raw_spec=comp_spec,
                                                                      comp_highres_fluxes=fluxes, \
                                                                      avals=avals, bvals=bvals, cvals=cvals, \
                                                                      calib_wave_start=waves[0],
                                                                      flux_wave_precision=precision,\
                                                                      print_itters=print_itters)

            elif counter < 3:
                compare_fiber = fibernames[counter - 1]
                [trasha, bbest, cbest, trash1, trash2, trash3] = all_coefs[compare_fiber]
                if (bbest < 0.96) or (bbest>1.04):
                    bbest = 1.0
                    cbest = 0.
                astep,bstep,cstep = 0.5, 0.04, 8.0e-6
                avals = (alow,   ahigh+astep,  astep)
                bvals = (bbest , bbest+bstep , bstep)
                cvals = (cbest , cbest+cstep , cstep)
                if print_itters:
                    print("\nItter 1 results, (fixing b and c to past vals):")
                abest, trashb, trashc, corrbest = fit_using_crosscorr(pixels=pix1, raw_spec=comp_spec,
                                                                    comp_highres_fluxes=fluxes, \
                                                                    avals=avals, bvals=bvals, cvals=cvals, \
                                                                    calib_wave_start=waves[0],
                                                                    flux_wave_precision=precision,\
                                                                      print_itters=print_itters)
            else:
                compare_fiber1 = fibernames[counter - 1]
                [a1, b1, c1, trash1, trash2, trash3] = all_coefs[compare_fiber1]
                compare_fiber2 = fibernames[counter - 2]
                [a2, b2, c2, trash1, trash2, trash3] = all_coefs[compare_fiber2]
                compare_fiber3 = fibernames[counter - 3]
                [a3, b3, c3, trash1, trash2, trash3] = all_coefs[compare_fiber3]
                abest = np.median([a1, a2, a3])
                bbest = np.median([b1, b2, b3])
                cbest = np.median([c1, c2, c3])

                if (bbest < 0.96) or (bbest>1.04):
                    bbest = 1.0
                    cbest = 0.
                astep,bstep,cstep = 0.5, 0.04, 8.0e-6
                aw = 30
                avals = (abest-aw,   abest+aw+astep,  astep)
                bvals = (bbest , bbest+bstep , bstep)
                cvals = (cbest , cbest+cstep , cstep)
                if print_itters:
                    print("\nItter 1 results, (fixing b and c to past vals):")
                abest, trashb, trashc, corrbest = fit_using_crosscorr(pixels=pix1, raw_spec=comp_spec,
                                                                    comp_highres_fluxes=fluxes, \
                                                                    avals=avals, bvals=bvals, cvals=cvals, \
                                                                    calib_wave_start=waves[0],
                                                                    flux_wave_precision=precision,\
                                                                      print_itters=print_itters)

                #awidth, bwidth, cwidth = 10,\
                #                         max([  min([  bwidth,np.max(bbest)-np.min(bbest)  ]),  1.0e-3    ]),\
                #                         max([  min([  cwidth,np.max(cbest)-np.min(cbest)  ]),  1.0e-7   ])

        else:
            [abest, bbest, cbest, trash1, trash2, trash3] = last_obs[fiber_identifier]
            awidth, bwidth, cwidth = awidth/2., bwidth/2., cwidth/2.
            if print_itters:
                print("\nItter 1 results:")
                print("--> Using previous obs value of:   a={:.2f}, b={:.5f}, c={:.2e}".format(abest, bbest, cbest))

        astep,bstep,cstep = 0.5, bwidth/10., cwidth/10.

        dcorr = 1.
        for itter in range(100):
            if print_itters:
                print("\nItter {:d} results:".format(itter+2))
            last_corrbest = corrbest
            incremental_res_div = 4.
            astep, bstep, cstep = astep/incremental_res_div, bstep/incremental_res_div, cstep/incremental_res_div
            awidth,bwidth,cwidth = awidth/incremental_res_div,bwidth/incremental_res_div,cwidth/incremental_res_div
            avals = ( abest-awidth, abest+awidth+astep, astep )
            bvals = ( bbest-bwidth, bbest+bwidth+bstep, bstep )
            cvals = ( cbest-cwidth, cbest+cwidth+cstep, cstep )
            abest_itt, bbest_itt, cbest_itt, corrbest = fit_using_crosscorr(pixels=pix1, raw_spec=comp_spec, comp_highres_fluxes=fluxes, \
                                                                avals=avals, bvals=bvals, cvals=cvals, \
                                                                calib_wave_start=waves[0], flux_wave_precision=precision,\
                                                                      print_itters=print_itters)
            if corrbest > last_corrbest:
                abest,bbest,cbest = abest_itt, bbest_itt, cbest_itt

            dcorr = np.abs(corrbest-last_corrbest)
            if dcorr < convergence_criteria:
                break

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
                if len(synthwaveinds) == 0:
                    continue
                elif len(synthwaveinds)>40 and (synthwaveinds[-40] < 0. or synthwaveinds[40] >= len(comp_highres_fluxes)):
                    continue
                elif synthwaveinds[0] < 0. or synthwaveinds[-1] >= len(comp_highres_fluxes):
                    waverestrict_cut = np.where(((synthwaveinds >= 0) & (synthwaveinds < len(comp_highres_fluxes))))[0]
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
    width = 1.0+0.2*np.log(height)/np.log(500.)
    twosig2 = 2.*width*width
    dx = xs-x0
    fluxes = height*np.exp(-(dx*dx)/twosig2)
    return fluxes

def generate_synthetic_spectra(compdict,compnames=[],precision=1.e-4,maxheight=10000.,\
                               minwave = 3400, maxwave = 7500):
    heights,waves = [],[]

    for compname in compnames:
        itterwaves,itterheights = compdict[compname]
        normalized_height = maxheight*np.asarray(itterheights).astype(np.float)/np.max(itterheights)
        if compname == 'Xe':
            normalized_height = normalized_height / 100.
        waves.extend(list(itterwaves))
        heights.extend(normalized_height.tolist())

    wave_order = np.argsort(waves)
    heights = np.asarray(heights)[wave_order]
    waves = np.asarray(waves)[wave_order]
    wavecut = ((waves>minwave)&(waves<maxwave))
    heights,waves = heights[wavecut],waves[wavecut]

    minwave,maxwave = np.floor(waves.min()),np.ceil(waves.max())
    wavelengths = np.arange(minwave,maxwave,precision)
    fluxes = np.zeros(len(wavelengths)).astype(np.float)

    idxs = ((waves - minwave) / precision).astype(int)
    idx_width = int(20/precision)
    for idx,center,height in zip(idxs,waves,heights):
        itterwaves = wavelengths[idx-idx_width:idx+idx_width+1]
        itterflux = gaussian(center,height,itterwaves)
        fluxes[idx-idx_width:idx+idx_width+1] += itterflux

    #plt.figure(); plt.plot(wavelengths,fluxes,'r-'); plt.plot(waves,maxheight*heights,'b.'); plt.show()
    return wavelengths,fluxes

def compare_outputs(raw_data,table1,table2,save_template='{fiber}.png',save_plots=True,show_plots=False):
    def waves(pixels, a, b, c,d,e,f):
        return a + (b * pixels) + (c * pixels * pixels)+\
               (d * np.power(pixels,3)) + (e * np.power(pixels,4))+ \
               (f * np.power(pixels, 5))
    fib1s = set(table1.colnames)
    fib2s = set(table2.colnames)
    matches = fib1s.intersection(fib2s)

    for match in matches:
        pixels = np.arange(len(raw_data[match])).astype(np.float64)
        a1,b1,c1,d1,e1,f1 = table1[match]
        a2, b2, c2, d2, e2, f2 = table2[match]
        waves1 = waves(pixels, a1,b1,c1,d1,e1,f1)
        waves2 = waves(pixels, a2, b2, c2, d2, e2, f2)
        dwaves = waves1-waves2
        print("\n"+match)
        print("--> Max deviation: {}  mean: {}  median: {}".format(dwaves[np.argmax(np.abs(dwaves))], np.mean(np.abs(dwaves)), np.median(np.abs(dwaves))))
        plt.figure()
        plt.plot(pixels, dwaves, 'r-')
        plt.title("{}  max dev={}".format(match,np.max(np.abs(dwaves))))
        plt.ylabel("Fit 1 - Fit 2 [Angstrom]")
        plt.xlabel("Pixel")
        if save_plots:
            plt.savefig(save_template.format(fiber=match))
        if show_plots:
            plt.show()
        plt.close()
    return matches

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


def interactive_plot(pixels,spectra,linelistdict,gal_identifier,\
                     default_dict, steps, default_key):
    fig = plt.figure()
    axsrc = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    axzoom = plt.subplot2grid((1, 3), (0, 2))

    bigwin_cal_lines, zoom_cal_lines = [], []
    colors = ['r', 'g', 'k', 'orange', 'm','y','m'][:len(linelistdict)]
    visibility = [True]*len(linelistdict)
    min_wavelengths,max_wavelengths = [],[]
    min_flux,max_flux = np.min(spectra),np.max(spectra)
    labels = []
    ## unpack lines
    for (linelabel,(wave,flux)),color in zip(linelistdict.items(),colors):
        ## signal
        #sl, = axsrc.plot(wave, flux, lw=2, color=color, label=linelabel)
        linewidths = 4*(flux/np.max(flux))
        linewidths = 3*(np.log(linewidths-np.min(linewidths)+1.))
        sl = axsrc.vlines(wave, ymin=min_flux,ymax=max_flux,lw=linewidths, color=color, label=linelabel)
        bigwin_cal_lines.append(sl)
        ## zoomed
        #zl, = axzoom.plot(wave, flux, lw=2, color=color, label=linelabel)
        zl = axzoom.vlines(wave,  ymin=min_flux,ymax=max_flux,lw=linewidths, color=color, label=linelabel)
        zoom_cal_lines.append(zl)
        min_wavelengths.append(wave[0])
        max_wavelengths.append(wave[-1])
        labels.append(linelabel)

    # from utility_funcs import VertSlider
    max_wavelength = max(max_wavelengths)+400
    min_wavelength = min(min_wavelengths)-400
    wavelength_halfwidth = max_wavelength//50

    spectra_is_good = True
    axcolor = 'lightgoldenrodyellow'

    coefs = {}
    coefs['a'], coefs['b'], coefs['c'] = default_dict[default_key]
    #coefs['d'], coefs['e'], coefs['f'] = 0, 0, 0

    waves = pix_to_wave(pixels, coefs)
    tog1 = spectra
    tog2 = medfilt(tog1, 3)

    smooth_noise_dict = {'Original': tog1, 'Smooth': tog2}

    meanwave = (max_wavelength   + min_wavelength) // 2

    # fig, (axsrc, axzoom) = plt.subplots(nrows=1,ncols=2)


    ## Signal
    stogl, = axsrc.plot(waves, tog1, lw=2, color='b', label='Original')
    ztogl, = axzoom.plot(waves, tog1, lw=2, color='b', label='Original')


    ## Button and slider funcs
    def showunshow_lines(label):
        index = labels.index(label)
        bigwin_cal_lines[index].set_visible(not bigwin_cal_lines[index].get_visible())
        zoom_cal_lines[index].set_visible(not zoom_cal_lines[index].get_visible())
        plt.draw()


    def zoom_adjust(event):
        if event.button != 1:
            return
        elif (event.inaxes == axsrc or event.inaxes == axzoom):
            x, y = event.xdata, event.ydata
            axzoom.set_xlim(x - wavelength_halfwidth, x + wavelength_halfwidth)
            fig.canvas.draw()


    def smooth_noise_flip(label):
        ydata = smooth_noise_dict[label]
        stogl.set_ydata(ydata)
        ztogl.set_ydata(ydata)
        plt.draw()


    def change_default_sliderset(label):
        def_off, def_off_fine, def_lin, def_lin_fine, def_quad_fine = split_params(*(default_dict[label]), \
                                                                                   *steps)

        off_slide.valinit, lin_slide.valinit = def_off, def_lin
        off_slide.vline.set_xdata([def_off] * len(off_slide.vline.get_xdata()))
        lin_slide.vline.set_xdata([def_lin] * len(lin_slide.vline.get_xdata()))

        off_slide_fine.valinit, lin_slide_fine.valinit = def_off_fine, def_lin_fine
        off_slide_fine.vline.set_xdata([def_off_fine] * len(off_slide.vline.get_xdata()))
        lin_slide_fine.vline.set_xdata([def_lin_fine] * len(lin_slide.vline.get_xdata()))

        quad_slide_fine.valinit = def_quad_fine
        quad_slide_fine.vline.set_xdata([def_quad_fine]*len(lin_slide.vline.get_xdata()))

        plt.draw()


    def slider_spec_update(val):
        coefs['a'] = off_slide.val + off_slide_fine.val
        coefs['b'] = lin_slide.val + lin_slide_fine.val
        coefs['c'] = quad_slide_fine.val  # quad_slide.val+quad_slide_fine.val
        waves = pix_to_wave(pixels, coefs)
        stogl.set_xdata(waves)
        ztogl.set_xdata(waves)
        fig.canvas.draw_idle()


    def reset_sliders(event):
        lin_slide.reset()
        off_slide.reset()
        # quad_slide.reset()
        lin_slide_fine.reset()
        off_slide_fine.reset()
        quad_slide_fine.reset()
        slider_spec_update(None)


    def flag_spec(event):
        print("\n\tSpec flagged as bad\n")
        spectra_is_good = False
        plt.close()


    def save_and_close(event):
        print("\n\tSpec closed with the following params: a={} b={} c={}\n".format(coefs['a'], coefs['b'], \
                                                                             coefs['c']))
        plt.close()


    def print_to_screen(event):
        print("\ta={} b={} c={}".format(coefs['a'], coefs['b'], coefs['c']))


    ## Make checkbuttons with all plotted lines with correct visibility
    ## [x,y,width,height]
    plot_ystart = 0.36

    slider_ystart = 0.03

    slider_xstart = 0.04
    boxes_xstart_row1 = 0.7
    boxes_xstart_row2 = 0.8
    boxes_xstart_row3 = 0.92

    box_width = 0.08
    box_width2 = 0.14
    slider_width = 0.62

    height2 = 0.1
    height3 = 0.15
    height_slider = 0.03
    height_button = 0.04

    ## Move subplot over to make room for checkboxes
    plt.subplots_adjust(left=slider_xstart, right=1 - slider_xstart, \
                        bottom=plot_ystart, top=1 - slider_xstart)

    ## Change the name and limits of each axis
    axsrc.set(xlim=(min_wavelength, max_wavelength), ylim=(min_flux,max_flux), autoscale_on=False,
              title='Click to zoom')
    axzoom.set(xlim=(meanwave - wavelength_halfwidth, meanwave + wavelength_halfwidth), \
               ylim=(min_flux,max_flux), autoscale_on=False, title='Zoom window')

    ## Setup button locations
    # slider1_rax = plt.axes([slider_xstart, slider_ystart+10*height_slider, slider_width, height_slider], facecolor=axcolor)
    slider2_rax = plt.axes([slider_xstart, slider_ystart + 8 * height_slider, slider_width, height_slider],
                           facecolor=axcolor)
    slider3_rax = plt.axes([slider_xstart, slider_ystart + 6 * height_slider, slider_width, height_slider],
                           facecolor=axcolor)
    slider4_rax = plt.axes([slider_xstart, slider_ystart + 4 * height_slider, slider_width, height_slider],
                           facecolor=axcolor)
    slider5_rax = plt.axes([slider_xstart, slider_ystart + 2 * height_slider, slider_width, height_slider],
                           facecolor=axcolor)
    slider6_rax = plt.axes([slider_xstart, slider_ystart, slider_width, height_slider], facecolor=axcolor)

    linelists_rax = plt.axes([boxes_xstart_row1, slider_ystart, box_width, height3], facecolor=axcolor)

    spec_radio_rax = plt.axes([boxes_xstart_row1, 0.20, box_width, height2], facecolor=axcolor)
    def_radio_rax = plt.axes([boxes_xstart_row2, 0.15, box_width2, height3], facecolor=axcolor)

    close_rax = plt.axes([boxes_xstart_row3, slider_ystart, 0.05, height_button])
    reset_rax = plt.axes([boxes_xstart_row3, slider_ystart + 0.06, 0.05, height_button])
    flag_rax = plt.axes([boxes_xstart_row2, slider_ystart, 0.1, height_button])
    print_rax = plt.axes([boxes_xstart_row2, slider_ystart + 0.06, 0.1, height_button])

    ## Checkboxes
    linelist_check = CheckButtons(linelists_rax, labels, visibility)
    ## Radio boxes
    spec_radio = RadioButtons(spec_radio_rax, ['Original', 'Smooth'],active=[0])
    def_keys = list(default_dict.keys())
    def_key_index = def_keys.index(default_key)
    def_radio = RadioButtons(def_radio_rax, def_keys,active=def_key_index)
    ## Sliders

    def_off,def_off_fine, def_lin, def_lin_fine, def_quad_fine = split_params(*(default_dict[default_key]),\
                                                                              *steps)
    off_slide = Slider(slider2_rax, 'offset', -2000., 10000.0, valinit=def_off, valstep=steps[0])
    lin_slide = Slider(slider3_rax, 'stretch', 0.4,2.5, valinit=def_lin, valstep=steps[1])
    # quad_slide = Slider(slider3_rax, 'quad', -10.0, 10.0, valinit=default_dict[default_key][2], valstep=steps[2])
    off_slide_fine = Slider(slider4_rax, 'fine offset', -200, 200, valinit=def_off_fine, valstep=steps[0] / 100)
    lin_slide_fine = Slider(slider5_rax, 'fine stretch', -0.05,0.05, valinit=def_lin_fine, valstep=steps[1] / 100, \
                        valfmt='%1.4f')
    quad_slide_fine = Slider(slider6_rax, 'fine quad',-4e-5,4e-5, valinit=def_quad_fine, valstep=steps[2] / 100,\
                        valfmt='%1.6f')

    ## Buttons
    reset_button = Button(reset_rax, 'Reset', color=axcolor, hovercolor='0.975')
    flag_button = Button(flag_rax, 'Flag as Bad', color=axcolor, hovercolor='0.975')
    close_button = Button(close_rax, 'Close', color=axcolor, hovercolor='0.975')
    print_button = Button(print_rax, 'Print to Terminal', color=axcolor, hovercolor='0.975')

    ## Run the interactive buttons
    fig.canvas.mpl_connect('button_press_event', zoom_adjust)
    linelist_check.on_clicked(showunshow_lines)
    spec_radio.on_clicked(smooth_noise_flip)
    def_radio.on_clicked(change_default_sliderset)

    lin_slide.on_changed(slider_spec_update)
    off_slide.on_changed(slider_spec_update)
    # quad_slide.on_changed(slider_spec_update)
    lin_slide_fine.on_changed(slider_spec_update)
    off_slide_fine.on_changed(slider_spec_update)
    quad_slide_fine.on_changed(slider_spec_update)

    reset_button.on_clicked(reset_sliders)
    flag_button.on_clicked(flag_spec)
    close_button.on_clicked(save_and_close)
    print_button.on_clicked(print_to_screen)
    multi = MultiCursor(fig.canvas, (axsrc, axzoom), color='r', lw=1)

    ## plot the final canvas in a pop-up window
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    return spectra_is_good, coefs



def pix_to_wave(xs, coefs):
    return coefs['a'] + coefs['b'] * xs + coefs['c'] * np.power(xs, 2) #+ \
           #coefs['d'] * np.power(xs, 3) + coefs['e'] * np.power(xs, 4) + \
           #coefs['f'] * np.power(xs, 5)

def pix_to_wave_fifthorder(xs, coefs):
    return coefs[0] + coefs[1] * xs + coefs[2] * np.power(xs, 2) + \
           coefs[3] * np.power(xs, 3) + coefs[4] * np.power(xs, 4) + \
           coefs[5] * np.power(xs, 5)


def iterate_fib(fib,cam):
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
        outfib = iterate_fib(outfib,cam)
        outfib = ensure_match(outfib, allfibs, subset, cam)
    if outfib in subset:
        outfib = iterate_fib(outfib,cam)
        outfib = ensure_match(outfib, allfibs, subset, cam)
    return outfib

def find_devs(table1,table2):
    xs = np.arange(2000).astype(np.float64)
    overlaps = list(set(list(table1.colnames)).intersection(set(list(table2.colnames))))
    devs = []
    for fib in overlaps:
        coef_dev = np.asarray(table1[fib])-np.asarray(table2[fib])
        full_devs = np.polyval(coef_dev[::-1],xs)
        dev = np.std(full_devs)
        devs.append(dev)
    return np.mean(devs)

def split_params(off,lin,quad,offstep,linstep,quadstep):
    def_quad_fine = quad
    off_fine = off % offstep
    if off_fine < (offstep/1000) or off_fine > (offstep*(999/1000.)):
        def_off = off
        def_off_fine = 0.
    else:
        def_off = off - off_fine
        def_off_fine = off_fine
    lin_fine = lin % linstep
    if lin_fine < (linstep/1000) or lin_fine > (linstep*(999/1000.)):
        def_lin = lin
        def_lin_fine = 0.
    else:
        def_lin = lin - lin_fine
        def_lin_fine = lin_fine
    return def_off, def_off_fine, def_lin, def_lin_fine, def_quad_fine



if __name__ == '__main__':
    #cal_lamp = ['Hg', 'He','Ar', 'Ne', 'Xe']  ## crc
    #from calibrations import load_calibration_lines_crc_dict as load_calibration
    #cal_lamp = ['Hg', 'Ar', 'Ne','Xe'] ## nist
    #from calibrations import load_calibration_lines_nist_dict as load_calibration
    cal_lamp = ['HgAr', 'NeAr', 'Ar', 'Xe'] ## salt
    from calibrations import load_calibration_lines_salt_dict as load_calibration


    linelistdict, cal_states = load_calibration(cal_lamp,wavemincut=4500,wavemaxcut=6600)
    default = (4521,1.,0.)
    default_dict = {
        'default': default,
        'predicted from prev spec': (4521, 1., 0.),
        'predicted from history': (5625, 1., 0.),
        'from history': (5625, 1., 0.)
    }

    steps = (1, 0.01, 0.00001)
    default_key = 'predicted from prev spec'

    apcut_data = None
    import pickle as pkl
    with open('pkldump.pkl', 'rb') as pdump:
        apcut_data = pkl.load(pdump)

    specs = apcut_data['comp']['r'][628]
    fiber_identifier = 'r101'
    spectra = specs[fiber_identifier]
    interactive_plot(pixels=np.arange(len(spectra)), spectra=spectra,\
                     linelistdict=linelistdict, gal_identifier=fiber_identifier, \
                     default_dict=default_dict, steps=steps, default_key=default_key)

    cal_lamp = ['ThAr']  ## salt
    spectra = apcut_data['thar']['r'][627][fiber_identifier]
    linelistdict, cal_states = load_calibration(cal_lamp, \
                                                wavemincut=4500, \
                                                wavemaxcut=6600)
    interactive_plot(pixels=np.arange(len(spectra)), spectra=spectra,\
                     linelistdict=linelistdict, gal_identifier=fiber_identifier, \
                     default_dict=default_dict, steps=steps, default_key=default_key)


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