import os
from astropy.io import fits

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from interactive_plot import interactive_plot,pix_to_wave
from scipy.signal import argrelmax
import pickle as pkl
from multiprocessing import Pool
from wavelength_calibration import get_highestflux_waves,\
    top_peak_pixels,top_peak_wavelengths,update_default_dict



def run_automated_calibration(coarse_comp, complinelistdict, last_obs=None, print_itters = True):
    precision = 1e-2
    convergence_criteria = 1.0e-5 # change in correlation value from itteration to itteration
    waves, fluxes = generate_synthetic_spectra(complinelistdict, compnames=['HgAr', 'NeAr'],precision=precision,maxheight=10000.)
    init_default = (4523.4,1.0007,-1.6e-6)

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




def load_calibration_lines_dict(cal_lamp,wavemincut=4000,wavemaxcut=10000,use_selected=False):
    """Assumes the format of the salt linelist csvs privuded with this package"""
    from calibrations import air_to_vacuum
    #linelistdict = {}
    selectedlinesdict = {}
    print(('Using calibration lamps: ', cal_lamp))
    possibilities = ['Xe','Ar','HgNe','HgAr','NeAr','Hg','Ne','ThAr','Th']
    all_wms = []
    for lamp in possibilities:
        if lamp in cal_lamp:
            print(lamp)
            filname = lampline_template.format(mod='',lamp=lamp)
            sel_filname = lampline_template.format(mod='selected_',lamp=lamp)
            pathname = os.path.join(lampline_dir,filname)
            sel_pathname = os.path.join(lampline_dir,sel_filname)
            if use_selected and os.path.exists(sel_pathname):
                tab = Table.read(sel_pathname,format='ascii.csv',dtypes=[float,float,str,str])
            else:
                tab = Table.read(pathname, format='ascii.csv')
            fm = tab['Intensity'].data
            wm_vac = air_to_vacuum(tab['Wavelength'].data)
            boolean = np.array(tab['Use']=='Y').astype(bool)
            ## sort lines by wavelength
            sortd = np.argsort(wm_vac)
            srt_wm_vac, srt_fm, srt_bl = wm_vac[sortd], fm[sortd],boolean[sortd]
            good_waves = np.where((srt_wm_vac>=wavemincut)&(srt_wm_vac<=wavemaxcut))[0]
            out_wm_vac,out_fm_vac,out_bl = srt_wm_vac[good_waves], srt_fm[good_waves],srt_bl[good_waves]
            #linelistdict[lamp] = (out_wm_vac,out_fm_vac)
            selectedlinesdict[lamp] = (out_wm_vac[out_bl],out_fm_vac[out_bl])
            all_wms.extend(out_wm_vac.tolist())

    #return linelistdict, selectedlinesdict, all_wms
    return selectedlinesdict, np.asarray(all_wms)


def read_hdu(fileloc,filename):
    full_filename = os.path.join(fileloc,filename)
    inhdulist = fits.open(full_filename)
    if len(inhdulist)>1:
        if 'flux' in inhdulist:
            inhdu = inhdulist['flux']
        else:
            inhdu = inhdulist[1]
    else:
        inhdu = inhdulist[0]

    return inhdu

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

if __name__ == '__main__':
    ## r_calibration_basic-HgAr-NeAr-Xe_11C_628_199652
    fittype='basic-HgAr-NeAr-Xe'
    cam='r'
    config='11C'
    filenum=636#628
    #filenum_hist=628
    #timestamp=199652
    cal_lamps = ['HgAr','NeAr','Xe']

    basedir = os.path.abspath('../../')
    #calibration_template = '{cam}_calibration_{fittype}_{config}_{filenum}_{timestamp}.fits'
    path_to_mask = os.path.join(basedir,'OneDrive - umich.edu','Research','M2FSReductions','A02')
    # complete_calib_name = os.path.join(path_to_mask,'calibrations',calibration_template)
    # if 'basic' in fittype:
    #     filename = complete_calib_name.format(cam=cam, fittype=fittype, config=config, \
    #                                                 filenum=filenum, timestamp=timestamp)
    #     calib = Table.read(filename)
    # elif 'full' in fittype:
    #     filename = complete_calib_name.format(cam=cam, fittype=fittype, config=config, \
    #                                                 filenum=filenum, timestamp=timestamp)
    #     calib = fits.open(filename)

    lampline_dir = os.path.join(os.path.abspath('.'),'lamp_linelists','salt')
    lampline_template = '{mod}{lamp}.csv'
    complinelistdict,allwms = load_calibration_lines_dict(cal_lamps,wavemincut=3000,wavemaxcut=8000)

    filedir =os.path.join(path_to_mask,'oneds')
    filename = '{cam}_{imtype}_{filenum}_A02_1d_bc.fits'.format(cam=cam,imtype='coarse_comp',filenum=filenum)
    coarse_comp = read_hdu(fileloc=filedir,filename=filename)
    coarse_comp_data = Table(coarse_comp.data)#[['r101','r401','r501','r516','r606','r707','r808']]

    fibernames = np.sort(coarse_comp_data.colnames)
    fib1s = fibernames[:int(len(fibernames)/2)+1]
    fib2s = fibernames[int(len(fibernames) / 2)-1:][::-1]
    coarse_comp_data_hist = None
    #coarse_comp_data_hist = Table.read("out_coefs_{}.fits".format(filenum_hist),format='fits')
    obs1 = {'coarse_comp' : coarse_comp_data[fib1s.tolist()], 'complinelistdict' : complinelistdict, 'print_itters' : False}
    obs2 = {'coarse_comp': coarse_comp_data[fib2s.tolist()], 'complinelistdict': complinelistdict, 'print_itters': False}
    all_obs = [obs1,obs2]
    if len(all_obs)<4:
        NPROC = len(all_obs)
    else:
        NPROC = 4

    with Pool(NPROC) as pool:
        tabs = pool.map(automated_calib_wrapper_script,all_obs)
        print(tabs)

    # out_tab1 = run_interactive_slider_calibration(coarse_comp_data[fib1s], complinelistdict, default_vals=None,history_vals=None,\
    #                                    steps = None, default_key = None,last_obs=coarse_comp_data_hist)
    # out_tab2 = run_interactive_slider_calibration(coarse_comp_data[fib2s], complinelistdict, default_vals=None,history_vals=None,\
    #                                    steps = None, default_key = None,last_obs=coarse_comp_data_hist)
    from astropy.table import hstack

    compare_outputs(coarse_comp_data, tabs[0], tabs[1])

    tabs[1] = tabs[1][fib2s[::-1].tolist()]
    tabs[0].remove_column(fibernames[int(len(fibernames) / 2)])
    tabs[1].remove_column(fibernames[int(len(fibernames) / 2) - 1])

    out_tab = hstack([tabs[0],tabs[1]])

    out_tab.write("out_coefs_{}{}.fits".format(cam,filenum),format='fits',overwrite=True)
    out_tab.write("out_coefs_{}{}.csv".format(cam,filenum), format='ascii.csv',overwrite=True)
