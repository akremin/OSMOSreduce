import os
from astropy.io import fits

import numpy as np
from astropy.table import Table
from interactive_plot import interactive_plot,pix_to_wave
from scipy.signal import argrelmax
import pickle as pkl
from wavelength_calibration import get_highestflux_waves,\
    top_peak_pixels,top_peak_wavelengths,update_default_dict



def run_interactive_slider_calibration(coarse_comp, complinelistdict, default_vals=None,history_vals=None,\
                                   steps = None, default_key = None, trust_initial = False):
    precision = 1e-2
    waves, fluxes = generate_synthetic_spectra(complinelistdict, compnames=['HgAr', 'NeAr'],precision=precision,maxheight=10000.)
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
    fibernames = coarse_comp.colnames
    for fiber_identifier in fibernames:#['r101','r408','r409','r608','r816']:
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

        import matplotlib.pyplot as plt
        from scipy.stats import pearsonr

        pix1 = pixels
        pix2 = pixels*pixels
        subset = np.arange(0, len(pixels), 2).astype(int)
        subset_comp = comp_spec[subset]
        subpix1 = pix1[subset]
        subpix2 = pix2[subset]

        abest, bbest, cbest, corrbest, pvalbest = 0., 0., 0., 0., 0.
        alow, ahigh = 3000, 8000
        prec_multiplier = int(1 / precision)

        if counter == 1:
            for b in np.arange(0.96,1.04,0.01):
                subpixb = b*subpix1
                subpixinds = (subpixb *prec_multiplier).astype(int)
                for a in np.arange(alow, ahigh, 1.):
                    indoffset = int((a-waves[0])*prec_multiplier)
                    synthwaveinds = subpixinds+indoffset
                    cut_subset_comp = subset_comp
                    if synthwaveinds[-40] < 0. or synthwaveinds[40] >= len(fluxes):
                        continue
                    elif synthwaveinds[0]<0. or synthwaveinds[-1]>=len(fluxes):
                        waverestrict_cut = np.argwhere(((synthwaveinds>=0) & (synthwaveinds<len(fluxes))))
                        synthwaveinds = synthwaveinds[waverestrict_cut]
                        cut_subset_comp = subset_comp[waverestrict_cut]
                    synthflux = fluxes[synthwaveinds]
                    corr,pval = pearsonr(synthflux,cut_subset_comp)
                    if corr>corrbest:
                        abest,bbest,cbest,corrbest,pvalbest = a,b,0.,corr,pval
            print("Itter 1 results:")
            print(corrbest)
            print(abest,bbest,cbest)
        else:
            last_fiber = fibernames[counter-2]
            [abest, bbest, cbest, trash1, trash2, trash3] = all_coefs[last_fiber]

            subpixbc = bbest*subpix1 + cbest*subpix2
            subpixinds = (subpixbc *prec_multiplier).astype(int)
            for a in np.arange(alow,ahigh, 1.):
                indoffset = int((a-waves[0])*prec_multiplier)
                synthwaveinds = subpixinds+indoffset
                cut_subset_comp = subset_comp
                if synthwaveinds[-40] < 0. or synthwaveinds[40] >= len(fluxes):
                    continue
                elif synthwaveinds[0] < 0. or synthwaveinds[-1] >= len(fluxes):
                    waverestrict_cut = np.argwhere(((synthwaveinds >= 0) & (synthwaveinds < len(fluxes))))[0]
                    synthwaveinds = synthwaveinds[waverestrict_cut]
                    cut_subset_comp = subset_comp[waverestrict_cut]
                synthflux = fluxes[synthwaveinds]
                corr, pval = pearsonr(synthflux, cut_subset_comp)
                if corr>corrbest:
                    abest,corrbest,pvalbest = a,corr,pval
            print("Itter 1 results:")
            print(corrbest)
            print("coef a: ",abest)
            print("Using past vals coefs b,c: ",bbest,cbest)

        aitterbest,bitterbest,citterbest = 0.,0.,0.
        corrbest = 0.

        for b in np.arange(bbest-0.02,bbest+0.02,1.0e-3):#for b in np.arange(0.96,1.04,0.0001):
            subpixb = b*subpix1
            for c in np.arange(cbest-(4.0e-6),cbest+(4.0e-6),2.0e-7):
                subpixbc = subpixb + (c * subpix2)
                #pixwaves = np.round(pixbc, int(np.log10(prec_multiplier)))
                subpixinds = (subpixbc *prec_multiplier).astype(int)
                for a in np.arange(abest - 20, abest + 20, 1.):
                    indoffset = int((a-waves[0])*prec_multiplier)
                    synthwaveinds = subpixinds+indoffset
                    cut_subset_comp = subset_comp
                    if synthwaveinds[-40] < 0. or synthwaveinds[40] >= len(fluxes):
                        continue
                    elif synthwaveinds[0]<0. or synthwaveinds[-1]>=len(fluxes):
                        waverestrict_cut = np.argwhere(((synthwaveinds>=0) & (synthwaveinds<len(fluxes))))[0]
                        synthwaveinds = synthwaveinds[waverestrict_cut]
                        cut_subset_comp = subset_comp[waverestrict_cut]
                    synthflux = fluxes[synthwaveinds]
                    corr,pval = pearsonr(synthflux,cut_subset_comp)
                    if corr>corrbest:
                        aitterbest, bitterbest, citterbest, corrbest, pvalbest = a, b, c, corr, pval
        abest, bbest, cbest = aitterbest, bitterbest, citterbest
        print("Itter 2 results:")
        print(corrbest)
        print("coefs a,b,c: ",abest,bbest,cbest)

        aitterbest, bitterbest, citterbest = 0., 0., 0.
        corrbest = 0.

        for b in np.arange(bbest-0.01,bbest+0.01,1.0e-4):#for b in np.arange(0.96,1.04,0.0001):
            pixb = b*pix1
            for c in np.arange(cbest-(2.0e-6),cbest+(2.0e-6),1.0e-7):
                pixbc = pixb + (c * pix2)
                #pixwaves = np.round(pixbc, int(np.log10(prec_multiplier)))
                pixinds = (pixbc *prec_multiplier).astype(int)
                for a in np.arange(abest - 4., abest + 4., 0.1):
                    indoffset = int((a-waves[0])*prec_multiplier)
                    synthwaveinds = pixinds+indoffset
                    cut_comp_specp = comp_spec
                    if synthwaveinds[-40] < 0. or synthwaveinds[40] >= len(fluxes):
                        continue
                    elif synthwaveinds[0]<0. or synthwaveinds[-1]>=len(fluxes):
                        waverestrict_cut = np.argwhere(((synthwaveinds>=0) & (synthwaveinds<len(fluxes))))[0]
                        synthwaveinds = synthwaveinds[waverestrict_cut]
                        cut_comp_specp = comp_spec[waverestrict_cut]
                    synthflux = fluxes[synthwaveinds]
                    corr,pval = pearsonr(synthflux,cut_comp_specp)
                    if corr>corrbest:
                        aitterbest,bitterbest,citterbest,corrbest,pvalbest = a,b,c,corr,pval
        abest, bbest, cbest = aitterbest,bitterbest,citterbest
        print("Itter 3 results:")
        print(corrbest)
        print("coefs a,b,c: ",abest,bbest,cbest)

        all_coefs[fiber_identifier] = [abest, bbest, cbest, 0., 0., 0.]
        all_flags[fiber_identifier] = corrbest

        plt.figure()
        plt.plot(waves,fluxes,'r-',label='synth')
        plt.plot(abest+(bbest*pix1)+(cbest*pix2),comp_spec,'b-',label='data')
        plt.legend(loc='best')
        if counter % 20 == 0:
            plt.show()

        ## Do an interactive second order fit to the spectra
        # if trust_initial and counter != 1:
        #     good_spec = True
        #     out_coef = {}
        #     out_coef['a'],out_coef['b'],out_coef['c'] = default_dict[default_key]
        #     print("\t\tYou trusted {} which gave: a={} b={} c={}".format(default_key,*default_dict[default_key]))
        # else:
        #     good_spec,out_coef = interactive_plot(pixels=pixels, spectra=comp_spec,\
        #                      linelistdict=complinelistdict, gal_identifier=fiber_identifier,\
        #                      default_dict=default_dict,steps=steps,default_key=default_key)
        #
        # ## If it's the first iteration, use the results to compute the largest
        # ## flux lines and their true wavelength values
        # ## these are used in all future iterations of this loop in the cross cor
        # if first_iteration and good_spec:
        #     top_peak_waves = top_peak_wavelengths(pixels, comp_spec, out_coef)
        #
        #     for peak in top_peak_waves:
        #         index = np.argmin(np.abs(wsorted_top_wave-peak))
        #         matched_peak_waves.append(wsorted_top_wave[index])
        #         matched_peak_flux.append(wsorted_top_flux[index])
        #         matched_peak_index.append(index)
        #
        #     matched_peak_waves = np.asarray(matched_peak_waves)
        #     matched_peak_flux = np.asarray(matched_peak_flux)
        #     matched_peak_index = np.asarray(matched_peak_index)
        #     print("Returned waves: {}\nMatched_waves:{}\n".format(top_peak_waves,matched_peak_waves))
        #
        # ## Save the flag
        # all_flags[fiber_identifier] = good_spec
        #
        # ## Save the coefficients if it's good
        # if good_spec:
        #     default_dict['predicted from prev spec'] = (out_coef['a'],out_coef['b'],out_coef['c'])
        #     all_coefs[fiber_identifier] = [out_coef['a'],out_coef['b'],out_coef['c'],0.,0.,0.]
        #     first_iteration = False
        # else:
        #     all_coefs[fiber_identifier] = [0.,0.,0.,0.,0.,0.]
        #
        # if counter == 999:
        #     counter = 0
        #     with open('_temp_wavecalib.pkl','wb') as temp_pkl:
        #         pkl.dump([all_coefs,all_flags],temp_pkl)
        #     print("Saving an incremental backup to _temp_wavecalib.pkl")
        #     cont = str(input("\n\n\tDo you want to continue? (y or n)\t\t"))
        #     if cont.lower() == 'n':
        #         break

    return Table(all_coefs)

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



if __name__ == '__main__':
    ## r_calibration_basic-HgAr-NeAr-Xe_11C_628_199652
    fittype='basic-HgAr-NeAr-Xe'
    cam='r'
    config='11C'
    filenum=628
    timestamp=199652
    cal_lamps = ['HgAr','NeAr','Xe']

    basedir = os.path.abspath('../../')
    calibration_template = '{cam}_calibration_{fittype}_{config}_{filenum}_{timestamp}.fits'
    path_to_mask = os.path.join(basedir,'OneDrive - umich.edu','Research','M2FSReductions','A02')
    complete_calib_name = os.path.join(path_to_mask,'calibrations',calibration_template)
    if 'basic' in fittype:
        filename = complete_calib_name.format(cam=cam, fittype=fittype, config=config, \
                                                    filenum=filenum, timestamp=timestamp)
        calib = Table.read(filename)
    elif 'full' in fittype:
        filename = complete_calib_name.format(cam=cam, fittype=fittype, config=config, \
                                                    filenum=filenum, timestamp=timestamp)
        calib = fits.open(filename)

    lampline_dir = os.path.join(os.path.abspath('.'),'lamp_linelists','salt')
    lampline_template = '{mod}{lamp}.csv'
    complinelistdict,allwms = load_calibration_lines_dict(cal_lamps,wavemincut=3000,wavemaxcut=8000)

    filedir =os.path.join(path_to_mask,'oneds')
    filename = '{cam}_{imtype}_{filenum}_A02_1d_bc.fits'.format(cam=cam,imtype='coarse_comp',filenum=filenum)
    coarse_comp = read_hdu(fileloc=filedir,filename=filename)
    coarse_comp_data = coarse_comp.data
    run_interactive_slider_calibration(coarse_comp_data, complinelistdict, default_vals=None,history_vals=None,\
                                       steps = None, default_key = None, trust_initial = False)
