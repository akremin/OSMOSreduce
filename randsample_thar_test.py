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
from linebrowser import LineBrowser


def wavelength_fitting_by_line_selection(comp, selectedlistdict, fulllinelist, coef_table, select_lines = False, bounds=None):
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

    hand_fit_subset = [cam+'101',cam+'816',cam+'416',cam+'501']
    for fib in hand_fit_subset:
        if fib not in comp.colnames:
            hand_fit_subset.remove(fib)
    hand_fit_subset = np.asarray(hand_fit_subset)
    extrema_fiber = False
    for fiber in hand_fit_subset:#''r401','r801']:hand_fit_subset
        if fiber[1:] in ['101','816','501','416','501']:
            extrema_fiber = True
        else:
            extrema_fiber = False
        counter += 1
        f_x = comp[fiber].data
        coefs = coef_table[fiber]

        if len(all_coefs.keys())>0:
            coef_devs = np.zeros(len(coefs)).astype(np.float64)
            for key,key_coefs in all_coefs.items():
                dev = np.asarray(key_coefs)-np.asarray(coef_table[key])
                coef_devs += dev
            coef_devs /= len(all_coefs.keys())

            updated_coefs = coefs+coef_devs
        else:
            updated_coefs = coefs
        iteration_wm,iteration_fm = wm.copy(),fm.copy()

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

            browser = LineBrowser(iteration_wm, iteration_fm, f_x, coefs, fulllinelist, bounds=bounds,edge_line_distance=10.0)
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
    def waves(pixels, a, b, c, d, e, f):
        return a + (b * pixels) + (c * pixels * pixels) +\
               d*np.power(pixels,3) + e*np.power(pixels,4) + f*np.power(pixels,5)
    fib1s = set(table1.colnames)
    fib2s = set(table2.colnames)
    matches = fib1s.intersection(fib2s)

    for match in matches:
        pixels = np.arange(len(raw_data[match])).astype(np.float64)
        a1,b1,c1,d1,e1,f1 = table1[match]
        a2, b2, c2, d2, e2, f2 = table2[match]
        waves1 = waves(pixels, a1,b1,c1,d1,e1,f1 )
        waves2 = waves(pixels, a2, b2, c2, d2, e2, f2 )
        dwaves = waves1-waves2
        print("\n"+match)
        print("--> Max deviation: {}  mean: {}  median: {}".format(dwaves[np.argmax(np.abs(dwaves))], np.mean(np.abs(dwaves)), np.median(np.abs(dwaves))))
        plt.figure()
        plt.plot(pixels, dwaves, 'r-')
        plt.show()

def wrapper_script(input_dict):
    return run_interactive_slider_calibration(**input_dict)

if __name__ == '__main__':
    ## r_calibration_basic-HgAr-NeAr-Xe_11C_628_199652
    #fittype='basic-ThAr'
    cam='b'
    config='11C'
    filenum=635#627
    filenum_hist=636#636
    cal_lamps = ['ThAr']#['HgAr','NeAr','Xe']

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
    filename = '{cam}_{imtype}_{filenum}_A02_1d_bc.fits'.format(cam=cam,imtype='fine_comp',filenum=filenum)
    fine_comp = read_hdu(fileloc=filedir,filename=filename)
    fine_comp_data = Table(fine_comp.data)#[['r101','r401','r501','r516','r606','r707','r808']]

    fibernames = np.sort(fine_comp_data.colnames)

    coarse_comp_data_hist = Table.read("out_coefs_{}{}.fits".format(cam,filenum_hist),format='fits')
    wm,fm = [],[]#complinelistdict['ThAr']
    for key, (keys_wm, keys_fm) in complinelistdict.items():
        if key in ['ThAr', 'Th']:
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
    fulllinelist = np.asarray(wm)
    # out_calib, out_linelist, lambdas, pixels, variances = {},{},{},{},{}
    out_calib, out_linelist, lambdas, pixels, variances\
                    = wavelength_fitting_by_line_selection(fine_comp_data, complinelistdict,
                                                            fulllinelist,
                                                            coarse_comp_data_hist,
                                                            select_lines=True)

    # out_calib['r101']= [4514.785015940739, 0.9860219272012049, 3.104736292866176e-05, -3.100749348729148e-08, 1.4558304305105122e-11, -2.7280872529766505e-15]
    # out_calib['r816']= [ 4523.774491764235, 0.985262106058085, 2.7390668927679338e-05, -2.1573543794296492e-08, 7.647742258718624e-12, -1.129551800554679e-15]
    # out_calib['r416']= [ 4414.016899581367, 0.9887065901878827, 3.161259335774533e-05, -2.81792839871385e-08, 1.182524163468614e-11, -2.043657935271939e-15]
    # out_calib['r501']= [ 4415.516224600041, 0.9911388945311701, 2.3266239070990097e-05, -1.7173625829699902e-08, 5.69606141928215e-12, -8.41153658928217e-16]
    # out_calib['r201']= [ 4465.130784106039, 0.9910568326878101, 2.0828799432576087e-05, -1.8730763585837518e-08, 8.57534627753987e-12, -1.714504627531229e-15]
    # out_calib['r307']= [ 4429.65691319388, 0.9924010213537284, 2.195969710503122e-05, -1.9325511228052312e-08, 8.270075194702245e-12, -1.54732871665399e-15]
    # out_calib['r603']= [ 4425.628437358703, 0.9932580008909366, 1.7480972116062678e-05, -1.1444677851391455e-08, 2.929644731491331e-12, -3.195059775040317e-16]
    # out_calib['r707']= [ 4459.000887261994, 0.9929247701347993, 1.4994889623686432e-05, -9.020477916922683e-09, 2.221349617669442e-12, -3.181665600807387e-16]
    #
    # hand_fit_names = np.asarray(list(out_calib.keys()))
    # numeric_hand_fit_names = np.asarray([ 16*int(fiber[1])+int(fiber[2:]) for fiber in hand_fit_names])
    #
    # save_lines = Table.read('selected_thar_lines.fits',format='fits')
    # wm = np.asarray(save_lines['wm'])
    # fm = np.asarray(save_lines['fm'])
    # from linebrowser import least_squares_fit
    # last_fiber = 'r101'
    #
    # for fiber in fine_comp_data.colnames:
    #     if fiber in hand_fit_names:
    #         continue
    #     coefs = np.asarray(coarse_comp_data_hist[fiber])
    #     f_x = fine_comp_data[fiber].data
    #
    #     fibern = 16*int(fiber[1])+int(fiber[2:])
    #
    #     nearest_fibs = hand_fit_names[np.argsort(np.abs(fibern-numeric_hand_fit_names))[:2]]
    #     diffs_fib1 = np.asarray(out_calib[nearest_fibs[0]]) - np.asarray(coarse_comp_data_hist[nearest_fibs[0]])
    #     diffs_fib2 = np.asarray(out_calib[nearest_fibs[1]]) - np.asarray(coarse_comp_data_hist[nearest_fibs[1]])
    #
    #     nearest_fib = np.asarray(out_calib[last_fiber]) - np.asarray(coarse_comp_data_hist[last_fiber])
    #
    #     diffs_mean = (0.25*diffs_fib1)+(0.25*diffs_fib2)+(0.5*nearest_fib)
    #
    #     adjusted_coefs_guess = coefs+diffs_mean
    #     browser = LineBrowser(wm,fm, f_x, adjusted_coefs_guess, fulllinelist, bounds=None)
    #
    #     params,covs = browser.fit()
    #
    #     plt.close()
    #     browser.create_saveplot(params, covs, 'fiberfits/{}'.format(fiber))
    #     print('\n\n',fiber,*params)
    #     out_calib[fiber] = params
    #     variances[fiber] = covs.diagonal()
    #     normd_vars = variances[fiber]/(params*params)
    #     print(np.sqrt(np.sum(normd_vars)))
    #     print(np.sqrt(normd_vars))
    #
    #     #savename = '{}'.format(fiber)
    #     #browser.create_saveplot(params,covs, savename)
    #
    #     pixels[fiber] = browser.line_matches['peaks_p']
    #     lambdas[fiber] = browser.line_matches['lines']
    #     del browser
    #     last_fiber = fiber

    #['fine_comp': fine_comp_data[fibs.tolist()], 'complinelistdict': complinelistdict, \
    #        'print_itters': True, 'last_obs': hist_data[fibs.tolist()]
    out_tab = Table(out_calib)
    out_tab.write("out_coefs_thar_{}{}.fits".format(cam,filenum),format='fits',overwrite=True)
    out_tab.write("out_coefs_thar_{}{}.csv".format(cam,filenum), format='ascii.csv',overwrite=True)
