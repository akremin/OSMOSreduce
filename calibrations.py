
import pickle as pkl
from collections import OrderedDict
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack

from calibration_funcs import compare_outputs, automated_calib_wrapper_script, interactive_plot, \
    fit_using_crosscorr, get_highestflux_waves, update_default_dict, \
    generate_synthetic_spectra, top_peak_wavelengths,pix_to_wave_fifthorder,\
    ensure_match, find_devs

from linebrowser import LineBrowser


class Calibrations:
    def __init__(self, camera, lamptypesc, lamptypesf, coarse_calibrations, filemanager, config, \
                 fine_calibrations=None, pairings=None, load_history=True, trust_after_first=False,\
                 default_fit_key='cross correlation',use_selected_calib_lines=False):

        self.imtype = 'comp'

        self.camera = camera
        self.filemanager = filemanager
        self.config = config
        self.lamptypesc = lamptypesc
        self.lamptypesf = lamptypesf
        self.trust_after_first = trust_after_first
        self.default_fit_key = default_fit_key

        #self.linelistc,selected_linesc,all_linesc = filemanager.load_calibration_lines_dict(lamptypesc,use_selected=use_selected_calib_lines)
        self.linelistc,all_linesc = filemanager.load_calibration_lines_dict(lamptypesc,use_selected=use_selected_calib_lines)

        self.load_history = load_history
        self.coarse_calibrations = coarse_calibrations

        self.ncalibs = len(coarse_calibrations.keys())
        self.do_fine_calib = (fine_calibrations is not None)

        self.lampstr_c = 'basic'
        self.lampstr_f = 'full'

        for lamp in lamptypesc:
            self.lampstr_c += '-'+str(lamp)

        if self.do_fine_calib:
            self.linelistf,self.all_lines = filemanager.load_calibration_lines_dict(lamptypesf,use_selected=use_selected_calib_lines)
            #self.linelistf,self.selected_lines,self.all_lines = filemanager.load_calibration_lines_dict(lamptypesf,use_selected=use_selected_calib_lines)
            self.fine_calibrations = fine_calibrations
            for lamp in lamptypesf:
                self.lampstr_f += '-' + str(lamp)
        else:
            self.linelistf = self.linelistc.copy()
            #self.selected_lines = selected_linesc
            self.all_lines = all_linesc
            self.lampstr_f = self.lamstr_c.replace('basic','full')
            self.fine_calibrations = coarse_calibrations

        self.selected_lines = self.linelistf.copy()
        self.calibc_pairlookup = {}
        self.calibf_pairlookup = {}
        if pairings is None:
            self.pairings = OrderedDict()
            for ii, cc_filnum, cf_filnum in enumerate(zip(self.calibc_filenums,self.calibf_filenums)):
                self.pairings[ii] = (cc_filnum, cf_filnum)
                self.calibc_pairlookup[cc_filnum] = ii
                self.calibf_pairlookup[cf_filnum] = ii
        else:
            self.pairings = pairings
            for pairnum,(cc_filnum,cf_filnum) in pairings.items():
                self.calibc_pairlookup[cc_filnum] = pairnum
                self.calibf_pairlookup[cf_filnum] = pairnum

        self.pairnums = np.sort(list(self.pairings.keys()))

        self.history_calibration_coefs = {ii:None for ii in self.pairings.keys()}
        self.default_calibration_coefs = None
        self.coarse_calibration_coefs = OrderedDict()
        self.fine_calibration_coefs = OrderedDict()
        self.final_calibrated_hdulists = OrderedDict()
        self.evolution_in_coarse_coefs = OrderedDict()

        self.load_default_coefs()
        if load_history:
            self.load_most_recent_coefs()

    def load_default_coefs(self):
        from calibration_funcs import aperature_number_pixoffset
        self.default_calibration_coefs = self.filemanager.load_calib_dict('default', self.camera, self.config)
        if self.default_calibration_coefs is None:
            outdict = {}
            fibernames = Table(self.coarse_calibrations[self.pairings[0][0]].data).colnames
            adef, bdef, cdef, ddef, edef, fdef = (4465.4, 0.9896, 1.932e-05, 0., 0., 0.)
            for fibname in fibernames:
                aoff = aperature_number_pixoffset(fibname,self.camera)
                outdict[fibname] = (adef+aoff, bdef, cdef, ddef, edef, fdef)
            self.default_calibration_coefs = outdict

    def load_most_recent_coefs(self):
        couldntfind = False
        if self.do_fine_calib:
            for pairnum, (cc_filnum, cf_filnum) in self.pairings.items():
                name = self.lampstr_f
                calib,thetype = self.filemanager.locate_calib_dict(name, self.camera, self.config,cf_filnum)
                if thetype == 'full':
                    calib_tab = Table(calib['calib coefs'].data)
                else:
                    calib_tab = calib
                if calib_tab is None:
                    couldntfind = True
                    break
                else:
                    self.history_calibration_coefs[pairnum] = calib_tab
        if couldntfind or not self.do_fine_calib:
            for pairnum, (cc_filnum, cf_filnum) in self.pairings.items():
                name = self.lampstr_c
                calib,thetype = self.filemanager.locate_calib_dict(name, self.camera, self.config,cc_filnum)
                if thetype == 'full':
                    calib_tab = Table(calib['calib coefs'].data)
                else:
                    calib_tab = calib
                self.history_calibration_coefs[pairnum] = calib_tab


    def run_initial_calibrations(self,skip_coarse=False,single_core=False):
        for pairnum,(cc_filnum, throwaway) in self.pairings.items():
            if skip_coarse and self.history_calibration_coefs[pairnum] is not None:
                self.coarse_calibration_coefs[pairnum] = self.history_calibration_coefs[pairnum].copy()
            else:
                comp_data = Table(self.coarse_calibrations[cc_filnum].data)
                fibernames = np.sort(comp_data.colnames)

                if single_core:
                    histories = self.history_calibration_coefs[pairnum]
                    obs1 = {
                        'coarse_comp': comp_data, 'complinelistdict': self.linelistc,
                        'print_itters': True, 'last_obs': histories
                    }

                    out_calib = automated_calib_wrapper_script(obs1)
                else:
                    fib1s = fibernames[:int(len(fibernames) / 2) + 1]
                    fib2s = fibernames[int(len(fibernames) / 2) - 1:][::-1]
                    histories = self.history_calibration_coefs[pairnum]
                    if histories is not None:
                        hist1 = histories[fib1s.tolist()]
                        hist2 = histories[fib2s.tolist()]
                    else:
                        hist1,hist2 = None, None
                    coarse_comp_data_hist = None
                    # coarse_comp_data_hist = Table.read("out_coefs_{}.fits".format(filenum_hist),format='fits')
                    obs1 = {'coarse_comp': comp_data[fib1s.tolist()], 'complinelistdict': self.linelistc,
                        'print_itters': False,'last_obs': hist1}
                    obs2 = {'coarse_comp': comp_data[fib2s.tolist()], 'complinelistdict': self.linelistc,
                        'print_itters': False,'last_obs':hist2}

                    all_obs = [obs1, obs2]
                    if len(all_obs) < 4:
                        NPROC = len(all_obs)
                    else:
                        NPROC = 4

                    with Pool(NPROC) as pool:
                        tabs = pool.map(automated_calib_wrapper_script, all_obs)
                    print(tabs)

                    compare_outputs(comp_data, tabs[0], tabs[1])

                    tabs[1] = tabs[1][fib2s[::-1].tolist()]
                    tabs[0].remove_column(fibernames[int(len(fibernames) / 2)])
                    tabs[1].remove_column(fibernames[int(len(fibernames) / 2) - 1])

                    out_calib = hstack([tabs[0], tabs[1]])


                self.coarse_calibration_coefs[pairnum] = out_calib.copy()

                self.filemanager.save_basic_calib_dict(out_calib, self.lampstr_c, self.camera, self.config, filenum=cc_filnum)


    def run_final_calibrations(self):
        if not self.do_fine_calib:
            print("There doesn't seem to be a fine calibration defined. Using the supplied calibc's")
        select_lines = True
        if self.do_fine_calib:
            filenum_ind = 1
        else:
            filenum_ind = 0

        dev_allowance = 1.
        devs = 2.
        initial_coef_table = self.coarse_calibration_coefs[0]
        for pairnum,filnums in self.pairings.items():
            filenum = filnums[filenum_ind]

            ## Note that if there isn't a fine calibration, fine_calibrations
            ## has already been set equal to coarse_calibrations hdus
            data = Table(self.fine_calibrations[filenum].data)
            linelist = self.selected_lines

            if pairnum == 0:
                user_input = 'some'
            elif pairnum == 1:
                user_input = 'minimal'
            elif pairnum > 1 and devs < dev_allowance:
                user_input = 'none'

            out_calib, out_linelist, lambdas, pixels, variances  = self.wavelength_fitting_by_line_selection(data, linelist, self.all_lines, initial_coef_table,select_lines=select_lines,filenum=filenum,user_input=user_input)#bounds=None)

            if select_lines:
                self.selected_lines = out_linelist

            out_calib_table =  Table(out_calib)
            self.fine_calibration_coefs[pairnum] = out_calib_table.copy()

            if pairnum > 0:
                devs = find_devs(initial_coef_table,out_calib_table)

            initial_coef_table =  out_calib_table

            ## Create hdulist to export
            prim = fits.PrimaryHDU(header=self.fine_calibrations[filenum].header)
            out_calib = Table(out_calib)
            calibs = fits.BinTableHDU(data=out_calib,name='calib coefs')
            variances = Table(variances)
            varis = fits.BinTableHDU(data=variances,name='fit variances')

            ## Zero pad rows so that the table won't throw an error for unequal sizes
            maxlams = 0
            maxpix = 0
            for fib in lambdas.keys():
                nlams = len(lambdas[fib])
                npix = len(pixels[fib])
                if nlams>maxlams:
                    maxlams = nlams
                if npix > maxpix:
                    maxpix = npix
            for fib in lambdas.keys():
                lamarr = lambdas[fib]
                pixarr = pixels[fib]
                if len(lamarr)!=maxlams:
                    lambdas[fib] = np.append(lamarr,np.zeros(shape=maxlams-len(lamarr)))
                if len(pixarr)!=maxpix:
                    pixels[fib] = np.append(pixarr, np.zeros(shape=maxpix - len(pixarr)))

            lambdas = Table(lambdas)
            lambs = fits.BinTableHDU(data=lambdas,name='wavelengths')
            pixels = Table(pixels)
            pix = fits.BinTableHDU(data=pixels,name='pixels')

            hdulist = fits.HDUList([prim,calibs,lambs,pix,varis])
            self.final_calibrated_hdulists[pairnum] = hdulist
            self.filemanager.save_full_calib_dict(hdulist, self.lampstr_f, self.camera, self.config, filenum=filenum)

    def create_calibration_default(self,save=True):
        npairs = len(self.pairnums)
        default_outtable = self.fine_calibration_coefs[self.pairnums[0]]
        if npairs > 1:
            for pairnum in self.pairnums[1:]:
                curtable = self.fine_calibration_coefs[pairnum]
                for fiber in curtable.colnames:
                    default_outtable[fiber] += curtable[fiber]

            for fiber in curtable.colnames:
                default_outtable[fiber] /= npairs
        if save:
            self.filemanager.save_basic_calib_dict(default_outtable, 'default', self.camera, self.config)
        else:
            return default_outtable

    def save_initial_calibrations(self):
        for pairnum,table in self.coarse_calibration_coefs.items():
            filenum = self.pairings[pairnum][0]
            self.filemanager.save_basic_calib_dict(table, self.lampstr_c, self.camera, self.config, filenum=filenum)

    def save_final_calibrations(self):
        for pairnum,outlist in self.final_calibrated_hdulists.items():
            if self.do_fine_calib:
                filenum = self.pairings[pairnum][1]
            else:
                filenum = self.pairings[pairnum][0]
            self.filemanager.save_full_calib_dict(outlist, self.lampstr_f, self.camera, self.config, filenum=filenum)


    def wavelength_fitting_by_line_selection(self, comp, selectedlistdict, fulllinelist, coef_table, select_lines = False, bounds=None,filenum=None,user_input='some'):
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
        cam = comp.colnames[0][0]
        hand_fit_subset = []
        if user_input=='some':
            specific_set = [cam+'101',cam+'816',cam+'416',cam+'501']

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
        elif user_input == 'minimal':
            specific_set = [cam+'101',cam+'416',cam+'816']
            for i, fib in enumerate(specific_set):
                outfib = ensure_match(fib, comp.colnames, hand_fit_subset, cam)
                hand_fit_subset.append(outfib)
        elif user_input == 'all':
            hand_fit_subset = list(coef_table.colnames)
        else:
            pass

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

        if len(hand_fit_subset)>0:
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

                    plt.close()

                    savename = './fiberfits/{}'.format(fiber)
                    if filenum is not None:
                        savename = savename + '_' + str(filenum)
                    browser.create_saveplot(params, covs, savename)

                    del browser
                    fiber = input("\n\tName the fiber")

        # if len(hand_fit_subset) > 0 and select_lines:
        #     with open('_selected_lines.pkl','wb') as pklfil:
        #         pkl.dump((wm,fm,hand_fit_subset,all_coefs),pklfil)
        # else:
        #     with open('_selected_lines.pkl','rb') as pklfil:
        #         (wm,fm,hand_fit_subset,all_coefs) = pkl.load(pklfil)

        if user_input != 'none':
            numeric_hand_fit_names = np.asarray([ 16*int(fiber[1])+int(fiber[2:]) for fiber in hand_fit_subset])
            last_fiber = cam+'101'

        coef_table = Table(coef_table)
        all_fibers = np.sort(list(comp.colnames))

        for fiber in all_fibers:
            if fiber in hand_fit_subset:
                continue
            if fiber not in coef_table.colnames:
                continue
            coefs = np.asarray(coef_table[fiber])
            f_x = comp[fiber].data

            if user_input != 'none' and len(hand_fit_subset)>1:
                fibern = 16*int(fiber[1])+int(fiber[2:])

                nearest_fibs = hand_fit_subset[np.argsort(np.abs(fibern-numeric_hand_fit_names))[:2]]
                diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                diffs_fib2 = np.asarray(all_coefs[nearest_fibs[1]]) - np.asarray(coef_table[nearest_fibs[1]])

                nearest_fib = np.asarray(all_coefs[last_fiber]) - np.asarray(coef_table[last_fiber])

                diffs_mean = (0.25*diffs_fib1)+(0.25*diffs_fib2)+(0.5*nearest_fib)

                adjusted_coefs_guess = coefs+diffs_mean
            else:
                adjusted_coefs_guess = coefs

            browser = LineBrowser(wm,fm, f_x, adjusted_coefs_guess, fulllinelist, bounds=None, edge_line_distance=-20.0)

            params,covs = browser.fit()

            plt.close()

            savename = './fiberfits/{}'.format(fiber)
            if filenum is not None:
                savename = savename+'_'+str(filenum)
            browser.create_saveplot(params, covs, savename)
            print('\n\n',fiber,'{:.2f}{:.6e}{:.6e}{:.6e}{:.6e}{:.6e}'.format(*params))
            all_coefs[fiber] = params
            variances[fiber] = covs.diagonal()
            normd_vars = variances[fiber]/(params*params)

            print("".format(np.sqrt(normd_vars)))
            print("".format(np.sqrt(np.sum(normd_vars))))

            #savename = '{}'.format(fiber)
            #browser.create_saveplot(params,covs, savename)

            app_fit_pix[fiber] = browser.line_matches['peaks_p']
            app_fit_lambs[fiber] = browser.line_matches['lines']
            del browser
            last_fiber = fiber

            fitlamb = pix_to_wave_fifthorder(np.asarray(app_fit_pix[fiber]), params)
            dlamb = fitlamb - app_fit_lambs[fiber]
            print("mean={}, median={}, std={}".format(np.mean(dlamb),np.median(dlamb),np.std(dlamb)))

        if not select_lines:
            app_specific_linelists = None
        return all_coefs, app_specific_linelists, app_fit_lambs, app_fit_pix, variances


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


