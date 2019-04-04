
import pickle as pkl
import gc
from collections import OrderedDict
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack

from calibration_funcs import compare_outputs, \
    automated_calib_wrapper_script, interactive_plot, \
    get_highestflux_waves, update_default_dict, \
    top_peak_wavelengths,pix_to_wave_fifthorder,\
    ensure_match, find_devs

from linebrowser import LineBrowser


class Calibrations:
    def __init__(self, camera, instrument, lamptypesc, lamptypesf, coarse_calibrations, filemanager, config, \
                 fine_calibrations=None, pairings=None, load_history=True, trust_after_first=False,\
                 default_fit_key='cross correlation',use_selected_calib_lines=False, \
                 single_core=False, save_plots=False, savetemplate_funcs=None,show_plots=False):

        self.imtype = 'comp'

        self.camera = camera
        self.instrument = instrument
        self.filemanager = filemanager
        self.config = config
        self.lamptypesc = lamptypesc
        self.lamptypesf = lamptypesf
        self.default_fit_key = default_fit_key
        self.savetemplate_funcs = savetemplate_funcs

        self.trust_after_first = trust_after_first
        self.single_core = single_core
        self.save_plots = save_plots
        self.show_plots = show_plots

        #self.linelistc,selected_linesc,all_linesc = filemanager.load_calibration_lines_dict(lamptypesc,use_selected=use_selected_calib_lines)
        self.linelistc,all_linesc = filemanager.load_calibration_lines_dict(lamptypesc,use_selected=use_selected_calib_lines)

        self.load_history = load_history
        self.coarse_calibrations = coarse_calibrations

        self.ncalibs = len(coarse_calibrations.keys())
        self.do_fine_calib = (fine_calibrations is not None)

        if self.do_fine_calib:
            self.filenum_ind = 1
        else:
            self.filenum_ind = 0

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


    def run_initial_calibrations(self,skip_coarse=False,use_history_calibs=False,only_use_peaks = True):
        if skip_coarse:
            match = 0
            for pairnum in self.pairings.keys():
                if self.default_calibration_coefs is not None:
                    self.coarse_calibration_coefs[pairnum] = Table(self.default_calibration_coefs)
                    match += 1
                elif self.history_calibration_coefs[pairnum] is not None:
                    self.coarse_calibration_coefs[pairnum] = self.history_calibration_coefs[pairnum].copy()
                    match += 1
                else:
                    self.coarse_calibration_coefs[pairnum] = None
            if match > 0:
                return

        if use_history_calibs:
            if self.history_calibration_coefs[0] is not None:
                histories = self.get_medianfits_of(self.history_calibration_coefs)
            else:
                histories = None

        for pairnum,(cc_filnum, throwaway) in self.pairings.items():

            comp_data = Table(self.coarse_calibrations[cc_filnum].data)

            if self.single_core:
                obs1 = {
                    'coarse_comp': comp_data, 'complinelistdict': self.linelistc,
                    'print_itters': False, 'last_obs': histories,'only_use_peaks': only_use_peaks
                }

                out_calib = automated_calib_wrapper_script(obs1)
            else:
                fib1s = np.append(self.instrument.lower_half_fibs[self.camera],self.instrument.overlapping_fibs[self.camera][1])
                fib2s = np.append(self.instrument.upper_half_fibs[self.camera],self.instrument.overlapping_fibs[self.camera][0])
                #histories = None#self.history_calibration_coefs[pairnum]
                if histories is not None:
                    hist1 = histories[fib1s.tolist()]
                    hist2 = histories[fib2s.tolist()]
                else:
                    hist1,hist2 = None, None
                coarse_comp_data_hist = None
                # coarse_comp_data_hist = Table.read("out_coefs_{}.fits".format(filenum_hist),format='fits')
                obs1 = {'coarse_comp': comp_data[fib1s.tolist()], 'complinelistdict': self.linelistc,
                    'print_itters': False,'last_obs': hist1,'only_use_peaks': only_use_peaks}
                obs2 = {'coarse_comp': comp_data[fib2s.tolist()], 'complinelistdict': self.linelistc,
                    'print_itters': False,'last_obs':hist2,'only_use_peaks': only_use_peaks}

                all_obs = [obs1, obs2]
                if len(all_obs) < 4:
                    NPROC = len(all_obs)
                else:
                    NPROC = 4

                with Pool(NPROC) as pool:
                    tabs = pool.map(automated_calib_wrapper_script, all_obs)

                overlaps = self.instrument.overlapping_fibs[self.camera]
                if pairnum == 0:
                    template = self.savetemplate_funcs(cam=str(cc_filnum) + '_', ap='{fiber}', imtype='coarse_calib',
                                                       step='calib_directionfit_comparison', comment='auto')
                    matches = compare_outputs(comp_data, tabs[0], tabs[1],save_template=template,\
                                              save_plots=self.save_plots,show_plots=self.show_plots)
                    matches = list(matches)

                    if np.any(np.sort(overlaps) != np.sort(matches)):
                        print("The overlaps returned from the matching didn't match the overlaps")
                        print("that were explicitly defined: {}  {}".format(matches, overlaps))
                else:
                    lastcc,lastfc = self.pairings[pairnum-1]
                    template = self.savetemplate_funcs(cam=str(cc_filnum)-str(lastcc) + '_', ap='{fiber}',
                                                       imtype='coarse_calib',
                                                       step='calib_time_comparison', comment='auto')
                    matches = compare_outputs(comp_data, tabs[0][overlaps],\
                                              self.coarse_calibration_coefs[pairnum-1][overlaps],\
                                              save_template=template, \
                                              save_plots=self.save_plots, show_plots=self.show_plots)

                tabs[0].remove_column(overlaps[1])
                tabs[1].remove_column(overlaps[0])

                out_calib = hstack([tabs[0], tabs[1]])
                out_calib = out_calib[self.instrument.full_fibs[self.camera].tolist()]

            self.coarse_calibration_coefs[pairnum] = out_calib.copy()

            self.filemanager.save_basic_calib_dict(out_calib, self.lampstr_c, self.camera, self.config, filenum=cc_filnum)

            histories = out_calib

    def run_final_calibrations(self):
        if not self.do_fine_calib:
            print("There doesn't seem to be a fine calibration defined. Using the supplied coarse calibs")
        select_lines = True

        dev_allowance = 1.
        devs = 2.

        if self.default_calibration_coefs is None:
            initial_coef_table = Table(self.get_medianfits_of(self.coarse_calibration_coefs))
        else:
            initial_coef_table = Table(self.default_calibration_coefs)

        for pairnum,filnums in self.pairings.items():
            filenum = filnums[self.filenum_ind]

            data = Table(self.fine_calibrations[filenum].data)

            linelist = self.selected_lines

            if pairnum == 0:
                user_input = 'some'
            elif pairnum == 1:
                user_input = 'minimal'
            elif pairnum > 1 and devs < dev_allowance:
                user_input = 'none'

            hand_fit_subset = []
            cam = self.camera
            if user_input == 'some':
                if cam == 'r':
                    specific_set = [cam + '101', cam + '816', cam + '416', cam + '501']
                else:
                    specific_set = [cam + '116', cam + '801', cam + '516', cam + '401']
                for i, fib in enumerate(specific_set):
                    outfib = ensure_match(fib, data.colnames, hand_fit_subset, cam)
                    hand_fit_subset.append(outfib)
                seed = int(filenum)
                np.random.seed(seed)
                randfibs = ['{:02d}'.format(x) for x in np.random.randint(1, 16, 4)]
                for tetn, fibn in zip([2, 3, 6, 7], randfibs):
                    fib = '{}{}{}'.format(cam, tetn, fibn)
                    outfib = ensure_match(fib, data.colnames, hand_fit_subset, cam)
                    hand_fit_subset.append(outfib)
            elif user_input == 'minimal':
                if cam == 'r':
                    specific_set = [cam + '101', cam + '816', cam + '416']
                else:
                    specific_set = [cam + '116', cam + '801', cam + '516']
                for i, fib in enumerate(specific_set):
                    outfib = ensure_match(fib, data.colnames, hand_fit_subset, cam)
                    hand_fit_subset.append(outfib)
            elif user_input == 'all':
                hand_fit_subset = list(initial_coef_table.colnames)
            else:
                pass

            hand_fit_subset = np.asarray(hand_fit_subset)
            out_calib_h, out_linelist_h, lambdas_h, pixels_h, variances_h, wm, fm  = \
                                    self.wavelength_fitting_by_line_selection(data, linelist, \
                                    self.all_lines, initial_coef_table,select_lines=select_lines,\
                                    filenum=filenum,subset=hand_fit_subset)#bounds=None)

            if self.single_core:
                out_calib, outlinelist, lambdas, pixels, variances = \
                    auto_wavelength_fitting_by_lines(data, self.all_lines, initial_coef_table, wm, fm,\
                                                          out_calib_h, user_input=user_input,filenum=filenum, \
                                                          save_plots=self.save_plots, savetemplate_funcs=self.savetemplate_funcs)
                for key in out_calib_h.keys():
                    lambdas[key] = lambdas_h[key]
                    pixels[key] = pixels_h[key]
                    variances[key] = variances_h[key]
            else:
                fib1s = np.append(self.instrument.lower_half_fibs[self.camera],
                                  self.instrument.overlapping_fibs[self.camera][1])
                fib2s = np.append(self.instrument.upper_half_fibs[self.camera],
                                  self.instrument.overlapping_fibs[self.camera][0])
                obs1 = {
                    'comp': data[fib1s.tolist()], 'fulllinelist': self.all_lines.copy(),
                    'coef_table': initial_coef_table, 'wm':wm,'fm':fm,\
                    'all_coefs':out_calib_h,'user_input': user_input, 'filenum':filenum,
                    'save_plots':self.save_plots, "savetemplate_funcs":self.savetemplate_funcs
                }
                obs2 = {
                    'comp': data[fib2s.tolist()], 'fulllinelist': self.all_lines.copy(),
                    'coef_table': initial_coef_table.copy(), 'wm':wm.copy(),'fm':fm.copy(),\
                    'all_coefs':out_calib_h.copy(),'user_input': user_input, 'filenum':filenum,
                    'save_plots': self.save_plots, "savetemplate_funcs": self.savetemplate_funcs
                }

                all_obs = [obs1, obs2]
                if len(all_obs) < 4:
                    NPROC = len(all_obs)
                else:
                    NPROC = 4

                with Pool(NPROC) as pool:
                    tabs = pool.map(auto_wavelength_fitting_by_lines_wrapper, all_obs)

                out_calib, out_linelist,lambdas, pixels, variances = tabs[0]
                out_calib_a2, out_linelist_a2,lambdas_a2, pixels_a2, variances_a2 = tabs[1]

                for key in out_calib_h.keys():
                    out_calib_a2.pop(key)
                    lambdas[key] = lambdas_h[key]
                    pixels[key] = pixels_h[key]
                    variances[key] = variances_h[key]
                    out_linelist[key] = out_linelist_h[key]

                template = self.savetemplate_funcs(cam=str(filenum) + '_', ap='{fiber}', imtype='fine_calib', step='wavecalib',
                                                   comment='fit_comparison')

                matches = compare_outputs(data, Table(out_calib), Table(out_calib_a2),save_template=template,\
                                          save_plots=self.save_plots,show_plots=self.show_plots)

                if np.sort(self.instrument.overlapping_fibs[self.camera])!=np.sort(matches):
                    print("The overlaps returned from the matching didn't match the overlaps")
                    print("that were explicitly defined: {}  {}".format(matches,self.instrument.overlapping_fibs[self.camera]))

                for key in out_calib_a2.keys():
                    if key in matches:
                        continue
                    out_calib[key] = out_calib_a2[key]
                    lambdas[key] = lambdas_a2[key]
                    pixels[key] = pixels_a2[key]
                    variances[key] = variances_a2[key]
                    out_linelist[key] = out_linelist_a2[key]

            if select_lines:
                self.selected_lines = out_linelist

            hand_fit_subset = []
            for key in variances.keys():
                true_waves = np.array(lambdas[key])
                test_pix = np.array(pixels[key])
                a,b,c,d,e,f = out_calib[key]
                test_waves = waves(test_pix,a,b,c,d,e,f)
                if np.std(test_waves-true_waves)> 0.6:
                    hand_fit_subset.append(key)

            hand_fit_subset = np.array(hand_fit_subset)
            out_calib_h, out_linelist_h, lambdas_h, pixels_h, variances_h, wm, fm = \
                self.wavelength_fitting_by_line_selection(data, self.selected_lines, \
                                                          self.all_lines, out_calib, select_lines=False, \
                                                          filenum=filenum, subset=hand_fit_subset)


            for key in out_calib_h.keys():
                out_calib[key] = out_calib_h[key]
                lambdas[key] = lambdas_h[key]
                pixels[key] = pixels_h[key]
                variances[key] = variances_h[key]

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

            out_calib_table =  Table()
            variances_tab = Table()
            lambdas_tab = Table()
            pixels_tab = Table()

            for key in self.instrument.full_fibs[self.camera]:
                out_calib_table.add_column(Table.Column(data=out_calib[key],name=key))
                variances_tab.add_column(Table.Column(data=variances[key],name=key))
                lambdas_tab.add_column(Table.Column(data=lambdas[key],name=key))
                pixels_tab.add_column(Table.Column(data=pixels[key],name=key))

            ## Create hdulist to export
            prim = fits.PrimaryHDU(header=self.fine_calibrations[filenum].header)
            calibs = fits.BinTableHDU(data=out_calib_table,name='calib coefs')
            varis = fits.BinTableHDU(data=variances_tab,name='fit variances')
            lambs = fits.BinTableHDU(data=lambdas_tab,name='wavelengths')
            pix = fits.BinTableHDU(data=pixels_tab,name='pixels')
            hdulist = fits.HDUList([prim, calibs, lambs, pix, varis])

            #out_calib_table = out_calib_table[np.sort(out_calib_table.colnames)]
            self.fine_calibration_coefs[pairnum] = out_calib_table.copy()

            if pairnum > 0:
                devs = find_devs(initial_coef_table,out_calib_table)

            initial_coef_table =  out_calib_table.copy()

            self.final_calibrated_hdulists[pairnum] = hdulist
            self.filemanager.save_full_calib_dict(hdulist, self.lampstr_f, self.camera, self.config, filenum=filenum)

            gc.collect()

    def get_medianfits_of(self,ordered_dict):
        coarse_tables = [Table(coarse) for coarse in ordered_dict.values() if coarse is not None]
        initial_coef_table = OrderedDict()
        for fib in coarse_tables[0].colnames:
            coeff_arr = np.zeros((len(coarse_tables,6)))
            for ii,tab in enumerate(coarse_tables):
                coeff_arr[ii,:] = tab[fib]
            coeff_med = np.median(coeff_arr,axis=0)
            initial_coef_table[fib] = coeff_med
        return initial_coef_table

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


    def wavelength_fitting_by_line_selection(self, comp, selectedlistdict, fulllinelist, coef_table, select_lines = False, \
                                             bounds=None,filenum='',subset=[]):
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

        iteration_wm, iteration_fm = [],[]
        if cam == 'r':
            extremes = ['101','816','416','501']
        else:
            extremes =  ['116', '801', '516','401']

        extrema_fiber = False
        hand_fit_subset = np.array(subset)
        for fiber in hand_fit_subset:
            if fiber[1:] in extremes:
                extrema_fiber = True
            else:
                extrema_fiber = False
            counter += 1
            f_x = comp[fiber].data
            coefs = coef_table[fiber]
            if select_lines:
                iteration_wm, iteration_fm = wm.copy(), fm.copy()
            else:
                iteration_wm, iteration_fm = selectedlistdict[fiber]

            if len(all_coefs)>0:
                curtet = int(fiber[1])
                tets,fibs = [],[]
                for fib in all_coefs.keys():
                    tets.append(int(fib[1]))
                    fibs.append(fib)
                tets,fibs = np.array(tets),np.array(fibs)
                if 9-curtet in tets:
                    key = fibs[np.where((9-curtet)==tets)[0][0]]
                    coef_dev_med = np.asarray(all_coefs[key])-np.asarray(coef_table[key])
                else:
                    coef_devs = np.zeros(shape=(len(all_coefs),6)).astype(np.float64)
                    for ii,(key,key_coefs) in enumerate(all_coefs.items()):
                        dev = np.asarray(key_coefs)-np.asarray(coef_table[key])
                        coef_devs[ii,:] = dev
                    coef_dev_med = np.median(coef_devs,axis=0)

                updated_coefs = coefs+coef_dev_med
            else:
                updated_coefs = coefs

            browser = LineBrowser(iteration_wm,iteration_fm, f_x, updated_coefs, fulllinelist, bounds=bounds, \
                                  edge_line_distance=10.0,fibname=fiber)
            if np.any((np.asarray(browser.line_matches['lines'])-np.asarray(browser.line_matches['peaks_w']))>0.3):
                browser.plot()
            params,covs = browser.fit()

            print(fiber,*params)
            all_coefs[fiber] = params
            variances[fiber] = covs.diagonal()
            print(np.sum(variances[fiber]))

            template =  self.savetemplate_funcs(cam=str(filenum)+'_',ap=fiber,imtype='calib',step='finalfit',comment='byhand')
            if self.save_plots:
                browser.create_saveplot(params,covs, template)

            app_fit_pix[fiber] = browser.line_matches['peaks_p']
            app_fit_lambs[fiber] = browser.line_matches['lines']
            if select_lines:
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

        if len(hand_fit_subset)>0:
            cont = str(input("\n\n\tDo you need to repeat any? (y or n)")).strip(' \t\r\n')
            if cont.lower() == 'y':
                cam = comp.colnames[0][0]
                for itter in range(1000):
                    fiber = str(input("\n\tName the fiber")).strip(' \t\r\n')
                    print("Received: '{}'".format(fiber))

                    if fiber.strip(' \t\r\n') == '' or fiber is None:
                        break
                    else:
                        if cam not in fiber:
                            fiber = cam + fiber
                        f_x = comp[fiber].data
                        coefs = coef_table[fiber]

                        if len(all_coefs) > 0:
                            curtet = int(fiber[1])
                            tets, fibs = [], []
                            for fib in all_coefs.keys():
                                tets.append(int(fib[1]))
                                fibs.append(fib)
                            tets, fibs = np.array(tets), np.array(fibs)
                            if 9 - curtet in tets:
                                key = fibs[np.where((9 - curtet) == tets)[0][0]]
                                coef_dev_med = np.asarray(all_coefs[key]) - np.asarray(coef_table[key])
                            else:
                                coef_devs = np.zeros(shape=(len(all_coefs), 6)).astype(np.float64)
                                for ii, (key, key_coefs) in enumerate(all_coefs.items()):
                                    dev = np.asarray(key_coefs) - np.asarray(coef_table[key])
                                    coef_devs[ii, :] = dev
                                coef_dev_med = np.median(coef_devs, axis=0)

                            updated_coefs = coefs + coef_dev_med
                        else:
                            updated_coefs = coefs

                        if select_lines:
                            iteration_wm, iteration_fm = wm.copy(), fm.copy()
                        else:
                            iteration_wm, iteration_fm = selectedlistdict[fiber]

                        browser = LineBrowser(iteration_wm, iteration_fm, f_x, updated_coefs, fulllinelist, bounds=bounds,edge_line_distance=-20.0,fibname=fiber)
                        browser.plot()
                        params, covs = browser.fit()

                        print(fiber, *params)
                        all_coefs[fiber] = params
                        variances[fiber] = covs.diagonal()
                        print(np.dot(variances[fiber], variances[fiber]))

                        if select_lines:
                            app_specific_linelists[fiber] = (browser.wm, browser.fm)

                        template = self.savetemplate_funcs(cam=str(filenum)+'_', ap=fiber, imtype='calib', step='finalfit',
                                                           comment='byhand')
                        if self.save_plots:
                            browser.create_saveplot(params, covs, template)

                        plt.close()
                        del browser

        if not select_lines:
            app_specific_linelists = None
            wm,fm = iteration_wm, iteration_fm

        return all_coefs, app_specific_linelists, app_fit_lambs, app_fit_pix, variances, wm, fm




def auto_wavelength_fitting_by_lines_wrapper(input_dict):
    return auto_wavelength_fitting_by_lines(**input_dict)


def auto_wavelength_fitting_by_lines(comp, fulllinelist, coef_table, wm,fm,all_coefs,user_input='some',filenum='',\
                                     bounds=None, save_plots = True,  savetemplate_funcs='{}{}{}{}{}'.format):
    comp = Table(comp)

    variances = {}
    app_fit_pix = {}
    app_fit_lambs = {}
    outlinelist = {}
    hand_fit_subset = np.array(list(all_coefs.keys()))

    cam = comp.colnames[0][0]
    if user_input != 'none':
        if cam =='b':
            numeric_hand_fit_names = np.asarray([ (16*(9-int(fiber[1])))+int(fiber[2:]) for fiber in hand_fit_subset])
        else:
            numeric_hand_fit_names = np.asarray([ (16*int(fiber[1]))+int(fiber[2:]) for fiber in hand_fit_subset])

    coef_table = Table(coef_table)

    if cam =='b':
        numerics = np.asarray([(16 * (9 - int(fiber[1]))) + int(fiber[2:]) for fiber in comp.colnames])
    else:
        numerics = np.asarray([(16 * int(fiber[1])) + int(fiber[2:]) for fiber in comp.colnames])

    sorted = np.argsort(numerics)
    all_fibers = np.array(comp.colnames)[sorted]
    del sorted,numerics

    if cam == 'r' and int(all_fibers[0][1]) > 3:
        all_fibers = all_fibers[::-1]
    elif cam =='b' and int(all_fibers[0][1]) < 6:
        all_fibers = all_fibers[::-1]

    last_fiber = None

    for fiber in all_fibers:
        if fiber in hand_fit_subset:
            continue
        if fiber not in coef_table.colnames:
            continue
        coefs = np.asarray(coef_table[fiber])
        f_x = comp[fiber].data

        if user_input != 'none' and len(hand_fit_subset)>0:
            if cam == 'b':
                fibern = (16 * (9-int(fiber[1]))) + int(fiber[2:])
            else:
                fibern = (16 * int(fiber[1])) + int(fiber[2:])

            if len(hand_fit_subset)>1:
                dists = np.abs(fibern-numeric_hand_fit_names)
                closest = np.argsort(dists)[:2]
                nearest_fibs = hand_fit_subset[closest]
                diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                diffs_fib2 = np.asarray(all_coefs[nearest_fibs[1]]) - np.asarray(coef_table[nearest_fibs[1]])
                d1,d2 = dists[closest[0]], dists[closest[1]]
                diffs_hfib = (d2*diffs_fib1 + d1*diffs_fib2)/(d1+d2)
            else:
                nearest_fibs = hand_fit_subset[np.argsort(np.abs(fibern-numeric_hand_fit_names))]
                diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                diffs_hfib = diffs_fib1

            if last_fiber is None:
                last_fiber = nearest_fibs[0]

            nearest_fib = np.asarray(all_coefs[last_fiber]) - np.asarray(coef_table[last_fiber])

            diffs_mean = (0.5*diffs_hfib)+(0.5*nearest_fib)

            adjusted_coefs_guess = coefs+diffs_mean
        else:
            adjusted_coefs_guess = coefs

        browser = LineBrowser(wm,fm, f_x, adjusted_coefs_guess, fulllinelist, bounds=None, edge_line_distance=(-20.0),initiate=False)

        params,covs = browser.fit()

        if save_plots:
            template = savetemplate_funcs(cam=str(filenum)+'_', ap=fiber, imtype='calib', step='finalfit', comment='auto')
            browser.initiate_browser()
            browser.create_saveplot(params, covs, template)

        all_coefs[fiber] = params
        variances[fiber] = covs.diagonal()
        outlinelist[fiber] = (wm,fm)
        normd_vars = variances[fiber]/(params*params)

        app_fit_pix[fiber] = browser.line_matches['peaks_p']
        app_fit_lambs[fiber] = browser.line_matches['lines']

        print('\n\n',fiber,'{:.2f} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}'.format(*params))
        fitlamb = pix_to_wave_fifthorder(np.asarray(app_fit_pix[fiber]), params)
        dlamb = fitlamb - app_fit_lambs[fiber]
        print("  ----> mean={}, median={}, std={}, root normd fit var={}".format(np.mean(dlamb),np.median(dlamb),np.std(dlamb),np.sqrt(np.sum(normd_vars))))

        plt.close()
        del browser
        if np.std(dlamb) < 0.6:
            last_fiber = fiber

    return all_coefs, outlinelist, app_fit_lambs, app_fit_pix, variances



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


def waves(pixels, a, b, c,d,e,f):
    return a + (b * pixels) + (c * pixels * pixels)+\
           (d * np.power(pixels,3)) + (e * np.power(pixels,4))+ \
           (f * np.power(pixels, 5))