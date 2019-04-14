
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
from collections import Counter

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
                    template = self.savetemplate_funcs(cam='{}-{}_'.format(cc_filnum,lastcc), ap='{fiber}',
                                                       imtype='coarse_calib',
                                                       step='calib_time_comparison', comment='auto')
                    matches = compare_outputs(comp_data, tabs[0][overlaps.tolist()],\
                                              self.coarse_calibration_coefs[pairnum-1][overlaps.tolist()],\
                                              save_template=template, \
                                              save_plots=self.save_plots, show_plots=self.show_plots)

                tabs[0].remove_column(overlaps[1])
                tabs[1].remove_column(overlaps[0])

                out_calib = hstack([tabs[0], tabs[1]])
                out_calib = out_calib[self.instrument.full_fibs[self.camera].tolist()]

            self.coarse_calibration_coefs[pairnum] = out_calib.copy()

            self.filemanager.save_basic_calib_dict(out_calib, self.lampstr_c, self.camera, self.config, filenum=cc_filnum)

            histories = out_calib

    def run_final_calibrations(self,initial_priors='median'):
        if not self.do_fine_calib:
            print("There doesn't seem to be a fine calibration defined. Using the supplied coarse calibs")
        select_lines = True

        dev_allowance = 1.
        devs = 2.
        using_defaults = False
        if initial_priors == 'defaults':
            if self.default_calibration_coefs is None:
                print("Couldn't find the default calibration coefficients, so using a parametrization of the coarse coefs")
                initial_coef_table = Table(self.get_parametricfits_of(self.coarse_calibration_coefs))
            else:
                initial_coef_table = Table(self.default_calibration_coefs)
                using_defaults = True
        elif initial_priors == 'medians':
            initial_coef_table = Table(self.get_medianfits_of(self.coarse_calibration_coefs))
        else:
            initial_coef_table = Table(self.get_parametricfits_of(self.coarse_calibration_coefs))

        for pairnum,filnums in self.pairings.items():

            # if pairnum == 0:
            #     from astropy.io import fits
            #     first = fits.open('/nfs/kremin/M2FS_analysis/data/B09/calibrations/r_calibration_full-ThAr_11J_1303_359770.fits')
            #     initial_coef_table = Table(first[1].data)
            #     self.fine_calibration_coefs[0] = initial_coef_table.copy()
            #     select_lines = False
            #     wm,fm = self.selected_lines['ThAr']
            #     select_linedict = {}
            #     waves = Table(first[2].data)
            #     for fib in waves.colnames:
            #         curwaves = waves[fib]
            #         itterwm,itterfm = [],[]
            #         for wave in curwaves:
            #             if wave == 0.:
            #                 continue
            #             loc = np.argmin(np.abs(wave-wm))
            #             itterwm.append(wm[loc])
            #             itterfm.append(fm[loc])
            #         select_linedict[fib] = (np.array(itterwm),np.array(itterfm))
            #     self.selected_lines = select_linedict
            #     continue
            # elif pairnum == 1:
            #     from astropy.io import fits
            #     second = fits.open('/nfs/kremin/M2FS_analysis/data/B09/calibrations/r_calibration_full-ThAr_11J_1309_359879.fits')
            #     initial_coef_table = Table(second[1].data)
            #     self.fine_calibration_coefs[1] = initial_coef_table.copy()
            #     devs = 0
            #     continue

            filenum = filnums[self.filenum_ind]

            data = Table(self.fine_calibrations[filenum].data)

            linelist = self.selected_lines

            effective_iteration = pairnum + int(using_defaults)
            if effective_iteration == 0:
                user_input = 'some'
            elif effective_iteration == 1:
                user_input = 'minimal'
            elif effective_iteration > 1:# and devs < dev_allowance:
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
                                    filenum=filenum,subset=hand_fit_subset,completed_coefs={})#bounds=None)

            if select_lines:
                linelistdict = {'ThAr': (wm, fm)}
            else:
                linelistdict = self.selected_lines

            if self.single_core:
                out_calib, outlinelist, lambdas, pixels, variances, badfits = \
                    auto_wavelength_fitting_by_lines(data, self.all_lines, initial_coef_table, linelistdict.copy(),\
                                                          out_calib_h, user_input=user_input,filenum=filenum, \
                                                          save_plots=self.save_plots, savetemplate_funcs=self.savetemplate_funcs)
                for key in lambdas_h.keys():
                    lambdas[key] = lambdas_h[key]
                    pixels[key] = pixels_h[key]
                    variances[key] = variances_h[key]
                    out_linelist[key] = out_linelist_h[key]
                badfits = np.array(badfits)
            else:
                fib1s = self.instrument.lower_half_fibs[self.camera]
                fib2s = self.instrument.upper_half_fibs[self.camera]

                obs1 = {
                    'comp': data[fib1s.tolist()], 'fulllinelist': self.all_lines.copy(),
                    'coef_table': initial_coef_table, 'linelistdict':linelistdict.copy(),\
                    'all_coefs':out_calib_h.copy(),'user_input': user_input, 'filenum':filenum,
                    'save_plots':self.save_plots, "savetemplate_funcs":self.savetemplate_funcs
                }
                obs2 = {
                    'comp': data[fib2s.tolist()], 'fulllinelist': self.all_lines.copy(),
                    'coef_table': initial_coef_table.copy(), 'linelistdict':linelistdict.copy(),\
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

                out_calib, out_linelist,lambdas, pixels, variances,badfits_a1 = tabs[0]
                out_calib_a2, out_linelist_a2,lambdas_a2, pixels_a2, variances_a2,badfits_a2 = tabs[1]

                ## The hand fit calibrations are in both returned dicts, remove from the second
                ## Assign the other calibration info from hand fits to the output dicts
                for key in lambdas_h.keys():
                    out_calib_a2.pop(key)
                    lambdas[key] = lambdas_h[key]
                    pixels[key] = pixels_h[key]
                    variances[key] = variances_h[key]
                    out_linelist[key] = out_linelist_h[key]

                ## The hand fit calibrations are in both returned dicts, remove from the second
                ## Assign the other calibration info from hand fits to the output dicts
                for key in lambdas_a2.keys():
                    out_calib[key] = out_calib_a2[key]
                    lambdas[key] = lambdas_a2[key]
                    pixels[key] = pixels_a2[key]
                    variances[key] = variances_a2[key]
                    out_linelist[key] = out_linelist_a2[key]

                badfits = np.unique(np.concatenate([badfits_a1,badfits_a2]))

            out_calib_b, out_linelist_b, lambdas_b, pixels_b, variances_b, wm, fm = \
                self.wavelength_fitting_by_line_selection(data, linelistdict, \
                                                          self.all_lines, initial_coef_table, select_lines=select_lines, \
                                                          filenum=filenum, subset=badfits,completed_coefs=out_calib.copy())
            for key in lambdas_b.keys():
                out_calib[key] = out_calib_b[key]
                lambdas[key] = lambdas_b[key]
                pixels[key] = pixels_b[key]
                variances[key] = variances_b[key]
                out_linelist[key] = out_linelist_b[key]

            if len(variances)!= len(out_calib):
                print("initial_coef_table:\n", initial_coef_table.keys())
                print("hand_fit_subset:\n",hand_fit_subset)
                print("badfits_a1:\n",badfits_a1)
                print("badfits_a2:\n",badfits_a2)
                print("badfits:\n",badfits)
                print("linelistdict:\n",linelistdict.keys())

                print("\n\n\n")
                print("Variances:\n",variances)
                print("Pixels:\n",pixels)
                print("Outlinelist:\n",out_linelist.keys())
                print("lambdas:\n",lambdas)
                print("out_calib:\n",out_calib)

                print("\n\n\n")
                print("Variances_h:\n",variances_h)
                print("Pixels_h:\n",pixels_h)
                print("Outlinelist_h:\n",out_linelist_h.keys())
                print("lambdas_h:\n",lambdas_h)
                print("out_calib_h:\n",out_calib_h)

                print("\n\n\n")
                print("Variances_a2:\n",variances_a2)
                print("Pixels_a2:\n",pixels_a2)
                print("Outlinelist_a2:\n",out_linelist_a2.keys())
                print("lambdas_a2:\n",lambdas_a2)
                print("out_calib_a2:\n",out_calib_a2)

                print("\n\n\n")
                print("Variances_b:\n",variances_b)
                print("Pixels_b:\n",pixels_b)
                print("Outlinelist_b:\n",out_linelist_b.keys())
                print("lambdas_b:\n",lambdas_b)
                print("out_calib_b:\n",out_calib_b)


            if select_lines:
                self.selected_lines = out_linelist.copy()
                select_lines = False

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

            initial_coef_table = out_calib_table.copy()

            self.final_calibrated_hdulists[pairnum] = hdulist
            self.filemanager.save_full_calib_dict(hdulist, self.lampstr_f, self.camera, self.config, filenum=filenum)

            gc.collect()

    def get_medianfits_of(self,ordered_dict):
        coarse_tables = [Table(coarse) for coarse in ordered_dict.values() if coarse is not None]
        initial_coef_table = OrderedDict()
        for fib in coarse_tables[0].colnames:
            coeff_arr = np.zeros((len(coarse_tables),6))
            for ii,tab in enumerate(coarse_tables):
                coeff_arr[ii,:] = tab[fib]
            coeff_med = np.median(coeff_arr,axis=0)
            initial_coef_table[fib] = coeff_med
        return initial_coef_table

    def get_parametricfits_of(self,ordered_dict,caltype='coarse'):
        from scipy.optimize import curve_fit
        ## assumes quadratic fits  ( as it only fits 3 params per coef)
        yparametrized_coefs = np.zeros(shape=(3,6))
        if caltype == 'coarse':
            ncoefs = 3

        else:
            ncoefs = 6
        coef_xys = {coef: {'x': [], 'y': []} for coef in range(ncoefs)}
        for pairnum,(cc_filnum, fc_filnum) in self.pairings.items():
            if caltype == 'coarse':
                header = dict(self.coarse_calibrations[cc_filnum].header)
            else:
                header = dict(self.fine_calibrations[fc_filnum].header)
            coarse_table = Table(ordered_dict[pairnum])
            if pairnum == 0:
                fibers = list(coarse_table.colnames)
                outheader = header.copy()
            for fiber in coarse_table.colnames:
                yval = header['YLOC_{}'.format(fiber[1:])]
                for ii in range(ncoefs):
                    ## note the rotation here. Otherwise we're fitting a sideways parabola
                    coef_xys[ii]['y'].append(coarse_table[fiber][ii])
                    coef_xys[ii]['x'].append(yval)

        for ii in range(ncoefs):
            fit_params,cov = curve_fit(f=quadratic,xdata=coef_xys[ii]['x'],ydata=coef_xys[ii]['y'])
            yparametrized_coefs[:,ii] = fit_params

        out_dict = OrderedDict()
        for fiber in fibers:
            yval = outheader['YLOC_{}'.format(fiber[1:])]
            coefs = np.zeros(6)
            for ii in range(ncoefs):
                coefs[ii] = quadratic(yval,yparametrized_coefs[0,ii],yparametrized_coefs[1,ii],yparametrized_coefs[2,ii])
            out_dict[fiber] = coefs

        import matplotlib.pyplot as plt

        offsets = np.array(coef_xys[0]['y'])
        min_off = np.min(offsets)
        shifted_offsets = offsets - min_off
        linears = np.array(coef_xys[1]['y'])
        shifted_linears = linears - 1
        quads = np.array(coef_xys[2]['y'])
        xs = np.array(coef_xys[1]['x'])

        fitted_xs = np.arange(2056)
        fitted_offsets = quadratic(fitted_xs, *yparametrized_coefs[:, 0])
        shifted_fitted_offsets = fitted_offsets - np.min(fitted_offsets)
        fitted_linears = quadratic(fitted_xs, *yparametrized_coefs[:, 1])
        shifted_fitted_linears = fitted_linears - 1
        fitted_quads = quadratic(fitted_xs, *yparametrized_coefs[:, 2])

        plt.figure()
        plt.title("Offsets")
        plt.plot(fitted_xs, fitted_offsets, 'r-', label='fit offset')
        plt.plot(xs, offsets, '.', label='offsets')
        plt.legend(loc='best')

        plt.figure()
        plt.title('Linears')
        plt.plot(xs, linears, '.', label='linears')
        plt.plot(fitted_xs, fitted_linears, 'r-', label='fit linear')
        plt.legend(loc='best')

        plt.figure()
        plt.title("Quads")
        plt.plot(fitted_xs, fitted_quads, 'r-', label='fit quad')
        plt.plot(xs, quads, '.', label='quads')
        plt.legend(loc='best')

        plt.figure()
        for pixel in [100, 1000, 2000]:
            plt.plot(fitted_xs,
                     shifted_fitted_offsets + (shifted_fitted_linears * pixel) + (fitted_quads * pixel * pixel), 'r-',
                     'fit {}'.format(pixel))
            plt.plot(xs, shifted_offsets + (shifted_linears * pixel) + (quads * pixel * pixel), '.', label=str(pixel))
            plt.title("Offset_fits_expanded {}".format(pixel))

        plt.legend(loc='best')
        plt.show()
        return out_dict


    def create_calibration_default(self,save=True):
        default_outtable = Table(self.get_medianfits_of(self.fine_calibration_coefs))
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
                                             bounds=None,filenum='',subset=[],completed_coefs = {}):
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
        for fiber in hand_fit_subset:
            if fiber in extremes:
                extrema_fiber = True
            else:
                extrema_fiber = False
            counter += 1
            f_x = comp[fiber].data

            ## HACK!!
            coefs = coef_table[fiber]
            # if fiber == 'r101':
            #     coefs = [4377.517995094989,
            #         0.9924585019372959,
            #         5.142950231647317e-06,
            #         5.410738316527114e-09, - 5.913504598741283e-12,
            #         1.295764253221375e-15]
            # if fiber == 'r816':
            #     coefs = [4369.022963110598,
            #         0.9912306552914903,
            #         1.2018259690352653e-05 ,- 5.34465531869862e-09,
            #         7.861696346099334e-13, - 1.6224337469649862e-16]
            # if fiber == 'r416':
            #     coefs = [4268.231048499829,
            #         1.003521407204926, - 1.0718754879101437e-05,
            #         2.2807984834809712e-08 ,- 1.471914199718311e-11,
            #         2.9274061328248153e-15]
            # if fiber == 'r501':
            #     coefs = [4266.999521530618,
            #         0.9997335166179256, - 2.504181224499703e-06,
            #         1.3765663270362488e-08, - 9.74601813932534e-12,
            #         1.8788845035312016e-15]
            # if fiber == 'r214':
            #     coefs = [4306.755027434568,
            #         0.996075246372463,
            #         7.795415099421535e-06, - 1.628278660204676e-09, - 8.109792880170026e-13,
            #         1.2141470243476217e-16]
            # if fiber == 'r303':
            #     coefs = [4289.203753188707,
            #         0.9996452204941928 ,- 2.726889657361145e-07,
            #         8.196734766456035e-09, - 6.0533976379463196e-12,
            #         1.1267086805921927e-15]
            # if fiber == 'r612':
            #     coefs = [4284.3107579147,
            #         0.9998319207691722, - 1.620205582018001e-06,
            #         1.0453879123151255e-08, - 7.469050894347352e-12,
            #         1.4246267819397567e-15]
            # if fiber == 'r705':
            #     coefs = [4303.1225144976115,
            #         1.0037566403877034, - 2.0388949216387168e-05,
            #         3.653978743018491e-08, - 2.2439189521386893e-11,
            #         4.454053636623292e-15]
            if select_lines:
                iteration_wm, iteration_fm = wm.copy(), fm.copy()
            else:
                iteration_wm, iteration_fm = selectedlistdict[fiber]

            ## HACK!!
            if len(completed_coefs)>0:# and fiber not in ['r705','r612','r303','r214','r501','r416','r101','r816']:
                curtet = int(fiber[1])
                tets,fibs = [],[]
                for fib in completed_coefs.keys():
                    tets.append(int(fib[1]))
                    fibs.append(fib)
                tets,fibs = np.array(tets),np.array(fibs)
                if 9-curtet in tets:
                    key = fibs[np.where((9-curtet)==tets)[0][0]]
                    coef_dev_med = np.asarray(completed_coefs[key])-np.asarray(coef_table[key])
                else:
                    coef_devs = np.zeros(shape=(len(completed_coefs),6)).astype(np.float64)
                    for ii,(key,key_coefs) in enumerate(completed_coefs.items()):
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
            params,covs,resid = browser.fit()

            print(fiber,*params)
            print(np.sqrt(resid))
            out_coefs[fiber] = params
            completed_coefs[fiber] = params
            variances[fiber] = covs.diagonal()
            app_fit_pix[fiber] = browser.line_matches['peaks_p']
            app_fit_lambs[fiber] = browser.line_matches['lines']

            template =  self.savetemplate_funcs(cam=str(filenum)+'_',ap=fiber,imtype='calib',step='finalfit',comment='byhand')
            if self.save_plots:
                browser.create_saveplot(params,covs, template)

            if select_lines:
                app_specific_linelists[fiber] = (browser.wm, browser.fm)
                init_deleted_wm = np.asarray(browser.last['wm'])
                init_deleted_fm = np.asarray(browser.last['fm'])
                # if fiber == 'r101':
                #     for w,f in zip(init_deleted_wm,init_deleted_fm):
                #         print(w,f)
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

                        if len(completed_coefs) > 0:
                            curtet = int(fiber[1])
                            tets, fibs = [], []
                            for fib in completed_coefs.keys():
                                tets.append(int(fib[1]))
                                fibs.append(fib)
                            tets, fibs = np.array(tets), np.array(fibs)
                            if 9 - curtet in tets:
                                key = fibs[np.where((9 - curtet) == tets)[0][0]]
                                coef_dev_med = np.asarray(completed_coefs[key]) - np.asarray(coef_table[key])
                            else:
                                coef_devs = np.zeros(shape=(len(completed_coefs), 6)).astype(np.float64)
                                for ii, (key, key_coefs) in enumerate(completed_coefs.items()):
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
                        params, covs, resid = browser.fit()

                        print(fiber, *params)
                        print(np.sqrt(resid))
                        out_coefs[fiber] = params
                        variances[fiber] = covs.diagonal()
                        app_fit_pix[fiber] = browser.line_matches['peaks_p']
                        app_fit_lambs[fiber] = browser.line_matches['lines']

                        if select_lines:
                            app_specific_linelists[fiber] = (browser.wm, browser.fm)

                        template = self.savetemplate_funcs(cam=str(filenum)+'_', ap=fiber, imtype='calib', step='finalfit',
                                                           comment='byhand')
                        if self.save_plots:
                            browser.create_saveplot(params, covs, template)

                        plt.close()
                        del browser

        if not select_lines:
            app_specific_linelists = selectedlistdict
            wm,fm = iteration_wm, iteration_fm

        return out_coefs, app_specific_linelists, app_fit_lambs, app_fit_pix, variances, wm, fm


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
        for fiber_identifier in coarse_comp.colnames:
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

def auto_wavelength_fitting_by_lines_wrapper(input_dict):
    return auto_wavelength_fitting_by_lines(**input_dict)


def auto_wavelength_fitting_by_lines(comp, fulllinelist, coef_table, linelistdict,all_coefs,user_input='some',filenum='',\
                                     bounds=None, save_plots = True,  savetemplate_funcs='{}{}{}{}{}'.format):
    if 'ThAr' in linelistdict.keys():
        wm, fm = linelistdict['ThAr']
        app_specific = False
    else:
        app_specific = True
    comp = Table(comp)

    variances = {}
    app_fit_pix = {}
    app_fit_lambs = {}
    outlinelist = {}
    hand_fit_subset = np.array(list(all_coefs.keys()))

    cam = comp.colnames[0][0]

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

    # ## go from outside in
    # if cam == 'r' and int(all_fibers[0][1]) > 3:
    #     all_fibers = all_fibers[::-1]
    # elif cam =='b' and int(all_fibers[0][1]) < 6:
    #     all_fibers = all_fibers[::-1]

    ## go from inside out
    if cam == 'r' and int(all_fibers[0][1]) < 4:
        all_fibers = all_fibers[::-1]
    elif cam =='b' and int(all_fibers[0][1]) > 5:
        all_fibers = all_fibers[::-1]

    upper_limit_resid = 0.4
    for itter in range(10):
        badfits = []
        maxdevs = []
        upper_limit_resid += itter/100.
        for fiber in all_fibers:
            if fiber in hand_fit_subset:
                continue
            if fiber not in coef_table.colnames:
                continue
            if app_specific:
                wm,fm = linelistdict[fiber]
            coefs = np.asarray(coef_table[fiber])
            f_x = comp[fiber].data

            if cam == 'b':
                fibern = (16 * (9 - int(fiber[1]))) + int(fiber[2:])
            else:
                fibern = (16 * int(fiber[1])) + int(fiber[2:])

            if len(hand_fit_subset)>0:
                # if len(hand_fit_subset)>1:
                #     dists = np.abs(fibern-numeric_hand_fit_names)
                #     closest = np.argsort(dists)[:2]
                #     nearest_fibs = hand_fit_subset[closest]
                #     # diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                #     # diffs_fib2 = np.asarray(all_coefs[nearest_fibs[1]]) - np.asarray(coef_table[nearest_fibs[1]])
                #     # d1,d2 = dists[closest[0]], dists[closest[1]]
                #     # diffs_hfib = (d2*diffs_fib1 + d1*diffs_fib2)/(d1+d2)
                #     diffs_hfib = closest
                # else:
                #     nearest_fibs = hand_fit_subset[np.argsort(np.abs(fibern-numeric_hand_fit_names))]
                #     diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                #     diffs_hfib = diffs_fib1
                #
                # if last_fiber is None:
                #     last_fiber = nearest_fibs[0]
                #
                # nearest_fib = np.asarray(all_coefs[last_fiber]) - np.asarray(coef_table[last_fiber])
                #
                # diffs_mean = (0.5*diffs_hfib)+(0.5*nearest_fib)
                #
                # adjusted_coefs_guess = coefs+diffs_mean
                nearest_fibs = hand_fit_subset[np.argsort(np.abs(fibern - numeric_hand_fit_names))]
                diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                diffs_mean = diffs_fib1
                adjusted_coefs_guess = coefs + diffs_mean
            else:
                adjusted_coefs_guess = coefs

            browser = LineBrowser(wm,fm, f_x, adjusted_coefs_guess, fulllinelist, bounds=None, edge_line_distance=(-20.0),initiate=False)

            params,covs, resid = browser.fit()

            print('\n\n',fiber,'{:.2f} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}'.format(*params))
            fitlamb = np.polyval(params[::-1],np.asarray(browser.line_matches['peaks_p']))
            dlamb = fitlamb - browser.line_matches['lines']
            print("  ----> mean={}, median={}, std={}, sqrt(resid)={}".format(np.mean(dlamb),np.median(dlamb),np.std(dlamb),np.sqrt(resid)))

            if np.max(np.abs(dlamb)) > 1.0 and np.sqrt(resid) < 1.0:
                maxdev_line = browser.line_matches['lines'][np.argmax(np.abs(dlamb))]
                maxdevs.append(maxdev_line)
            if np.sqrt(resid) < upper_limit_resid:
                if save_plots:
                    template = savetemplate_funcs(cam=str(filenum) + '_', ap=fiber, imtype='calib', step='finalfit',
                                                  comment='auto')
                    browser.initiate_browser()
                    browser.create_saveplot(params, covs, template)

                all_coefs[fiber] = params
                variances[fiber] = covs.diagonal()
                outlinelist[fiber] = (wm, fm)
                app_fit_pix[fiber] = browser.line_matches['peaks_p']
                app_fit_lambs[fiber] = browser.line_matches['lines']
                # if np.sqrt(resid) < upper_limit_resid:
                numeric_hand_fit_names = np.append(numeric_hand_fit_names,fibern)
                hand_fit_subset = np.append(hand_fit_subset,fiber)
            else:
                badfits.append(fiber)

            plt.close()
            del browser

        if (len(badfits) > 0) and (len(maxdevs) > 0) and (len(wm) > 11):
            count = Counter(maxdevs)
            line, num = count.most_common(1)[0]
            if (num > (len(maxdevs)//2)) and (num > (len(all_fibers)//2)):
                wm_loc = np.where(wm == line)[0][0]
                wmlist,fmlist = wm.tolist(),fm.tolist()
                wmlist.pop(wm_loc)
                fmlist.pop(wm_loc)
                wm,fm = np.asarray(wmlist),np.asarray(fmlist)

        all_fibers = np.array(badfits)[::-1]

    return all_coefs, outlinelist, app_fit_lambs, app_fit_pix, variances, all_fibers


def quadratic(xs,a,b,c):
    return a + b*xs + c*xs*xs

