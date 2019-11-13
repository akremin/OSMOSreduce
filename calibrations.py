
import pickle as pkl
import gc
from collections import OrderedDict
from multiprocessing import Pool
import datetime as dt
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack

from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter

from calibration_helper_funcs import compare_outputs, interactive_plot, \
    get_highestflux_waves, update_default_dict, \
    top_peak_wavelengths, pix_to_wave,\
    ensure_match, find_devs, coarse_calib_configure_tables,\
    get_fiber_number, pix_to_wave_explicit_coefs2, get_meantime_and_date

from coarse_calibration import run_automated_calibration, run_automated_calibration_wrapper
from fine_calibration import auto_wavelength_fitting_by_lines, auto_wavelength_fitting_by_lines_wrapper
from linebrowser import LineBrowser
from collections import Counter

class Calibrations:
    def __init__(self, camera, instrument, coarse_calibrations, filemanager,  \
                 fine_calibrations=None, pairings=None, load_history=True, trust_after_first=False,\
                 default_fit_key='cross correlation',use_selected_calib_lines=False, \
                 single_core=False, save_plots=False, show_plots=False):

        self.imtype = 'comp'

        self.camera = camera
        self.instrument = instrument
        self.filemanager = filemanager
        self.config = instrument.configuration
        self.lamptypesc = instrument.coarse_lamp_names
        self.lamptypesf = instrument.fine_lamp_names
        self.wavemin = instrument.wavemin
        self.wavemax = instrument.wavemax
        self.default_fit_key = default_fit_key
        self.savetemplate_funcs = filemanager.get_saveplot_template

        self.trust_after_first = trust_after_first
        self.single_core = single_core
        self.save_plots = save_plots
        self.show_plots = show_plots

        #self.linelistc,selected_linesc,all_linesc = filemanager.load_calibration_lines_dict(lamptypesc,use_selected=use_selected_calib_lines)
        self.linelistc,all_linesc = filemanager.load_calibration_lines_dict(self.lamptypesc,wavemincut=self.wavemin, \
                                                                wavemaxcut=self.wavemax,use_selected=use_selected_calib_lines)

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

        for lamp in self.lamptypesc:
            self.lampstr_c += '-'+str(lamp)

        if self.do_fine_calib:
            self.linelistf,self.all_lines = filemanager.load_calibration_lines_dict(self.lamptypesf,wavemincut=self.wavemin, \
                                                                wavemaxcut=self.wavemax,use_selected=use_selected_calib_lines)
            #self.linelistf,self.selected_lines,self.all_lines = filemanager.load_calibration_lines_dict(lamptypesf,use_selected=use_selected_calib_lines)
            self.fine_calibrations = fine_calibrations
            for lamp in self.lamptypesf:
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
        self.fine_calibration_date_info = OrderedDict()
        self.interpolated_coef_fits = OrderedDict()

        self.load_default_coefs()
        if load_history:
            self.load_most_recent_coefs()

    def load_default_coefs(self):
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

    def generate_time_interpolated_calibration(self,science_hdu):
        if len(self.interpolated_coef_fits) == 0:
            self.interpolate_final_calibrations()
        mean_timestamp, mean_datetime, night = get_meantime_and_date(science_hdu.header)

        night_interpolator,fit_model = self.interpolated_coef_fits[night]
        out_cols = []
        for fib, coefs in night_interpolator.items():
            fitted_coefs = []
            for coef_name, coefvals in coefs.items():
                fitted_coef = fit_model(mean_timestamp,*coefvals)
                fitted_coefs.append(fitted_coef)
            out_cols.append(Table.Column(name=fib,data=fitted_coefs))
        out_table = Table(out_cols)
        return out_table


    def interpolate_final_calibrations(self):
        assert(len(self.fine_calibration_date_info) > 0 and len(self.fine_calibration_coefs) > 0,\
               "Can't interpolate data until the fine calibrations are performed and properly loaded. Exiting")

        coef_names = ['a', 'b', 'c', 'd', 'e', 'f']
        pairnums_by_night,mean_timestamp_by_night = {},{}
        for pairnum in self.fine_calibration_coefs.keys():
            mean_timestamp, mean_datetime, night = self.fine_calibration_date_info[pairnum]
            if night in pairnums_by_night.keys():
                pairnums_by_night[night].append(pairnum)
                mean_timestamp_by_night[night].append(mean_timestamp)
            else:
                pairnums_by_night[night] = [pairnum]
                mean_timestamp_by_night[night] =[mean_timestamp]
        for night, pairnums in pairnums_by_night.items():
            fiber_fit_dict = OrderedDict()
            if len(pairnums) == 1:
                ## no interpolation
                def fit_model(xs, a):
                    return a
            else:
                if len(pairnums) == 2:
                    ## linear interp
                    p0 = [0, 0]
                    def fit_model(xs, a, b):
                        return a + b * xs
                else:
                    ## quadratic
                    p0 = [0, 0, 0]
                    def fit_model(xs, a, b, c):
                        return a + b * xs + c * xs * xs

            for fib in self.fine_calibration_coefs[pairnums[0]].colnames:
                coef_arrs,coef_fits = OrderedDict(), OrderedDict()

                for coef in coef_names:
                    coef_arrs[coef] = []
                for pairnum in pairnums:
                    for ii, coef in enumerate(coef_arrs.keys()):
                        coef_arrs[coef].append(self.fine_calibration_coefs[pairnum][fib][ii])
                    if len(pairnums) == 1:
                        for coef, coef_arr in coef_arrs.items():
                            coef_fits[coef] = coef_arr
                    else:
                        for coef,coef_arr in coef_arrs.items():
                            p0[0] = coef_arr[0]
                            params,cov = curve_fit(fit_model,mean_timestamp_by_night[night],coef_arr,p0=p0)
                            coef_fits[coef] = params
                fiber_fit_dict[fib] = coef_fits
            self.interpolated_coef_fits[night] = (fiber_fit_dict, fit_model)



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

        if use_history_calibs and self.history_calibration_coefs[0] is not None:
            histories = self.get_medianfits_of(self.history_calibration_coefs)
        else:
            histories = None

        for pairnum,(cc_filnum, throwaway) in self.pairings.items():

            comp_data = Table(self.coarse_calibrations[cc_filnum].data)
            print("\n\n#########################################")
            print("####   Now running {:d} (filnum={:04d})   ####".format(pairnum,cc_filnum))
            print("#########################################\n")
            if self.single_core:
                obs1 = {
                    'coarse_comp': comp_data, 'complinelistdict': self.linelistc,
                    'last_obs': histories     }

                out = run_automated_calibration_wrapper(obs1)
                tabs = coarse_calib_configure_tables(out)
                out_calib = tabs['coefs']
                out_calib = out_calib[self.instrument.full_fibs[self.camera].tolist()]

                out_metric = tabs['metric']
                out_metric = out_metric[self.instrument.full_fibs[self.camera].tolist()]

                out_lines = tabs['clines']
                out_lines = out_lines[self.instrument.full_fibs[self.camera].tolist()]

                out_pixels = tabs['pixels']
                out_pixels = out_pixels[self.instrument.full_fibs[self.camera].tolist()]
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
                obs1 = {'coarse_comp': comp_data[fib1s.tolist()], 'complinelistdict': self.linelistc, 'last_obs': hist1}
                obs2 = {'coarse_comp': comp_data[fib2s.tolist()], 'complinelistdict': self.linelistc, 'last_obs': hist2}

                all_obs = [obs1, obs2]
                if len(all_obs) < 4:
                    NPROC = len(all_obs)
                else:
                    NPROC = 4

                with Pool(NPROC) as pool:
                    outs = pool.map(run_automated_calibration_wrapper, all_obs)

                # out = {'coefs': coeftab,
                #        'metric': metrictab,
                #        'clines': linestab,
                #        'pixels': pixtab}
                tabs0 = coarse_calib_configure_tables(outs[0])
                tabs1 = coarse_calib_configure_tables(outs[1])

                overlaps = self.instrument.overlapping_fibs[self.camera]

                template = self.savetemplate_funcs(cam=str(cc_filnum) + '_', ap='{fiber}', imtype='coarse_calib',
                                                   step='calib_directionfit_comparison', comment='auto')
                matches = compare_outputs(comp_data, tabs0['coefs'],tabs1['coefs'],save_template=template,\
                                          save_plots=self.save_plots,show_plots=self.show_plots)
                matches = list(matches)

                if np.any(np.sort(overlaps) != np.sort(matches)):
                    print("The overlaps returned from the matching didn't match the overlaps")
                    print("that were explicitly defined: {}  {}".format(matches, overlaps))
                if pairnum > 0:
                    lastcc,lastfc = self.pairings[pairnum-1]
                    template = self.savetemplate_funcs(cam='{}-{}_'.format(cc_filnum,lastcc), ap='{fiber}',
                                                       imtype='coarse_calib',
                                                       step='calib_time_comparison', comment='auto')
                    matches = compare_outputs(comp_data, tabs0['coefs'][overlaps.tolist()],\
                                              self.coarse_calibration_coefs[pairnum-1][overlaps.tolist()],\
                                              save_template=template, \
                                              save_plots=self.save_plots, show_plots=self.show_plots)
                for tab in tabs0:
                    tab.remove_column(overlaps[1])
                for tab in tabs1:
                    tab.remove_column(overlaps[0])

                out_calib = hstack([tabs0['coefs'], tabs1['coefs']])
                out_calib = out_calib[self.instrument.full_fibs[self.camera].tolist()]

                out_metric = hstack([tabs0['metric'], tabs1['metric']])
                out_metric = out_metric[self.instrument.full_fibs[self.camera].tolist()]

                out_lines = hstack([tabs0['clines'], tabs1['clines']])
                out_lines = out_lines[self.instrument.full_fibs[self.camera].tolist()]

                out_pixels = hstack([tabs0['pixels'], tabs1['pixels']])
                out_pixels = out_pixels[self.instrument.full_fibs[self.camera].tolist()]

            self.coarse_calibration_coefs[pairnum] = out_calib.copy()
            prim = fits.PrimaryHDU(header=self.coarse_calibrations[cc_filnum].header)
            calibs = fits.BinTableHDU(data=out_calib, name='calib coefs')
            varis = fits.BinTableHDU(data=out_metric, name='metric')
            lambs = fits.BinTableHDU(data=out_lines, name='wavelengths')
            pix = fits.BinTableHDU(data=out_pixels, name='pixels')
            hdulist = fits.HDUList([prim, calibs, lambs, pix, varis])

            self.filemanager.save_full_calib_dict(hdulist, self.lampstr_c, self.camera, self.config, filenum=cc_filnum)

            histories = out_calib


    def run_final_calibrations(self,initial_priors='parametric'):
        if not self.do_fine_calib:
            print("There doesn't seem to be a fine calibration defined. Using the supplied coarse calibs")
        select_lines = True

        dev_allowance = 1.
        devs = 2.
        using_defaults = False
        if initial_priors == 'defaults':
            if self.default_calibration_coefs is None:
                print("Couldn't find the default calibration coefficients, so using a parametrization of the coarse coefs")
                initial_coef_table = Table(self.get_parametricfits_of(caltype='coarse'))
            else:
                need_to_parametrize = False
                initial_coef_table = Table(self.default_calibration_coefs)
                for fib in self.instrument.full_fibs[self.camera]:
                    if fib not in initial_coef_table.colnames:
                        need_to_parametrize = True
                        break
                if need_to_parametrize:
                    paramd_table = Table(self.get_parametricfits_of(caltype='default'))
                    for fib in self.instrument.full_fibs[self.camera]:
                        if fib not in initial_coef_table.colnames:
                            initial_coef_table[fib] = paramd_table[fib]
                using_defaults = True
        elif initial_priors == 'medians':
            initial_coef_table = Table(self.get_medianfits_of(self.coarse_calibration_coefs))
        else:
            initial_coef_table = Table(self.get_parametricfits_of(caltype='coarse'))

        for pairnum,filnums in self.pairings.items():
            filenum = filnums[self.filenum_ind]

            data = Table(self.fine_calibrations[filenum].data)
            self.fine_calibration_date_info[pairnum] = get_meantime_and_date(self.fine_calibrations[filenum].header)

            linelist = self.selected_lines

            effective_iteration = np.max([pairnum,int(using_defaults)])
            if effective_iteration == 0:
                user_input = 'some'
            elif effective_iteration == 1:
                user_input = 'minimal'
            elif effective_iteration > 1:# and devs < dev_allowance:
                user_input = 'single'#'none'

            hand_fit_subset = []
            cam = self.camera
            if user_input == 'all':
                hand_fit_subset = list(initial_coef_table.colnames)
            elif user_input in ['some', 'minimal', 'single']:
                if cam == 'r':
                    specific_set = [cam + '101', cam + '816', cam + '416', cam + '501']
                else:
                    specific_set = [cam + '116', cam + '801', cam + '516', cam + '401']
                for i, fib in enumerate(specific_set):
                    outfib = ensure_match(fib, data.colnames, hand_fit_subset, cam)
                    hand_fit_subset.append(outfib)

                if user_input == 'some':
                    seed = int(filenum)
                    np.random.seed(seed)
                    randfibs = ['{:02d}'.format(x) for x in np.random.randint(1, 16, 4)]
                    for tetn, fibn in zip([2, 3, 6, 7], randfibs):
                        fib = '{}{}{}'.format(cam, tetn, fibn)
                        outfib = ensure_match(fib, data.colnames, hand_fit_subset, cam)
                        hand_fit_subset.append(outfib)
                elif user_input == 'single':
                    hand_fit_subset = hand_fit_subset[:1]
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
                out_calib, out_linelist, lambdas, pixels, variances, badfits = \
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

    def get_parametricfits_of(self,caltype='coarse'):
        from scipy.optimize import curve_fit
        ## assumes quadratic fits  ( as it only fits 3 params per coef)
        yparametrized_coefs = np.zeros(shape=(3,6))
        fibers = list(self.instrument.full_fibs[self.camera])
        if caltype == 'default':
            ncoefs = 6
            ordered_dict = OrderedDict()
            ordered_dict[0] = self.default_calibration_coefs
            header = dict(self.coarse_calibrations[self.pairings[0][0]].header)
        elif caltype == 'fine':
            ncoefs = 6
            ordered_dict = self.fine_calibration_coefs
            header = dict(self.fine_calibrations[self.pairings[0][1]].header)
        elif caltype == 'coarse':
            ncoefs = 3
            ordered_dict = self.coarse_calibration_coefs
            header = dict(self.coarse_calibrations[self.pairings[0][0]].header)
        else:
            ncoefs = 3
            ordered_dict = self.coarse_calibration_coefs
            header = dict(self.coarse_calibrations[self.pairings[0][0]].header)

        coef_xys = {coef: {'x': [], 'y': []} for coef in range(ncoefs)}

        for pairnum,iterdict in ordered_dict.items():
            iter_table = Table(iterdict)
            for fiber in fibers:
                if fiber in iter_table.colnames:
                    yval = header['YLOC_{}'.format(fiber[1:])]
                    for ii in range(ncoefs):
                        ## note the rotation here. Otherwise we're fitting a sideways parabola
                        coef_xys[ii]['y'].append(iter_table[fiber][ii])
                        coef_xys[ii]['x'].append(yval)

        for ii in range(ncoefs):
            fit_params,cov = curve_fit(f=pix_to_wave_explicit_coefs2,xdata=coef_xys[ii]['x'],ydata=coef_xys[ii]['y'])
            yparametrized_coefs[:,ii] = fit_params

        out_dict = OrderedDict()
        for fiber in fibers:
            yval = header['YLOC_{}'.format(fiber[1:])]
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

        template = self.savetemplate_funcs(cam=self.camera, ap='_all',
                                           imtype=caltype,
                                           step='parametric_calib_coef_fits', comment='{coeff}')
        plt.figure()
        plt.title("Offsets {}".format(self.camera))
        plt.plot(fitted_xs, fitted_offsets, 'r-', label='fit offset')
        plt.plot(xs, offsets, '.', label='offsets')
        plt.legend(loc='best')
        if self.save_plots:
            plt.savefig(template.format(coeff='Offset'))

        plt.figure()
        plt.title('Linears {}'.format(self.camera))
        plt.plot(xs, linears, '.', label='linears')
        plt.plot(fitted_xs, fitted_linears, 'r-', label='fit linear')
        plt.legend(loc='best')
        if self.save_plots:
            plt.savefig(template.format(coeff='linear'))

        plt.figure()
        plt.title("Quads {}".format(self.camera))
        plt.plot(fitted_xs, fitted_quads, 'r-', label='fit quad')
        plt.plot(xs, quads, '.', label='quads')
        plt.legend(loc='best')
        if self.save_plots:
            plt.savefig(template.format(coeff='quadratic'))

        plt.figure()
        for pixel in [100, 1000, 2000]:
            plt.plot(fitted_xs,
                     shifted_fitted_offsets + (shifted_fitted_linears * pixel) + (fitted_quads * pixel * pixel), 'r-',
                     'fit {}'.format(pixel))
            plt.plot(xs, shifted_offsets + (shifted_linears * pixel) + (quads * pixel * pixel), '.', label=str(pixel))
        plt.title("Offset_fits_expanded {}".format(self.camera))
        plt.legend(loc='best')
        if self.save_plots:
            plt.savefig(template.format(coeff='example_pixels'))
        if self.show_plots:
            plt.show()
        plt.close('all')
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

    from calibration_helper_funcs import run_interactive_slider_calibration





def quadratic(xs,a,b,c):
    return a + b*xs + c*xs*xs

