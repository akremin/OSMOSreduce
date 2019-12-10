import gc
from collections import OrderedDict
from multiprocessing import Pool

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
from scipy.optimize import curve_fit

from calibration_helper_funcs import compare_outputs, create_simple_line_spectra, \
    ensure_match, find_devs, coarse_calib_configure_tables, \
    pix_to_wave_explicit_coefs2, get_meantime_and_date
from coarse_calibration import run_automated_calibration_wrapper
from fine_calibration import auto_wavelength_fitting_by_lines, auto_wavelength_fitting_by_lines_wrapper, \
    wavelength_fitting_by_line_selection


class Calibrations:
    def __init__(self, camera, instrument, coarse_calibrations, filemanager,  \
                 fine_calibrations=None, pairings=None, load_history=True, trust_after_first=False,\
                 default_fit_key='cross correlation',use_selected_calib_lines=False, \
                 single_core=False, save_plots=False, show_plots=False, try_load_finals=False):

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
            mock_spec_wls, mock_spec_fls = filemanager.load_synth_calibration_spec(self.lamptypesf,
                                                                                         wavemincut=self.wavemin, \
                                                                                         wavemaxcut=self.wavemax)
            self.mock_spec_w, self.mock_spec_f = create_simple_line_spectra(['ThAr'], {'ThAr': (mock_spec_wls, mock_spec_fls)}, \
                               wave_low=self.wavemin, wave_high=self.wavemax, clab_step=0.01)

            #self.linelistf,self.selected_lines,self.all_lines = filemanager.load_calibration_lines_dict(lamptypesf,use_selected=use_selected_calib_lines)
            self.fine_calibrations = fine_calibrations
            for lamp in self.lamptypesf:
                self.lampstr_f += '-' + str(lamp)
        else:
            self.linelistf = self.linelistc.copy()
            #self.selected_lines = selected_linesc
            self.all_lines = all_linesc
            self.lampstr_f = self.lampstr_c.replace('basic','full')
            self.fine_calibrations = coarse_calibrations
            self.mock_spec_w, self.mock_spec_f = None, None

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

        if try_load_finals:
            self.load_final_calib_hdus()
        else:
            self.load_default_coefs()
            if load_history:
                self.load_most_recent_coefs()

            if try_load_finals:
                self.load_final_calib_hdus()

    def load_final_calib_hdus(self):
        couldntfind = False

        if type(self.lamptypesf) in [list, np.ndarray]:
            name = 'full-' + '-'.join(self.lamptypesf)
        else:
            name = 'full-' + self.lamptypesf
        for pairnum, filnums in self.pairings.items():
            filnum = filnums[self.filenum_ind]
            calib, thetype = self.filemanager.locate_calib_dict(name, self.camera, self.instrument.configuration,
                                                                filnum, locate_type='full')
            if calib is None:
                couldntfind = True
                break
            elif thetype != 'full':
                print("Something went wrong when loading calibrations")
                print("Specified 'full' but got back {}".format(thetype))
                couldntfind = True
                break
            else:
                self.final_calibrated_hdulists[pairnum] = calib
                self.fine_calibration_coefs[pairnum] = Table(calib['calib coefs'].data)
        if couldntfind:
            print("Couldn't find matching calibrations. Please make sure the step has been run fully")


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

        if night in self.interpolated_coef_fits.keys():
            night_interpolator,fit_model = self.interpolated_coef_fits[night]
        else:
            print("WARNING: Requested {}, but that isn't a known night. Possible options were: ".format(night),self.interpolated_coef_fits.keys())
            night_interpolator, fit_model = self.interpolated_coef_fits[night]
            timestamps,nights = [],[]
            for (mean_timestamp, mean_datetime, night) in self.fine_calibration_date_info.values():
                timestamps.append(mean_timestamp)
                nights.append(night)
            time_devs = np.array(timestamps)-mean_timestamp
            closest_night_loc = np.argmin(time_devs)
            closest_night = nights[closest_night_loc]
            print("Using night: {} instead, as it's nearest at {}m away".format(closest_night,time_devs[closest_night_loc]/60.))
            night_interpolator, fit_model = self.interpolated_coef_fits[closest_night]


        out_cols = []
        for fib, coefs in night_interpolator.items():
            fitted_coefs = []
            for coef_name, coefvals in coefs.items():
                if fit_model is None:
                    fitted_coef = coefvals(mean_timestamp)
                else:
                    fitted_coef = fit_model(mean_timestamp,*coefvals)
                fitted_coefs.append(fitted_coef)
            out_cols.append(Table.Column(name=fib,data=fitted_coefs))
        out_table = Table(out_cols)
        return out_table


    def interpolate_final_calibrations(self):
        from scipy.interpolate import UnivariateSpline
        dointerp = True
        doquadratic = False
        if len(self.fine_calibration_coefs) == 0:
            print("Can't interpolate data until the fine calibrations are performed and properly loaded. Exiting")

        coef_names = ['a', 'b', 'c', 'd', 'e', 'f']
        pairnums_by_night,mean_timestamps_by_night = {},{}
        for pairnum,(cc_filnum,cf_filnum) in self.pairings.items():
            mean_timestamp, mean_datetime, night = get_meantime_and_date(self.final_calibrated_hdulists[pairnum][0].header)
            self.fine_calibration_date_info[pairnum] = (mean_timestamp, mean_datetime, night)
            if night in pairnums_by_night.keys():
                pairnums_by_night[night].append(pairnum)
                mean_timestamps_by_night[night].append(mean_timestamp)
            else:
                pairnums_by_night[night] = [pairnum]
                mean_timestamps_by_night[night] = [mean_timestamp]

        for night, pairnums in pairnums_by_night.items():
            mean_timestamps = np.array(mean_timestamps_by_night[night])
            srtd_timestamp_inds = np.argsort(mean_timestamps)
            srtd_timestamps = mean_timestamps[srtd_timestamp_inds]
            srtd_pairnums = np.array(pairnums)[srtd_timestamp_inds]
            fiber_fit_dict,fiber_dat_dict = OrderedDict(),OrderedDict()
            if len(srtd_pairnums) == 1:
                ## no interpolation
                def fit_model(xs, a):
                    return a
            elif len(srtd_pairnums) == 2:
                ## linear interp
                def fit_model(xs, a, b):
                    return a + b*xs
            else:
                if doquadratic:
                    ## quadratic
                    p0 = [0, 0, 0]
                    def fit_model(xs, a, b, c):
                        return a + b * xs + c * xs * xs
                elif dointerp:
                    fit_model = None
                else:
                    # linear interp
                    p0 = [0, 0]
                    def fit_model(xs, a, b):
                        return a + b * xs

            if len(srtd_pairnums) == 3 and doquadratic:
                def getc(xlist,ylist):
                    x1, x2, x3 = xlist[:3]
                    y1, y2, y3 = ylist[:3]
                    return (1 / (x1 - x2)) * (((y1 - y3) / (x1 - x3)) - ((y2 - y3) / (x2 - x3)))

                def getb(xlist,ylist):
                    x1, x2, x3 = xlist[:3]
                    y1, y2, y3 = ylist[:3]
                    first = ((y2 - y3) / (x2 - x3))
                    second = (((x2 + x3) / (x1 - x2)) * (((y1 - y3) / (x1 - x3)) - ((y2 - y3) / (x2 - x3))))
                    return first - second

                def geta(x3, y3, b, c):
                    return y3 - b * x3 - c * x3 * x3

            for fib in self.fine_calibration_coefs[srtd_pairnums[0]].colnames:
                coef_arrs,coef_fits = OrderedDict(), OrderedDict()

                for coef in coef_names:
                    coef_arrs[coef] = []
                for pairnum in srtd_pairnums:
                    for ii, coef in enumerate(coef_arrs.keys()):
                        coef_arrs[coef].append(self.fine_calibration_coefs[pairnum][fib][ii])
                if len(srtd_pairnums) == 1:
                    for coef, coef_arr in coef_arrs.items():
                        coef_fits[coef] = coef_arr
                elif len(srtd_pairnums) == 2:
                    for coef, coef_arr in coef_arrs.items():
                        b = (coef_arr[1]-coef_arr[0])/(srtd_timestamps[1]-srtd_timestamps[0])
                        a = coef_arr[0] - b*srtd_timestamps[0]
                        coef_fits[coef] = [a,b]
                elif len(srtd_pairnums) == 3 and doquadratic:
                    for coef, coef_arr in coef_arrs.items():
                        c = getc(srtd_timestamps, coef_arr)
                        b = getb(srtd_timestamps, coef_arr)
                        a = geta(srtd_timestamps[0], coef_arr[0], b, c)
                        coef_fits[coef] = [a,b,c]
                elif dointerp:
                    for coef,coef_arr in coef_arrs.items():
                        coef_fits[coef] = UnivariateSpline(srtd_timestamps,coef_arr,k=1,s=0)
                else:
                    for coef,coef_arr in coef_arrs.items():
                        p0[0] = coef_arr[0]
                        params,cov = curve_fit(fit_model,srtd_timestamps,coef_arr,p0=p0)
                        coef_fits[coef] = params
                fiber_fit_dict[fib] = coef_fits
                fiber_dat_dict[fib] = coef_arrs
            self.interpolated_coef_fits[night] = (fiber_fit_dict, fit_model)

            plt.subplots(3, 2)
            if self.save_plots or self.show_plots:
                means = []
                minmeans = srtd_timestamps - srtd_timestamps[0]
                if len(minmeans)> 1:
                    interpd_minmeans = np.arange(minmeans[-1])
                else:
                    interpd_minmeans = minmeans
                for fib in fiber_fit_dict.keys():
                    means.extend(list(minmeans))
                for ii,name in enumerate(coef_names):
                    plt.subplot(3,2,int(1+ii))
                    dats = []
                    for fib,coef_fits in fiber_fit_dict.items():
                        dat = np.array(fiber_dat_dict[fib][name])
                        if fit_model is None:
                            fitd = coef_fits[name](interpd_minmeans+srtd_timestamps[0])
                        else:
                            fitd = fit_model(interpd_minmeans+srtd_timestamps[0], *coef_fits[name])
                        plt.plot(interpd_minmeans, fitd, alpha=0.1)
                        dats.extend(list(dat))

                    plt.plot([], [], '-', alpha=0.3, label='Fits')
                    plt.plot(means, dats, '.',alpha=0.3, label='Data')

                    if ii in [1,3,5]:
                        plt.ylabel('Value')
                    if ii in [5,6]:
                        plt.xlabel(r'$\Delta$Time [s]')
                    plt.title("{}".format(name))
                    if ii == 0:
                        plt.legend()
                plt.suptitle("{} Calibration Evolution".format(night))
                plt.tight_layout()

            if self.save_plots:
                plt.savefig(self.filemanager.get_saveplot_template(cam='', ap='all', imtype='science', step='calib',
                                                                   comment='_coefficients_interpolation'), dpi=600)
            if self.show_plots:
                plt.show()
            plt.close()

            plt.subplots(3, 2)
            if self.save_plots or self.show_plots:
                for ii,name in enumerate(coef_names):
                    plt.subplot(3,2,int(1+ii))
                    dats = []
                    for fib,coef_fits in fiber_fit_dict.items():
                        dat = np.array(fiber_dat_dict[fib][name])
                        if fit_model is None:
                            fitd = coef_fits[name](interpd_minmeans + srtd_timestamps[0])-dat[0]
                        else:
                            fitd = fit_model(interpd_minmeans+srtd_timestamps[0], *coef_fits[name])-dat[0]
                        plt.plot(interpd_minmeans, fitd, alpha=0.1)
                        dats.extend(list(dat-dat[0]))

                    plt.plot([], [], '-', alpha=0.3, label='Fits-Data[0]')
                    plt.plot(means, dats, '.',alpha=0.3, label='Data-Data[0]')
                    if ii+1 in [1,3,5]:
                        plt.ylabel(r'$\Delta$Value')
                    if ii+1 in [5,6]:
                        plt.xlabel(r'$\Delta$Time [s]')
                    plt.title("{}".format(name))
                    if ii+1 == 1:
                        plt.legend()
                plt.suptitle("{} Calibration Evolution".format(night))
                plt.tight_layout()

            if self.save_plots:
                plt.savefig(self.filemanager.get_saveplot_template(cam='', ap='all', imtype='science', step='calib',
                                                                   comment='_deviation_in_coefficients_interpolation'), dpi=600)
            if self.show_plots:
                plt.show()
            plt.close()


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
                obs1 = { 'coarse_comp': comp_data, 'complinelistdict': self.linelistc, 'last_obs': histories }
                out = run_automated_calibration_wrapper(obs1)
                tabs = coarse_calib_configure_tables(out)
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
                NPROC = np.clip(len(all_obs),1,4)

                with Pool(NPROC) as pool:
                    outs = pool.map(run_automated_calibration_wrapper, all_obs)

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
                tabs = {}

                for tabname in tabs0.keys():
                    tabs0[tabname].remove_column(overlaps[1])
                    tabs1[tabname].remove_column(overlaps[0])
                    tabs[tabname] = hstack([tabs0[tabname],tabs1[tabname]])

            ## Create hdulist to export
            out_hdus = [fits.PrimaryHDU(header=self.coarse_calibrations[cc_filnum].header)]
            for out_name,in_name in zip(['calib coefs', 'fit variances', 'wavelengths', 'pixels'],\
                                ['coefs', 'metric', 'clines', 'pixels']):
                outtab = tabs[in_name]
                outtab = outtab[self.instrument.full_fibs[self.camera].tolist()]
                out_hdus.append(fits.BinTableHDU(data=outtab.copy(), name=out_name))
                if in_name == 'coefs':
                    self.coarse_calibration_coefs[pairnum] = outtab.copy()

            hdulist = fits.HDUList(out_hdus)

                # out_calib_table = out_calib_table[np.sort(out_calib_table.colnames)]


            self.filemanager.save_full_calib_dict(hdulist, self.lampstr_c, self.camera, self.config, filenum=cc_filnum)

            histories = tabs['coefs']

    def generate_evolution_tables(self):
        for pairnum,filnums in self.pairings.items():
            if pairnum == 0:
                continue

            current_calibs = Table(self.coarse_calibration_coefs[pairnum].copy())
            past_calibs = Table(self.coarse_calibration_coefs[pairnum-1].copy())

            for column in current_calibs.colnames:
                current_calibs[column] = current_calibs[column] - past_calibs[column]

            self.evolution_in_coarse_coefs[pairnum] = current_calibs.copy()

    def run_final_calibrations(self,initial_priors='parametric'):
        self.generate_evolution_tables()
        output_names = ['calib coefs','fit variances','wavelengths','pixels']
        all_output_names = ['calib coefs', 'fit variances', 'wavelengths', 'pixels', 'linelist']
        mock_spec_w, mock_spec_f = self.mock_spec_w, self.mock_spec_f

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
            if pairnum > 0:
                coarse_table_differences = self.evolution_in_coarse_coefs[pairnum]
                for column in coarse_table_differences.colnames:
                    initial_coef_table[column] = initial_coef_table[column] + coarse_table_differences[column]
            ## HACK!!
            if pairnum == 0 and self.camera=='r':
                continue
            ## END HACK!
            filenum = filnums[self.filenum_ind]
            data = Table(self.fine_calibrations[filenum].data)

            linelist = self.selected_lines

            effective_iteration = pairnum#np.max([pairnum,int(using_defaults)])
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
                    # specific_set = [cam + '101', cam + '416']
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

            # hand_fit_subset = np.asarray(hand_fit_subset)

            ##HACK!
            # if pairnum == 0 and self.camera=='r':
            #     altered_coef_table = initial_coef_table.copy()
            #     hand_fit_subset = np.asarray(['r101','r816','r416','r501','r210','r602','r715'])
            #     altered_coef_table = {}#initial_coef_table.copy()
            #     altered_coef_table['r101'] = [5071.8187300612035, 0.9930979838081959, -5.769775729541421e-06,
            #                                   1.6219475654346627e-08, -1.060536238512127e-11, 2.027614894968671e-15]
            #
            #     altered_coef_table['r816'] = [5064.941399949152, 0.9887048293667995, 4.829092351762018e-06,
            #                                   5.280389577236655e-09, -5.618906483279477e-12, 1.1981097537960155e-15]
            #
            #     altered_coef_table['r416'] = [4966.43139830805, 0.9939388787553181, 5.244911711992524e-06,
            #                                   1.2291548669411035e-09, - 2.0296595329597448e-12, 2.9050877132565224e-16]
            #
            #     altered_coef_table['r501'] = [4965.341783218052, 0.9873531089008049, 2.4560812264245633e-05,
            #                                   -2.0293237635901715e-08, 8.081202360788054e-12, -1.397383927434781e-15]
            #
            #     altered_coef_table['r210'] = [5009.879532180203, 0.986418938077269,
            #                                   2.1117286784979934e-05, - 1.612921025968839e-08, 6.307242237439978e-12,
            #                                   -1.175841190977326e-15]
            #
            #     altered_coef_table['r309'] = [4981.847585300046, 0.9953409249278389, 6.616819915490353e-09,
            #                                   7.072942793437885e-09, -4.7799815890757634e-12, 7.369734622022845e-16]
            #
            #     altered_coef_table['r602'] = [4975.080088016758, 0.9916173886456268, 7.811003804278236e-06,
            #                                   1.1977785560589788e-09, -3.3762927213375386e-12, 7.593041888780153e-16]
            #
            #     altered_coef_table['r715'] = [5014.023681360571, 0.99147302071155, 4.748885129798807e-06,
            #                                   3.1454713162197196e-09, -3.4683774647827705e-12, 6.101876288746191e-16]
            #     handfit_fitting_dict = {}
            #     handfit_fitting_dict['calib coefs'] = altered_coef_table
            #     wm,fm = linelist['ThAr']
            # else:
            #     if pairnum==1 and self.camera=='r':
            #         # altered_coef_table,thetype = self.filemanager.locate_calib_dict(fittype='full-ThAr', camera=self.camera,
            #         #                                                  config=self.config, filenum=filenum)
            #         # print(thetype, altered_coef_table)
            #         altered_coef_table = self.filemanager.load_calib_dict(fittype='full-ThAr',cam=self.camera,config=self.config,filenum=1490,timestamp=679621)
            #         initial_coef_table = Table(altered_coef_table['CALIB COEFS'].data)
            ## End HACK!


            handfit_fitting_dict, wm, fm  = \
                                    wavelength_fitting_by_line_selection(data, initial_coef_table,\
                                    self.all_lines, linelist, self.mock_spec_w, self.mock_spec_f ,\
                                    select_lines=select_lines,save_plots=self.save_plots,savetemplate_funcs=self.savetemplate_funcs,\
                                    filenum=filenum,subset=hand_fit_subset,completed_coefs={})


            if select_lines:
                linelistdict = {'ThAr': (wm, fm)}
            else:
                linelistdict = self.selected_lines

            if self.single_core:
                full_fitting_dict, badfits = \
                    auto_wavelength_fitting_by_lines(data, initial_coef_table, handfit_fitting_dict['calib coefs'].copy(), self.all_lines, linelistdict.copy(),\
                                                          mock_spec_w=mock_spec_w,  mock_spec_f=mock_spec_f,\
                                                          filenum=filenum, \
                                                          save_plots=self.save_plots, savetemplate_funcs=self.savetemplate_funcs)

                # for datainfoname,datainfo in handfit_fitting_dict.items():
                #     for fib in datainfo.keys():
                #         full_fitting_dict[datainfoname][fib] = datainfo[fib]

                badfits = np.array(badfits)
            else:
                fib1s = self.instrument.lower_half_fibs[self.camera]
                fib2s = self.instrument.upper_half_fibs[self.camera]

                obs1 = {
                    'comp': data[fib1s.tolist()], 'fulllinelist': self.all_lines.copy(),
                    'coarse_coefs': initial_coef_table, 'linelistdict':linelistdict.copy(), \
                    'mock_spec_w':mock_spec_w.copy(), 'mock_spec_f': mock_spec_f.copy(), \
                    'out_coefs':handfit_fitting_dict['calib coefs'].copy(),'filenum':filenum,
                    'save_plots':self.save_plots, "savetemplate_funcs":self.savetemplate_funcs
                }
                obs2 = {
                    'comp': data[fib2s.tolist()], 'fulllinelist': self.all_lines.copy(),
                    'coarse_coefs': initial_coef_table.copy(), 'linelistdict':linelistdict.copy(), \
                    'mock_spec_w':mock_spec_w.copy(), 'mock_spec_f': mock_spec_f.copy(), \
                    'out_coefs':handfit_fitting_dict['calib coefs'].copy(),'filenum':filenum,
                    'save_plots': self.save_plots, "savetemplate_funcs": self.savetemplate_funcs
                }

                all_obs = [obs1, obs2]
                NPROC = np.clip(len(all_obs), 1, 4)

                with Pool(NPROC) as pool:
                    tabs = pool.map(auto_wavelength_fitting_by_lines_wrapper, all_obs)

                full_fitting_dict,badfits = tabs[0]
                full_fitting_dict2,badfits2 = tabs[1]

                # ## The hand fit calibrations are in both returned dicts, remove from the second
                # ## Assign the other calibration info from hand fits to the output dicts
                # for datainfoname, datainfo in handfit_fitting_dict.items():
                #     ## use the autofitted wavelength solution even for hand fits, note we're not
                #     ## assigning these values to the output array
                #     if 'coef' in datainfoname:
                #         for fib in datainfo.keys():
                #             full_fitting_dict2[datainfoname].pop(fib)
                #     else:
                #         for fib in datainfo.keys():
                #             full_fitting_dict[datainfoname][fib] = datainfo[fib]

                ## The hand fit calibrations are in both returned dicts, remove from the second
                ## Assign the other calibration info from hand fits to the output dicts
                for datainfoname, datainfo in full_fitting_dict2.items():
                    for fib in datainfo.keys():
                        full_fitting_dict[datainfoname][fib] = datainfo[fib]

                badfits = np.unique(np.append(badfits,badfits2))

            handfit_bad_subset_dict, wm, fm = \
                                                wavelength_fitting_by_line_selection(data, initial_coef_table, \
                                                     self.all_lines, linelistdict, self.mock_spec_w, self.mock_spec_f, \
                                                     select_lines=select_lines, save_plots=self.save_plots,
                                                     savetemplate_funcs=self.savetemplate_funcs, \
                                                     filenum=filenum, subset=badfits,
                                                     completed_coefs=full_fitting_dict['calib coefs'].copy())
            for datainfoname, datainfo in handfit_bad_subset_dict.items():
                for fib in datainfo.keys():
                    full_fitting_dict[datainfoname][fib] = datainfo[fib]

            if select_lines:
                self.selected_lines = full_fitting_dict['linelist'].copy()
                select_lines = False

            ## Zero pad rows so that the table won't throw an error for unequal sizes
            maxlams = int(np.max([len(full_fitting_dict['wavelengths'][fib]) for fib in full_fitting_dict['wavelengths'].keys()]))

            for fib in full_fitting_dict['wavelengths'].keys():
                nlams = len(full_fitting_dict['wavelengths'][fib])
                if nlams!=maxlams:
                    full_fitting_dict['wavelengths'][fib] = np.append(full_fitting_dict['wavelengths'][fib],np.zeros(maxlams-nlams))
                    full_fitting_dict['pixels'][fib] = np.append(full_fitting_dict['pixels'][fib], np.zeros(maxlams - nlams))

            ## Create hdulist to export
            out_hdus = [fits.PrimaryHDU(header=self.fine_calibrations[filenum].header)]
            for out_name in output_names:
                curtab = Table()
                curdict = full_fitting_dict[out_name]
                for key in self.instrument.full_fibs[self.camera]:
                    curtab.add_column(Table.Column(data=curdict[key],name=key))

                out_hdus.append(fits.BinTableHDU(data= curtab.copy(),name=out_name))

            hdulist = fits.HDUList(out_hdus)

            #out_calib_table = out_calib_table[np.sort(out_calib_table.colnames)]
            self.fine_calibration_coefs[pairnum] = full_fitting_dict['calib coefs'].copy()

            if pairnum > 0:
                devs = find_devs(initial_coef_table,full_fitting_dict['calib coefs'])

            initial_coef_table = Table(full_fitting_dict['calib coefs'].copy())

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
                coefs[ii] = pix_to_wave_explicit_coefs2(yval,yparametrized_coefs[0,ii],yparametrized_coefs[1,ii],yparametrized_coefs[2,ii])
            out_dict[fiber] = coefs

        offsets = np.array(coef_xys[0]['y'])
        min_off = np.min(offsets)
        shifted_offsets = offsets - min_off
        linears = np.array(coef_xys[1]['y'])
        shifted_linears = linears - 1
        quads = np.array(coef_xys[2]['y'])
        xs = np.array(coef_xys[1]['x'])

        fitted_xs = np.arange(2056)
        fitted_offsets = pix_to_wave_explicit_coefs2(fitted_xs, *yparametrized_coefs[:, 0])
        shifted_fitted_offsets = fitted_offsets - np.min(fitted_offsets)
        fitted_linears = pix_to_wave_explicit_coefs2(fitted_xs, *yparametrized_coefs[:, 1])
        shifted_fitted_linears = fitted_linears - 1
        fitted_quads = pix_to_wave_explicit_coefs2(fitted_xs, *yparametrized_coefs[:, 2])

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








