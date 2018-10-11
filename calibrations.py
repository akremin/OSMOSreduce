import os
from astropy.io import fits
#from quickreduce_funcs import print_data_neatly
import numpy as np
from astropy.table import Table

from inputoutput import FileManager
from instrument import InstrumentState
from collections import OrderedDict

import numpy as np
from astropy.table import Table


class Calibrations:
    from wavelength_calibration import wavelength_fitting_by_line_selection, run_interactive_slider_calibration

    def __init__(self, camera, lamptypes1, lamptypes2, first_calibrations, filemanager, config, \
                 second_calibrations=None, pairings=None, load_history=True, trust_after_first=False,\
                 default_fit_key='cross correlation',use_selected_calib_lines=False):

        self.imtype = 'comp'

        self.camera = camera
        self.filemanager = filemanager
        self.config = config
        self.lamptypes1 = lamptypes1
        self.lamptypes2 = lamptypes2
        self.trust_after_first = trust_after_first
        self.default_fit_key = default_fit_key

        #self.linelist1,selected_lines1,all_lines1 = filemanager.load_calibration_lines_dict(lamptypes1,use_selected=use_selected_calib_lines)
        self.linelist1,all_lines1 = filemanager.load_calibration_lines_dict(lamptypes1,use_selected=use_selected_calib_lines)

        self.load_history = load_history
        self.first_calibrations = first_calibrations
        # self.calib1_filnums,self.calib1_hdus = [],[]
        # for key,val in first_calibrations.items():
        #     self.calib1_filnums.append(int(key))
        #     self.calib1_hdus.append(Table(val.data))
        #
        # self.calib1_filnums = np.array(self.calib1_filnums)
        # self.calib1_hdus = np.array(self.calib1_hdus)

        self.ncalibs = len(first_calibrations.keys())
        self.do_secondary_calib = (second_calibrations is not None)

        self.lampstr_1 = 'basic'
        self.lampstr_2 = 'full'

        for lamp in lamptypes1:
            self.lampstr_1 += '-'+str(lamp)

        if self.do_secondary_calib:
            self.linelist2,self.all_lines = filemanager.load_calibration_lines_dict(lamptypes2,use_selected=use_selected_calib_lines)
            #self.linelist2,self.selected_lines,self.all_lines = filemanager.load_calibration_lines_dict(lamptypes2,use_selected=use_selected_calib_lines)
            self.second_calibrations = second_calibrations
            for lamp in lamptypes2:
                self.lampstr_2 += '-' + str(lamp)
        else:
            self.linelist2 = self.linelist1.copy()
            #self.selected_lines = selected_lines1
            self.all_lines = all_lines1
            self.lampstr_2 = self.lamstr_1.replace('basic','full')
            self.second_calibrations = first_calibrations

        self.selected_lines = self.linelist2.copy()
        self.calib1_pairlookup = {}
        self.calib2_pairlookup = {}
        if pairings is None:
            self.pairings = OrderedDict()
            for ii, c1_filnum, c2_filnum in enumerate(zip(self.calib1_filenums,self.calib2_filenums)):
                self.pairings[ii] = (c1_filnum, c2_filnum)
                self.calib1_pairlookup[c1_filnum] = ii
                self.calib2_pairlookup[c2_filnum] = ii
        else:
            self.pairings = pairings
            for pairnum,(c1_filnum,c2_filnum) in pairings.items():
                self.calib1_pairlookup[c1_filnum] = pairnum
                self.calib2_pairlookup[c2_filnum] = pairnum

        self.pairnums = np.sort(list(self.pairings.keys()))

        self.history_calibration_coefs = {ii:None for ii in self.pairings.keys()}
        self.default_calibration_coefs = None
        self.first_calibration_coefs = OrderedDict()
        self.second_calibration_coefs = OrderedDict()
        self.final_calibrated_hdulists = OrderedDict()
        self.evolution_in_first_coefs = OrderedDict()

        self.load_default_coefs()
        if load_history:
            self.load_most_recent_coefs()

    def load_default_coefs(self):
        from wavelength_calibration import aperature_number_pixoffset
        self.default_calibration_coefs = self.filemanager.load_calib_dict('default', self.camera, self.config)
        if self.default_calibration_coefs is None:
            outdict = {}
            fibernames = Table(self.first_calibrations[self.pairings[0][0]].data).colnames
            adef, bdef, cdef, ddef, edef, fdef = (4465.4, 0.9896, 1.932e-05, 0., 0., 0.)
            for fibname in fibernames:
                aoff = aperature_number_pixoffset(fibname,self.camera)
                outdict[fibname] = (adef+aoff, bdef, cdef, ddef, edef, fdef)
            self.default_calibration_coefs = outdict

    def load_most_recent_coefs(self):
        couldntfind = False
        if self.do_secondary_calib:
            for pairnum, (c1_filnum, c2_filnum) in self.pairings.items():
                name = self.imtype+self.lampstr_2
                calibs = self.filemanager.locate_calib_dict(name, self.camera, self.config,c2_filnum)
                if calibs is None:
                    couldntfind = True
                    break
                else:
                    self.history_calibration_coefs[pairnum] = calibs
        if couldntfind or not self.do_secondary_calib:
            for pairnum, (c1_filnum, c2_filnum) in self.pairings.items():
                name = self.imtype+self.lampstr_1
                calibs = self.filemanager.locate_calib_dict(name, self.camera, self.config,c1_filnum)
                self.history_calibration_coefs[pairnum] = calibs

    def run_initial_calibrations(self):
        import matplotlib.pyplot as plt
        for fiber in ['r101', 'r201', 'r301', 'r401', 'r501', 'r601', 'r701', 'r801']:  # comp.colnames:
            plt.figure()

            for pairnum, (c1_filnum, throwaway) in self.pairings.items():
                comp_data = self.first_calibrations[c1_filnum].data[fiber]
                comp_data[comp_data<10] = 10
                pix = np.arange(len(comp_data))
                plt.semilogy(pix,comp_data,'-')
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.tight_layout()
            plt.show()

        defaults = self.default_calibration_coefs
        default_fit = self.default_fit_key
        for pairnum,(c1_filnum, throwaway) in self.pairings.items():
            histories = self.history_calibration_coefs[pairnum]

            comp_data = Table(self.first_calibrations[c1_filnum].data)
            out_calib = self.run_interactive_slider_calibration(first_comp=comp_data, complinelistdict=self.linelist1, default_vals=defaults, \
                                                                history_vals=histories, trust_initial=self.trust_after_first, default_key=default_fit)#,  steps=None
            trust = self.trust_after_first
            self.first_calibration_coefs[pairnum] = out_calib.copy()
            out_evolution = OrderedDict()
            if pairnum == 0:
                for fiber,colvals in out_calib:
                    out_evolution[fiber] = 0.*colvals
            else:
                for fiber,colvals in out_calib:
                    out_evolution[fiber] = colvals-defaults[fiber]
            self.evolution_in_first_coefs[pairnum] = out_evolution

            self.filemanager.save_basic_calib_dict(out_calib, self.lampstr_1, self.camera, self.config, filenum=c1_filnum)
            defaults = out_calib
            default_fit = 'default'

    def run_final_calibrations(self):
        if not self.do_secondary_calib:
            print("There doesn't seem to be a second calibration defined. Using the supplied calib1's")
        select_lines = True
        if self.do_secondary_calib:
            filenum_ind = 1
        else:
            filenum_ind = 0
        for pairnum,filnums in self.pairings.items():
            filenum = filnums[filenum_ind]

            ## Note that if there isn't a secondary calibration, second_calibrations
            ## has already been set equal to first_calibrations hdus
            data = Table(self.second_calibrations[filenum].data)
            linelist = self.selected_lines
            initial_coef_table = OrderedDict()
            if pairnum == 0:
                initial_coef_table = self.first_calibration_coefs[pairnum]
            else:
                last_iteration_coefs = self.second_calibration_coefs[pairnum-1]
                evolution = self.evolution_in_first_coefs[pairnum]
                for fiber, colvals in last_iteration_coefs:
                    initial_coef_table[fiber] = colvals + evolution[fiber]

            out_calib, out_linelist, lambdas, pixels, variances  = self.wavelength_fitting_by_line_selection(data, linelist, self.all_lines, initial_coef_table,select_lines=select_lines)#bounds=None)
            if select_lines:
                self.selected_lines = out_linelist
                select_lines = False

            self.second_calibration_coefs[pairnum] = out_calib

            ## Create hdulist to export
            prim = fits.PrimaryHDU(header=self.second_calibrations[filenum].header)
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
            self.filemanager.save_full_calib_dict(hdulist, self.lampstr_2, self.camera, self.config, filenum=filenum)


    def create_calibration_default(self,save=True):
        npairs = len(self.pairnums)
        default_outtable = self.second_calibration_coefs[self.pairnums[0]]
        if npairs > 1:
            for pairnum in self.pairnums[1:]:
                curtable = self.second_calibration_coefs[pairnum]
                for fiber in curtable.colnames:
                    default_outtable[fiber] += curtable[fiber]

            for fiber in curtable.colnames:
                default_outtable[fiber] /= npairs
        if save:
            self.filemanager.save_basic_calib_dict(default_outtable, 'default', self.camera, self.config)
        else:
            return default_outtable

    def save_initial_calibrations(self):
        for pairnum,table in self.first_calibration_coefs.items():
            filenum = self.pairings[pairnum][0]
            self.filemanager.save_basic_calib_dict(table, self.lampstr_1, self.camera, self.config, filenum=filenum)

    def save_final_calibrations(self):
        for pairnum,outlist in self.final_calibrated_hdulists.items():
            if self.do_secondary_calib:
                filenum = self.pairings[pairnum][1]
            else:
                filenum = self.pairings[pairnum][0]
            self.filemanager.save_full_calib_dict(outlist, self.lampstr_2, self.camera, self.config, filenum=filenum)


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