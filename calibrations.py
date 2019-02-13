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
from multiprocessing import Pool
from astropy.table import hstack
from wavelength_calibration import compare_outputs,automated_calib_wrapper_script
import matplotlib.pyplot as plt

class Calibrations:
    from wavelength_calibration import wavelength_fitting_by_line_selection, run_interactive_slider_calibration

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
        # self.calibc_filnums,self.calibc_hdus = [],[]
        # for key,val in coarse_calibrations.items():
        #     self.calibc_filnums.append(int(key))
        #     self.calibc_hdus.append(Table(val.data))
        #
        # self.calibc_filnums = np.array(self.calibc_filnums)
        # self.calibc_hdus = np.array(self.calibc_hdus)

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
        from wavelength_calibration import aperature_number_pixoffset
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

    def load_final_calib_hdus(self):
        couldntfind = False
        if self.do_fine_calib:
            filnum_ind = c
        else:
            filnum_ind = 0
        for pairnum, filnums in self.pairings.items():
            filnum = filnums[filnum_ind]
            name = self.lampstr_f
            calib,thetype = self.filemanager.locate_calib_dict(name, self.camera, self.config,filnum,locate_type='full')
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
            raise(IOError,"Couldn't find matching calibrations. Please make sure the step has been run fully")


    def run_initial_calibrations(self):
        for pairnum,(cc_filnum, throwaway) in self.pairings.items():
            comp_data = Table(self.coarse_calibrations[cc_filnum].data)
            fibernames = np.sort(comp_data.colnames)
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
        for pairnum,filnums in self.pairings.items():
            filenum = filnums[filenum_ind]

            ## Note that if there isn't a fine calibration, fine_calibrations
            ## has already been set equal to coarse_calibrations hdus
            data = Table(self.fine_calibrations[filenum].data)
            linelist = self.selected_lines
            initial_coef_table = OrderedDict()
            if pairnum == 0:
                initial_coef_table = self.coarse_calibration_coefs[pairnum]
            else:
                last_iteration_coefs = self.fine_calibration_coefs[pairnum-1]
                evolution = self.evolution_in_coarse_coefs[pairnum]
                for fiber in last_iteration_coefs.columns:
                    colvals = last_iteration_coefs[fiber]
                    initial_coef_table[fiber] = colvals + evolution[fiber]

            out_calib, out_linelist, lambdas, pixels, variances  = self.wavelength_fitting_by_line_selection(data, linelist, self.all_lines, initial_coef_table,select_lines=select_lines)#bounds=None)
            if select_lines:
                self.selected_lines = out_linelist

            self.fine_calibration_coefs[pairnum] = out_calib

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