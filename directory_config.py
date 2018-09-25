import os
from astropy.io import fits
#from quickreduce_funcs import print_data_neatly
import numpy as np
from astropy.table import Table

from io import FileManager
from instrument import InstrumentState



from wavelength_calibration import wavelength_fitting_by_line_selection, run_interactive_slider_calibration

class Calibrations:
    def __init__(self, camera, lamptypes1, lamptypes2, first_calibrations, filemanager, config, \
                 second_calibrations=None, pairings=None, load_history=True, trust_after_first=False):
        from calibrations import load_calibration_lines_salt_dict as load_calibration
        self.imtype = 'comp'

        self.camera = camera
        self.filemanager = filemanager
        self.config = config
        self.lamptypes1 = lamptypes1
        self.lamptypes2 = lamptypes2
        self.trust_after_first = trust_after_first

        self.linelist1 = load_calibration(lamptypes1)
        self.selected_lines1 = self.linelist1.copy()
        self.calib1_filnums = np.array(list(first_calibrations.keys())).astype(int)
        self.calib1_hdus = np.array(list(first_calibrations.values())).astype(int)

        self.ncalibs = len(self.calib1_filnums)
        self.do_secondary_calib = (second_calibrations is not None)

        self.lampstr_1 = ''
        self.lampstr_2 = ''
        for lamp in lamptypes1:
            self.lampstr_1 += '-'+str(lamp)

        if self.do_secondary_calib:
            self.linelist2 = load_calibration(lamptypes2)
            self.selected_lines2 = self.linelist2.copy()
            self.calib2_filnums = np.array(list(second_calibrations.keys())).astype(int)
            self.calib2_hdus = np.array(list(second_calibrations.keys())).astype(int)
            for lamp in lamptypes2:
                self.lampstr_2 += '-' + str(lamp)
        else:
            self.calib2_filnums = np.array([None]*self.ncalibs)
            self.calib2_hdus = self.calib2_filnums

        self.calib1_pairlookup = {}
        self.calib2_pairlookup = {}
        from collections import OrderedDict
        if pairings is None:
            self.pairings = OrderedDict()
            for ii, c1_filnum, c2_filnum in enumerate(zip(self.calib1_filenums,self.calib2_filenums)):
                self.pairings[ii] = (c1_filnum, c2_filnum)
                self.calib1_pairlookup[c1_filnum] = ii
                self.calib2_pairlookup[c2_filnum] = ii
        else:
            self.pairings = pairings
            for pairnum,c1_filnum,c2_filnum in pairings.items():
                self.calib1_pairlookup[c1_filnum] = pairnum
                self.calib2_pairlookup[c2_filnum] = pairnum

        self.pairnums = np.sort(list(self.pairings.keys()))

        self.history_calibration_coefs = {}
        self.default_calibration_coefs = None
        self.first_calibration_coefs = {}
        self.second_calibration_coefs = {}

        self.load_default_coefs()
        if load_history:
            self.load_most_recent_coefs()

    def load_default_coefs(self):
        from wavelength_calibration import aperature_number_pixoffset
        self.default_calibration_coefs = self.filemanager.self.load_calib_dict('default', self.camera, self.config)
        if self.default_calibration_coefs is None:
            outdict = {}
            fibernames = self.calib1_hdus[0].data.colnames
            adef, bdef, cdef = (4523.4, 1.0007, -1.6e-6)
            for fibname in fibernames:
                adef += aperature_number_pixoffset(fibname,self.camera)
                outdict[fibname] = (adef, bdef, cdef)
            self.default_calibration_coefs = outdict

    def load_most_recent_coefs(self):
        couldntfind = False
        if self.do_secondary_calib:
            for pairnum, c1_filnum, c2_filnum in self.pairings.items():
                name = self.imtype+self.lampstr_2
                calibs = self.filemanager.locate_calib_dict(name, self.camera, self.config,c2_filnum)
                if calibs is None:
                    couldntfind = True
                    break
                else:
                    self.history_calibration_coefs[pairnum] = calibs
        if couldntfind or not self.do_secondary_calib:
            for pairnum, c1_filnum, c2_filnum in self.pairings.items():
                name = self.imtype+self.lampstr_1
                calibs = self.filemanager.locate_calib_dict(name, self.camera, self.config,c1_filnum)
                self.history_calibration_coefs[pairnum] = calibs

    def run_initial_calibrations(self):
        trust = False
        defaults = self.default_calibration_coefs

        for pairnum,c1_filnum, throwaway in self.pairings.items():
            histories = self.history_calibration_coefs[pairnum]
            comp_data = self.calib1_hdus[c1_filnum].data
            out_calib = self.run_interactive_slider_calibration(comp_data, self.linelist1, default=defaults, \
                                                                histories=histories, trust_initial=trust)#, \
                                               #  steps=None, default_key=None)
            defaults = out_calib
            trust = self.trust_after_first
            self.first_calibration_coefs[pairnum] = out_calib.copy()

    def run_final_calibrations(self):
        if not self.do_secondary_calib:
            print("There doesn't seem to be a second calibration defined. Using the supplied calib1's")
        select_lines = True
        for pairnum,c1_filnum,c2_filnum in self.pairings.items():
            if self.do_secondary_calib:
                data = self.calib2_hdus[c2_filnum].data
                linelist = self.selected_lines2
            else:
                data = self.calib1_hdus[c1_filnum].data
                linelist = self.selected_lines1

            initial_coef_table = self.first_calibration_coefs[pairnum]

            out_calib, covariances, out_linelist = self.wavelength_fitting_by_line_selection(data, linelist, initial_coef_table,select_lines=select_lines)#bounds=None)
            if select_lines:
                if self.do_secondary_calib:
                    self.selected_lines2 = out_linelist
                else:
                    self.selected_lines1 = out_linelist
                select_lines = False

            self.second_calibration_coefs[pairnum] = out_calib

    def create_calibration_default(self):
        npairs = len(self.pairnums)
        default_outtable = self.second_calibration_coefs[self.pairnums[0]]
        if npairs > 1:
            for pairnum in self.pairnums[1:]:
                curtable = self.second_calibration_coefs[pairnum]
                for fiber in curtable.colnames:
                    default_outtable[fiber] += curtable[fiber]

            for fiber in curtable.colnames:
                default_outtable[fiber] /= npairs

        self.filemanager.self.save_calib_dict(default_outtable, 'default', self.camera, self.config)

    def save_initial_calibrations(self):
        for pairnum,table in self.first_calibration_coefs.items():
            filenum = self.pairings[pairnum][0]
            self.filemanager.save_calib_dict(table, self.lampstr_1, self.camera, self.config, filenum=filenum)

    def save_final_calibrations(self):
        for pairnum,table in self.second_calibration_coefs.items():
            if self.do_secondary_calib:
                filenum = self.pairings[pairnum][1]
            else:
                filenum = self.pairings[pairnum][0]
            self.filemanager.save_calib_dict(table, self.lampstr_1, self.camera, self.config, filenum=filenum)


class Observation:
    def __init__(self,sciences,flats, first_calibrations, second_calibrations=None):
        from calibrations import load_calibration_lines_salt_dict as load_calibration

        self.camera = camera
        self.calib1_lamplist = lamplist1
        self.calib2_lamplist = lamplist2

        self.science = science_hdus
        self.first_calibs = first_calibration_hdu
        self.second_calibs = second_calibration_hdu
        self.flats = flat_hdus
        self.first_calibration_coefs = None
        self.second_calibration_coefs = None

    def load_default_coefs(self):

    def load_most_recent_coefs(self):

    def run_first_calibration(self):

    def run_second_calibration(self):
        if self.second_calibs is None:
            print("There doesn't seem to be a second calibration defined. Skipping this step and returning")
        else:
    def wavelength_calibrate_science(self):
        if self.first_calibration_coefs is None:
            print("You must run a calibration or load saved coefs before calibrating science data. Skipping and returning")
        else:

        first_wavelength_calibration()
        if self.second_calibs is not None:
            second_wavelength_calibrations()

    def flatten_science(self):




class FieldData:
    def __init__(self, science_filenums, bias_filenums, flat_filenums, fibermap_filenums, \
                 first_comp_filenums, second_comp_filenums=None,
                 filemanager=FileManager(), instrument=InstrumentState(), startstep=None,\
                 obs_pairing_strategy='nearest',calib_lamps1=list(),calib_lamps2=None,\
                 twod_to_oned_strategy='simple',debias_strategy='median'):

        if obs_pairing_strategy not in ['nearest','user']: # 'unique',
            # 'nearest' chooses closest filenumber to sci
            # 'unique' pairs comps with science as uniquely as possible
            #             while it's second priority is to pair closest
            #             NOTE YET IMPLEMENTED
            # 'user'  pairs in the exact order given in the filenum lists
            self.obs_pairing_strategy = 'nearest'
        else:
            self.obs_pairing_strategy = obs_pairing_strategy

        if twod_to_oned_strategy != 'simple':
            print("The only implemented 2D to 1D strategy is the simple summation. Defaulting to that.")
            self.twod_to_oned = twod_to_oned_strategy
        else:
            self.twod_to_oned = twod_to_oned_strategy

        if debias_strategy != 'median':
            print("Only median debias strategy is currently implemented. Defaulting to that.")
            self.debias_strategy = 'median'
        else:
            self.debias_strategy = debias_strategy

        self.twostep_wavecalib = (second_comp_filenums is not None)
        self.filemanager=filemanager
        self.instrument=instrument
        self.caliblamps1 = calib_lamps1
        self.caliblamps2 = calib_lamps2

        self.filenumbers = {
            'bias': np.array(bias_filenums), 'second_calib': np.array(second_comp_filenums), \
            'first_calib': np.array(first_comp_filenums), 'flat': np.array(flat_filenums), \
            'science': np.array(science_filenums), 'fibmap': np.array(fibermap_filenums)
        }

        self.master_types = []
        self.observations = []

        self.data_stitched = False
        self.fibersplit = False
        self.current_data_saved = False
        self.current_data_from_disk = False
        # self.bias_subtracted = False
        # self.cosmics_removed = False
        # self.wavelengths_calibrated = False
        # self.flattened = False
        # self.observations_combined = False

        self.reduction_order = {None: 0, 'raw': 0, 'stitch': 1, 'bias': 2, 'remove_crs': 3,   \
                                'apcut': 4, 'wavecalib': 5,'flat': 6, 'skysub': 7, 'combine': 8,\
                                'zfit': 9}

        self.step = None
        self.update_step(startstep)

        ## Dictionary of hdus
        ## Key is tuple: (camera,filnum,imtype,opamp)
        ## Value is the hdu for that tuple of information
        ## an aastropy.fits hdu with a header and data attribute
        if self.step is None:
            self.all_hdus = {}
        else:
            self.all_hdus = self.read_all_filedata()


    def update_step(self,step):
        self.step = step
        numeric_step_value = self.reduction_order[self.step]
        self.data_stitched = numeric_step_value>1
        # self.bias_subtracted = numeric_step_value>2
        # self.cosmics_removed = numeric_step_value>3
        self.fibersplit    = numeric_step_value>4
        # self.wavelengths_calibrated = numeric_step_value>5
        # self.flattened = numeric_step_value>6
        # self.observations_combined = numeric_step_value>7

        if numeric_step_value > 1:
            self.opamps = None
        if numeric_step_value > 2:
            if 'bias' in self.filenumbers.keys():
                self.filenumbers.pop('bias')
            if 'bias' not in self.master_types:
                self.master_types.append('bias')
        if numeric_step_value > 6:
            if 'flat' in self.filenumbers.keys():
                self.filenumbers.pop('flat')
            if 'flat' not in self.master_types:
                self.master_types.append('flat')
                
        self.filemanager.update_templates_for(step)

    def proceed_to(self,step):
        if step != self.step:
            self.update(step)

    def read_crashed_filedata(self,returndata=False):
        import pickle as pkl
        data_product_loc = self.filemanager.directory.data_product_loc
        infile = os.path.join(data_product_loc, '_precrashdata.pkl')
        with open(infile,'rb') as crashdata:
            self.all_hdus = pkl.load(crashdata)
        if returndata:
            return self.all_hdus

    def read_all_filedata(self,step=None,return_data=False):
        if step is None and self.step is None:
            raise(IOError,"No step in reduction specified. I don't know what data to read")
        elif step is None:
            step = self.step
        elif self.step != step:
            self.step = step
            self.update_step(step)

        if self.data_stitched:
            opamps = [None]
        else:
            opamps = self.instrument.opamps

        cameras = self.instrument.cameras
        all_hdus = {}
        for imtype, filenums in self.filenumbers.items():
            for camera in cameras:
                for filnum in filenums:
                    for opamp in opamps:
                        all_hdus[(camera,filnum,imtype,opamp)] = self.filemanager.read_hdu(camera=camera, filenum=filnum, imtype=imtype, amp=opamp, fibersplit=self.fibersplit)


        for imtype in self.master_types:
            for camera in cameras:
                all_hdus[(camera, 'master', imtype, None)] = self.filemanager.read_hdu(camera=camera, filenum='master', imtype=imtype, fibersplit=self.fibersplit)

        self.all_hdus = all_hdus

        self.assign_observational_sets()

        self.current_data_saved = True
        self.current_data_from_disk = True

        if return_data:
            return all_hdus

    def write_all_filedata(self):
        if self.data_stitched:
            opamps = [None]
        else:
            opamps = self.instrument.opamps

        cameras = self.instrument.cameras

        for imtype, filenums in self.filenumbers.items():
            for camera in cameras:
                for filnum in filenums:
                    for opamp in opamps:
                        outhdu = self.all_hdus[(camera,filnum,imtype,opamp)]
                        self.filemanager.write_hdu(outhdu=outhdu,camera=camera, filenum=filnum, imtype=imtype, amp=opamp)

        for imtype in self.master_types:
            for camera in cameras:
                outhdu = self.all_hdus[(camera, 'master', imtype, None)]
                self.filemanager.write_hdu(outhdu=outhdu,camera=camera, filenum='master', imtype=imtype)

        self.current_data_saved = True

    def check_data_ready_for_current_step(self):
        numeric_step = self.reduction_order[self.step]
        if numeric_step > 1: # after stitching
            if self.opamps != None:
                return False
        if numeric_step > 2: # after bias sub
            if 'bias' not in self.master_types:
                return False
            elif 'bias' in self.filenumbers.keys():
                return False
        if numeric_step == 4: # for app cutting
            if (self.cameras[0],'master','fibmap',None) not in self.all_hdus.keys():
                return False
        if numeric_step > 4: # after app cutting
            if len(self.observations) == 0:
                return False
        if numeric_step > 5: # after wavelength calibration
            if self.observations[0].calibration_coefs is None:
                return False
        if numeric_step > 6: # after flattening
            if 'flat' not in self.master_types:
                return False
            elif 'flat' in self.filenumbers.keys():
                return False

        return True

    def assign_observational_sets(self):
        """
        # 'nearest' chooses closest filenumber to sci
        # 'unique' pairs comps with science as uniquely as possible
        #             while it's second priority is to pair closest
        # 'user'  pairs in the exact order given in the filenum list
        :return: None
        """
        if self.twostep_wavecalib:
            calib1s = self.filenumbers['first_calib'].astype(int)
            calib2s = self.filenumbers['second_calib'].astype(int)
        else:
            calib1s = self.filenumbers['first_calib'].astype(int)
            calib2s = np.array([None]*len(calib1s))

        flats = self.filenumbers['flat'].astype(int)
        scis = self.filenumbers['science'].astype(int)

        if self.obs_pairing_strategy == 'unique':
            print("Unique observation pairing isn't yet implemented, defaulting to 'closest'")
            self.obs_pairing_strategy = 'closest'

        if self.obs_pairing_strategy == 'user':
            for sci,flat,calib1,calib2 in zip(scis,flats,calib1s,calib2s):
                self.observations.append(Observation(sciences=sci,flats=flat,\
                                                     first_calibrations=calib1,\
                                                     second_calibrations=calib2))
        if self.obs_pairing_strategy == 'closest':
            for sci in scis:
                nearest_flat = flats[np.argmin(np.abs(flats-sci))]
                nearest_calib1 = calib1s[np.argmin(np.abs(calib1s - sci))]
                if self.twostep_wavecalib:
                    nearest_calib1 = calib2s[np.argmin(np.abs(calib2s - sci))]
                else:
                    nearest_calib2 = None
                self.observations.append(Observation(sciences=sci,flats=nearest_flat,\
                                                     first_calibrations=nearest_calib1,\
                                                     second_calibrations=nearest_calib2))


    def run_step(self,step=None):
        if step is not None:
            self.proceed_to(step)

        if self.step not in self.reduction_order.keys():
            print("Step \"{}\" doesn't have an action assosciated with it".format(self.step))
        if step == 'stitch':
            from stitch import stitch_all_images
            self.all_hdus = stitch_all_images(self.all_hdus,self.filemanager.date_timestamp)
            self.opamps = None
        elif step == 'bias':
            from bias_subtract import bias_subtract
            self.all_hdus = bias_subtract(self.all_hdus,self.filemanager.date_timestamp,strategy=self.debias_strategy)
        elif step == 'remove_crs':
            self.write_all_filedata()
            import PyCosmic
            for (camera, filenum, imtype, opamp) in self.all_hdus.keys():
                readfile = self.filemanager.get_read_filename(camera=camera, imtype=imtype,\
                                                        filenum=filenum, amp=opamp)
                writefile = self.filemanager.get_write_filename(camera=camera, imtype=imtype,\
                                                        filenum=filenum, amp=opamp)
                maskfile = writefile.replace('.fits', '.crmask.fits')
                print("\nFor image type: {}, shoe: {},   filenum: {}".format(imtype, camera, filenum))
                outdat, pycosmask, pyheader = PyCosmic.detCos(readfile, maskfile, writefile, rdnoise='ENOISE',
                                                              sigma_det=8, gain='EGAIN', verbose=True, return_data=False)
            self.read_all_filedata()
        elif step == 'apcut':
            from apperature_cut import cutout_all_apperatures
            self.all_hdus = cutout_all_apperatures(self.all_hdus,self.instrument.cameras,\
                                                   deadfibers=self.instrument.deadfibers,summation_preference=self.twod_to_oned)
        elif step == 'wavecalib':
            for ii in range(self.calibrations.ncalibs):
                self.calibrations.calibrate_interactive('first',ii)
                self.calibrations.calibrate_line_selection('second',ii)
            default_vals = self.observations.get_default_calib()
            for observation in self.observations:
                observation.update_default(default_vals)
                observation.calibrate_interactive('first',returndata=False)
                default_vals = observation.calibrate_line_selection('second',returndata=True)
                observation.calibrate_flats()
                observation.calibrate_sciences()
        elif step == 'flat':
            for observation in self.observations:
                observation.flatten_sciences()
        elif step == 'skysub':
            for observation in self.observations:
                observation.subtract_skies()
        elif step == 'combine':
            sciences = self.observations[0].science_hdus
            if len(self.observations) > 1:
                for observation in self.observations[1:]:
                    sciences.extend(observation.science_hdus)
            self.combine_sciences(sciences)
        elif step == 'zfit':

        if step != 'remove_crs':
            self.current_data_saved = False
            self.current_data_from_disk = False



    def wavelength_calibrate_all_science(self):

    def flatten_all_science(self):

    def combine_science_observations(self):

    def fit_redshfits(self):


class ReductionController:
    def __init__(self,fielddata,filemanager,instrumentsetup):
        self.field_data = fielddata
        self.file_manager = filemanager
        self.instrument_setup = instrumentsetup

        data.load(step='bias')
        data.run_step(step='bias')
        data.save_data()
        data.proceed_to(step='remove_crs')
        data.check_data_consistent_for_current_step(step='remove_crs')
        data.run_step('remove_crs')
        data.save_data()
        data.proceed_to(step='sddsf')
        ...




