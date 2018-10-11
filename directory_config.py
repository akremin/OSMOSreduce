import os
from astropy.io import fits
#from quickreduce_funcs import print_data_neatly
import numpy as np
from astropy.table import Table

from io import FileManager
from instrument import InstrumentState
from calibrations_class import Calibrations
from collections import OrderedDict
from scipy.interpolate import CubicSpline

class Observations:
    def __init__(self,filenumbers):
        self.calib1s = filenumbers['first_calib']
        if filenumbers['second_calib'] is None:
            self.calib2s = np.array([None]*len(self.calib1s))
        elif len(filenumbers['second_calib']) == 1 and filenumbers['second_calib'][0] is None:
            self.calib2s = np.array([None] * len(self.calib1s))
        else:
            self.calib2s = filenumbers['second_calib']

        self.flats = filenumbers['flat']
        self.scis = filenumbers['science']

        self.two_step_calib = (self.calib2s[0] is not None)

        self.observations = OrderedDict()
        self.calibration_pairs = OrderedDict()

        self.assign_observational_sets()

    def assign_observational_sets(self):
        """
        # 'nearest' chooses closest filenumber to sci
        # 'unique' pairs comps with science as uniquely as possible
        #             while it's second priority is to pair closest
        # 'user'  pairs in the exact order given in the filenum list
        :return: None
        """
        calib1s = self.calib1s.astype(int)
        if self.two_step_calib:
            calib2s = self.calib2s.astype(int)
        else:
            calib2s = self.calibs2s

        flats = self.flats
        scis = self.scis

        calibration_pairs = OrderedDict()
        observation_sets = OrderedDict()

        if self.obs_pairing_strategy == 'unique':
            print("Unique observation pairing isn't yet implemented, defaulting to 'closest'")
            self.obs_pairing_strategy = 'closest'

        if self.obs_pairing_strategy == 'user':
            for ii,sci,flat,calib1,calib2 in enumerate(zip(scis,flats,calib1s,calib2s)):
                calibration_pairs[ii] = (calib1,calib2)
                observation_sets[ii] = (sci,flat,calib1,calib2)

        if self.obs_pairing_strategy == 'closest':
            for ii,sci in enumerate(scis):
                nearest_flat = flats[np.argmin(np.abs(flats-sci))]
                nearest_calib1 = calib1s[np.argmin(np.abs(calib1s - sci))]
                if self.two_step_calib:
                    nearest_calib2 = calib2s[np.argmin(np.abs(calib2s - sci))]
                else:
                    nearest_calib2 = None
                calibration_pairs[ii] = (nearest_calib1,nearest_calib2)
                observation_sets[ii] = (sci,nearest_flat,nearest_calib1,nearest_calib2)

        self.observations = observation_sets
        self.calibration_pairs = calibration_pairs

    def return_calibration_pairs(self):
        return self.calibration_pairs

    def return_observation_sets(self):
        return self.observations



class FieldData:
    def __init__(self, science_filenums, bias_filenums, flat_filenums, fibermap_filenums, \
                 first_comp_filenums, second_comp_filenums=None,
                 filemanager=FileManager(), instrument=InstrumentState(), startstep=None,\
                 obs_pairing_strategy='nearest',calib_lamps1=list(),calib_lamps2=None,\
                 twod_to_oned_strategy='simple',debias_strategy='median'):

        self.obs_pairing_strategy = obs_pairing_strategy
        self.twod_to_oned = twod_to_oned_strategy
        self.debias_strategy = debias_strategy
        self.check_parameter_flags()

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

        self.observations = Observations(self.filenumbers)
        self.calibrations = {}


    def check_parameter_flags(self):
        if self.obs_pairing_strategy not in ['nearest','user']: # 'unique',
            # 'nearest' chooses closest filenumber to sci
            # 'unique' pairs comps with science as uniquely as possible
            #             while it's second priority is to pair closest
            #             NOTE YET IMPLEMENTED
            # 'user'  pairs in the exact order given in the filenum lists
            self.obs_pairing_strategy = 'nearest'

        if self.twod_to_oned_strategy != 'simple':
            print("The only implemented 2D to 1D strategy is the simple summation. Defaulting to that.")
            self.twod_to_oned = twod_to_oned_strategy

        if self.debias_strategy != 'median':
            print("Only median debias strategy is currently implemented. Defaulting to that.")
            self.debias_strategy = 'median'

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
        self.all_hdus = self.filemanager.read_crashed_filedata()
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

        self.all_hdus = self.filemanager.read_all_filedata(self.filenumbers,self.master_types,self.instrument,\
                                                           self.data_stitched,self.fibersplit)

        self.assign_observational_sets()

        self.current_data_saved = True
        self.current_data_from_disk = True

        if return_data:
            return self.all_hdus

    def write_all_filedata(self):
        self.filemanager.write_all_filedata(self, self.all_hdus, self.filenumbers, self.master_types,\
                                            self.instrument, self.data_stitched)

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
            for camera in self.instrument.cameras:
                self.calibrations[camera].run_initial_calibrations()

            for camera in self.instrument.cameras:
                self.calibrations[camera].run_final_calibrations()

        elif step == 'flat':
            for camera in self.instrument.cameras:
                self.flatten_sciences(cam=camera)
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


    def populate_calibrations(self):
        calibration_pairs = self.observations.return_calibration_pairs()
        self.calibrations = {}
        for camera in self.instrument.cameras:
            calib1s = {filnum:self.all_hdus[(camera,filnum,'comp',None)] for filnum in self.observations.calib1s}
            if self.observations.two_step_calib:
                calib2s = {filnum: self.all_hdus[(camera, filnum, 'comp', None)] for filnum in self.observations.calib2s}
            else:
                calib2s = None
            calib = Calibrations(camera, self.calib_lamps1, self.calib_lamps2, calib1s, self.filemanager, \
                                 config=self.instrument.configuration, second_calibrations=calib2s,
                                 pairings=calibration_pairs, load_history=True, trust_after_first=False)

            self.calibrations[camera] = calib

    def run_initial_calibrations(self):
        for camera in self.instrument.cameras:
            self.calibrations[camera].run_initial_calibrations()

    def run_final_calibrations(self):
        for camera in self.instrument.cameras:
            self.calibrations[camera].run_final_calibrations()

    def combine_flats(self,camera,return_table=False):
        filnums = self.filenumbers['flat']
        all_flats = []
        flat_header = None
        for filnum in filnums:
            hdu = self.all_hdus.pop([(camera, filnum, 'flat', None)])
            if filnum == filnums[0]:
                flat_header = hdu.header
            all_flats.append(hdu.data)
        flat_table_one = all_flats[0]
        for key in flat_table_one.colnames:
            for flat in all_flats[1:]:
                flat_table_one[key].data += flat[key].data

        self.all_hdus[(camera,'master','flat',None)] = fits.BinTableHDU(data=flat_table_one,header=flat_header)
        if return_table:
            return flat_table_one

    def flatten_sciences(self,cam):
        from flatten import flatten_data
        fiber_fluxes = self.combine_flats(camera = cam, return_table=True)
        calib_table = self.calibrations[cam].create_calibration_default(save=False)
        final_table, final_flux_array = flatten_data(fiber_fluxes=fiber_fluxes,waves=calib_table)

        for filnum in self.filenumbers['science']:
            hdu = self.all_hdus[(cam,filnum,'science',None)]
            for col in final_table.colnames:
                hdu.data[col] /= final_table[col]
            self.all_hdus[(cam,filnum,'science',None)] = hdu

    def combine_science_observations(self, cam):
        observation_keys = list(self.observations.observations.keys())
        nobs = len(observation_keys)
        middle_obs = observation_keys[nobs//2]

        ref_science_filnum,throw,throw1,throw2 = self.observations.observations[middle_obs]
        ref_sci_hdu = self.all_hdus.pop((cam,ref_science_filnum,'science',None))
        ref_sci_data = ref_sci_hdu.data
        ref_calibs = self.calibrations[cam].second_calibration_coefs[middle_obs]
        ref_wavearrays = {}
        pixels = np.arange(len(ref_sci_data.columns[0]))

        combined_data = ref_sci_data.copy()
        for col in ref_calibs.colnames:
            a,b,c,d,e,f = ref_calibs[col]
            lams = a + b+pixels + c*np.power(pixels,2) + d*np.power(pixels,3) + \
                   e * np.power(pixels, 4) + f * np.power(pixels, 5)
            ref_wavearrays[col] = lams
        for obs in observation_keys:
            if obs == middle_obs:
                continue
            sci_filnum, throw, throw1, throw2 = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))
            sci_data = sci_hdu.data
            sci_calibs = self.calibrations[cam].second_calibration_coefs[obs]
            for col in sci_calibs.colnames:
                a, b, c, d, e, f = sci_calibs[col]
                lams = a + b + pixels + c * np.power(pixels, 2) + d * np.power(pixels, 3) + \
                       e * np.power(pixels, 4) + f * np.power(pixels, 5)
                flux = sci_data[col]
                scifit = CubicSpline(x=lams,y=flux)
                outwave = ref_wavearrays[col]
                outflux = scifit(outwave)
                combined_data[col] += outflux

        self.all_hdus[(cam,'master','science',None)] = fits.BinTableHDU(data=combined_data,header=ref_sci_hdu.header)

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




