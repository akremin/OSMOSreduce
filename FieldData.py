import os
from astropy.io import fits
#from quickreduce_funcs import print_data_neatly
import numpy as np
from astropy.table import Table

from inputoutput import FileManager
from instrument import InstrumentState
from calibrations import Calibrations
from collections import OrderedDict
from scipy.interpolate import CubicSpline

class Observations:
    def __init__(self,filenumbers,obs_pairing_strategy):
        self.calib1s = filenumbers['first_calib']
        if filenumbers['second_calib'] is None:
            self.calib2s = np.array([None]*len(self.calib1s))
        elif len(filenumbers['second_calib']) == 1 and filenumbers['second_calib'][0] is None:
            self.calib2s = np.array([None] * len(self.calib1s))
        else:
            self.calib2s = filenumbers['second_calib']

        self.flats = filenumbers['flat']
        self.scis = filenumbers['science']
        self.nobs = len(self.calib1s)


        self.two_step_calib = (self.calib2s[0] is not None)

        self.observations = OrderedDict()
        self.calibration_pairs = OrderedDict()
        self.obs_set_index_lookup = OrderedDict()
        self.cal_pair_index_lookup = OrderedDict()

        self.obs_pairing_strategy = obs_pairing_strategy

        if self.obs_pairing_strategy not in ['nearest','user']: # 'unique',
            # 'nearest' chooses closest filenumber to sci
            # 'unique' pairs comps with science as uniquely as possible
            #             while it's second priority is to pair closest
            #             NOTE YET IMPLEMENTED
            # 'user'  pairs in the exact order given in the filenum lists
            self.obs_pairing_strategy = 'nearest'

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
        obs_set_index_lookup = OrderedDict()
        cal_pair_index_lookup = OrderedDict()
        if self.obs_pairing_strategy == 'unique':
            print("Unique observation pairing isn't yet implemented, defaulting to 'nearest'")
            self.obs_pairing_strategy = 'nearest'

        if self.obs_pairing_strategy == 'user':
            for ii,sci,flat,calib1,calib2 in enumerate(zip(scis,flats,calib1s,calib2s)):
                calibration_pairs[ii] = (calib1,calib2)
                observation_sets[ii] = (sci,flat,calib1,calib2)

        if self.obs_pairing_strategy == 'nearest':
            for ii,cal1 in enumerate(calib1s):
                if self.two_step_calib:
                    nearest_calib2 = calib2s[np.argmin(np.abs(calib2s - cal1))]
                else:
                    nearest_calib2 = None
                calibration_pairs[ii] = (cal1,nearest_calib2)
                cal_pair_index_lookup[cal1] = ii
            for ii, sci in enumerate(scis):
                nearest_calib1 = calib1s[np.argmin(np.abs(calib1s - sci))]
                nearest_calib_pair_ind = cal_pair_index_lookup[nearest_calib1]
                cal1,cal2 = calibration_pairs[nearest_calib_pair_ind]
                observation_sets[ii] = (sci, cal1,cal2,nearest_calib_pair_ind)
                obs_set_index_lookup[sci] = ii

        self.observations = observation_sets
        self.calibration_pairs = calibration_pairs
        self.obs_set_index_lookup = obs_set_index_lookup
        self.cal_pair_index_lookup = cal_pair_index_lookup

    def return_calibration_pairs(self):
        return self.calibration_pairs

    def return_observation_sets(self):
        return self.observations



class FieldData:
    def __init__(self, science_filenums, bias_filenums, flat_filenums, fibermap_filenums, \
                 first_comp_filenums, second_comp_filenums=None,
                 filemanager=FileManager(), instrument=InstrumentState(), startstep=None,\
                 obs_pairing_strategy='nearest',calib_lamps1=list(),calib_lamps2=None,\
                 twod_to_oned_strategy='simple',debias_strategy='median',targeting_data=None, \
                 convert_adu_to_e=True):

        self.obs_pairing_strategy = obs_pairing_strategy
        self.twod_to_oned = twod_to_oned_strategy
        self.debias_strategy = debias_strategy
        self.convert_adu_to_e = convert_adu_to_e
        self.check_parameter_flags()

        self.twostep_wavecalib = (second_comp_filenums is not None)
        self.filemanager=filemanager
        self.instrument=instrument
        self.targeting_data = targeting_data
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

        self.reduction_order = {None: 0, 'raw': 0, 'bias': 1, 'stitch': 2, 'remove_crs': 3,   \
                                'apcut': 4, 'wavecalib': 5,'flat': 6, 'skysub': 7, 'combine': 8,\
                                'zfit': 9}

        self.step = None
        self.update_step(startstep)

        ## Dictionary of hdus
        ## Key is tuple: (camera,filnum,imtype,opamp)
        ## Value is the hdu for that tuple of information
        ## an aastropy.fits hdu with a header and data attribute
        self.all_hdus = {}
        if self.step is not None:
            self.read_all_filedata()

        self.observations = Observations(self.filenumbers,obs_pairing_strategy)
        self.calibrations = {}
        self.fit_data = {}

    def check_parameter_flags(self):
        if self.twod_to_oned != 'simple':
            print("The only implemented 2D to 1D strategy is the simple summation. Defaulting to that.")
            self.twod_to_oned = 'simple'

        if self.debias_strategy != 'median':
            print("Only median debias strategy is currently implemented. Defaulting to that.")
            self.debias_strategy = 'median'

    def update_step(self,step):
        self.step = step
        numeric_step_value = self.reduction_order[self.step]
        self.data_stitched = numeric_step_value>self.reduction_order['stitch']
        # self.bias_subtracted = numeric_step_value>2
        # self.cosmics_removed = numeric_step_value>3
        self.fibersplit    = numeric_step_value>self.reduction_order['apcut']
        # self.wavelengths_calibrated = numeric_step_value>5
        # self.flattened = numeric_step_value>6
        # self.observations_combined = numeric_step_value>7


        if numeric_step_value > self.reduction_order['bias']:
            if 'bias' in self.filenumbers.keys():
                self.filenumbers.pop('bias')
            if 'bias' not in self.master_types:
                self.master_types.append('bias')
            #for (camera, filnum, imtype, opamp) in self.all_hdus.keys():
            #    if imtype == 'bias':
            #        self.all_hdus.pop((camera, filnum, imtype, opamp))
        if numeric_step_value > self.reduction_order['stitch']:
            self.instrument.opamps = [None]
        if numeric_step_value > self.reduction_order['apcut']:
            if 'fibmap' in self.filenumbers.keys():
                self.filenumbers.pop('fibmap')
            if 'fibmap' not in self.master_types:
                self.master_types.append('fibmap')
            #for (camera, filnum, imtype, opamp) in self.all_hdus.keys():
            #    if imtype == 'fibmap':
            #        self.all_hdus.pop((camera, filnum, imtype, opamp))
        if numeric_step_value > self.reduction_order['wavecalib']:
            for camera in self.instrument.cameras:
                for filnum in self.filenumbers['first_calib']:
                    self.all_hdus.pop((camera,filnum,'first_calib',None))
                for filnum in self.filenumbers['second_calib']:
                    self.all_hdus.pop((camera,filnum,'second_calib',None))
            self.filenumbers.pop('first calib')
            self.filenumbers.pop('second calib')
        if numeric_step_value > self.reduction_order['flat']:
            if 'flat' in self.filenumbers.keys():
                self.filenumbers.pop('flat')
            if 'flat' not in self.master_types:
                self.master_types.append('flat')

        self.filemanager.update_templates_for(step)

    def proceed_to(self,step):
        if step != self.step:
            self.update_step(step)

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

        self.current_data_saved = True
        self.current_data_from_disk = True

        if return_data:
            return self.all_hdus

    def write_all_filedata(self):
        self.filemanager.write_all_filedata(self.all_hdus)
        self.current_data_saved = True

    def check_data_ready_for_current_step(self):
        numeric_step = self.reduction_order[self.step]
        if numeric_step > self.reduction_order['stitch']: # after stitching
            if self.instrument.opamps != [None]:
                return False
        if numeric_step >  self.reduction_order['bias']: # after bias sub
            if 'bias' not in self.master_types:
                return False
            elif 'bias' in self.filenumbers.keys():
                return False
        if numeric_step ==  self.reduction_order['apcut']: # for app cutting
            if (self.instrument.cameras[0],'master','fibmap',None) not in self.all_hdus.keys():
                return False
        if numeric_step >  self.reduction_order['apcut']: # after app cutting
            if self.observations.nobs == 0:
                return False
        if numeric_step >  self.reduction_order['wavecalib']: # after wavelength calibration
            if self.observations[0].calibration_coefs is None:
                return False
        if numeric_step >  self.reduction_order['flat']: # after flattening
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

        if self.step == 'bias':
            from bias_subtract import bias_subtract
            self.all_hdus = bias_subtract(self.all_hdus, self.filemanager.date_timestamp, strategy=self.debias_strategy)
        elif self.step == 'stitch':
            from stitch import stitch_all_images
            self.all_hdus = stitch_all_images(self.all_hdus,self.filemanager.date_timestamp, \
                                              convert_adu_to_e= self.convert_adu_to_e)
            self.instrument.opamps = [None]
            self.data_stitched = True
        elif self.step == 'remove_crs':
            if self.current_data_saved or self.current_data_from_disk:
                pass
            else:
                self.write_all_filedata()
            import PyCosmic
            for (camera, filenum, imtype, opamp) in self.all_hdus.keys():
                if imtype in ['bias','first_calib']:
                    continue

                readfile = self.filemanager.get_read_filename(camera=camera, imtype=imtype,\
                                                        filenum=filenum, amp=opamp)
                writefile = self.filemanager.get_write_filename(camera=camera, imtype=imtype,\
                                                        filenum=filenum, amp=opamp)
                maskfile = writefile.replace('.fits', '.crmask.fits')
                print("\nFor image type: {}, shoe: {},   filenum: {}".format(imtype, camera, filenum))
                # outdat, pycosmask, pyheader =
                PyCosmic.detCos(readfile, maskfile, writefile, rdnoise='ENOISE',parallel=False,\
                                                              sigma_det=8, gain='EGAIN', verbose=True, return_data=False)
            self.read_all_filedata()
        elif self.step == 'apcut':
            for camera in self.instrument.cameras:
                self.combine_fibermaps(camera, return_table=False)
            from apperature_cut import cutout_all_apperatures
            self.all_hdus = cutout_all_apperatures(self.all_hdus,self.instrument.cameras,\
                                                   deadfibers=self.instrument.deadfibers,summation_preference=self.twod_to_oned)
        elif self.step == 'wavecalib':
            calpairs = self.observations.return_calibration_pairs()
            calib1s = {}
            calib2s = {}
            for camera in self.instrument.cameras:
                cur_calib1s = {}
                cur_calib2s = {}
                for filnum in self.filenumbers['first_calib']:
                    cur_calib1s[filnum] = self.all_hdus[(camera,filnum,'first_calib',None)]
                calib1s[camera] = cur_calib1s
                for filnum in self.filenumbers['second_calib']:
                    cur_calib2s[filnum] = self.all_hdus[(camera,filnum,'second_calib',None)]
                calib2s[camera] = cur_calib2s

            for camera in self.instrument.cameras:
                self.calibrations[camera] = Calibrations(camera, self.caliblamps1, self.caliblamps2, calib1s[camera], self.filemanager, self.instrument.configuration, \
                 second_calibrations=calib2s[camera], pairings=calpairs, load_history=True, trust_after_first=False)
                self.calibrations[camera].run_initial_calibrations()

            for camera in self.instrument.cameras:
                self.calibrations[camera].run_final_calibrations()

        elif self.step == 'flat':
            for camera in self.instrument.cameras:
                self.flatten_sciences(cam=camera)
        elif self.step == 'skysub':
            for camera in self.instrument.cameras:
                self.match_skies(cam=camera)
                self.subtract_skies(cam=camera)
        elif self.step == 'combine':
            for camera in self.instrument.cameras:
                self.combine_science_observations(cam=camera)
        elif self.step == 'zfit':
            for camera in self.instrument.cameras:
                self.fit_redshfits(cam=camera)

        if self.step != 'remove_crs':
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

    def combine_fibermaps(self,camera,return_table=False):
        filnums = self.filenumbers['fibmap']
        master_fibmap = []
        fibmap_header = None
        for filnum in filnums:
            hdu = self.all_hdus.pop((camera, filnum, 'fibmap', None))
            if filnum == filnums[0]:
                fibmap_header = hdu.header
                master_fibmap = hdu.data
            else:
                master_fibmap += hdu.data

        self.all_hdus[(camera,'master','fibmap',None)] = fits.ImageHDU(data=master_fibmap,header=fibmap_header,name='flux')
        if return_table:
            return master_fibmap

    def combine_flats(self,camera,return_table=False):
        filnums = self.filenumbers['flat']
        all_flats = []
        flat_header = None
        for filnum in filnums:
            hdu = self.all_hdus.pop((camera, filnum, 'flat', None))
            if filnum == filnums[0]:
                flat_header = hdu.header
            all_flats.append(hdu.data)
        flat_table_one = all_flats[0]
        for key in flat_table_one.colnames:
            for flat in all_flats[1:]:
                flat_table_one[key].data += flat[key].data

        self.all_hdus[(camera,'master','flat',None)] = fits.BinTableHDU(data=flat_table_one,header=flat_header,name='flux')
        if return_table:
            return flat_table_one

    def flatten_sciences(self,cam):
        from flatten import flatten_data
        fiber_fluxes = self.combine_flats(camera = cam, return_table=True)
        calib_table = self.calibrations[cam].create_calibration_default(save=False)
        final_table, final_flux_array = flatten_data(fiber_fluxes=fiber_fluxes,waves=calib_table)

        for filnum in self.filenumbers['science']:
            hdu = self.all_hdus[(cam,filnum,'science',None)]
            data = hdu.data
            hdr = hdu.header
            for col in final_table.colnames:
                data[col] /= final_table[col]
            self.all_hdus[(cam,filnum,'science',None)] = fits.BinTableHDU(data=data,header=hdr,name='flux')

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

        self.all_hdus[(cam,'master','science',None)] = fits.BinTableHDU(data=combined_data,header=ref_sci_hdu.header,name='flux')

    def match_skies(self,cam):
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        first_obs = list(self.observations.observations.keys())[0]
        sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
        sci_hdu = self.all_hdus[(cam, sci_filnum, 'science', None)]
        sci_data = sci_hdu.data
        header = sci_hdu.header

        skies, scis = {},{}
        for key in sci_data.colnames:
            fibid = key.replace(cam,'FIBER')
            objname = header[fibid]
            if 'SKY' in objname:
                skies[key] = objname
            elif 'GAL' in objname:
                scis[key] = objname

        if self.targeting_data is None:
            skynums,skynames = [],[]
            for fibname, objname in skies.items():
                skynames.append(fibname)
                skynums.append(int(fibname.lstrip('rb')))
            skynums = np.array(skynums)
            target_sky_pair = {}
            for fibname, objname in scis.items():
                galnum = int(fibname.lstrip('rb'))
                minsepind = np.argmin(np.abs(galnum-skynums))
                target_sky_pair[fibname] = skynames[minsepind]
        else:
            skyloc_array = []
            skynames = []
            for fibname,objname in skies.items():
                objdat = self.targeting_data[objname]
                skyloc_array.append([objdat['RA'],objdat['DEC']])
                skynames.append(fibname)
            sky_coords = SkyCoord(skyloc_array,units=u.deg)

            target_sky_pair = {}
            for fibname,objname in scis.items():
                objdat = self.targeting_data[objname]
                ra,dec = objdat['RA'],objdat['DEC']
                coord = SkyCoord([ra,dec],units=u.deg)
                seps = coord.separation(sky_coords)
                minsepind = np.argmin(seps)
                target_sky_pair[fibname] = skynames[minsepind]

        self.targeting_sky_pairs[cam] = target_sky_pair


    def subtract_skies(self,cam):
        observation_keys = list(self.observations.observations.keys())
        first_obs = observation_keys[0]
        sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
        npixels = len(self.all_hdus[(cam, sci_filnum, 'science', None)].data.columns[0])
        pixels = np.arange(npixels)

        target_sky_pair = self.targeting_sky_pairs[cam]
        for obs in observation_keys:
            sci_filnum, throw, throw1, final_calib_filnum = self.observations.observations[obs]
            sci_hdu = self.all_hdus[(cam, sci_filnum, 'science', None)]
            calib_data = self.all_hdus[(cam, final_calib_filnum, 'comp', None)].data
            sci_data = sci_hdu.data
            for galfib,skyfib in target_sky_pair.items():
                a, b, c, d, e, f = calib_data[skyfib]
                skylams = a + b + pixels + c * np.power(pixels, 2) + d * np.power(pixels, 3) + \
                       e * np.power(pixels, 4) + f * np.power(pixels, 5)
                skyflux = sci_data[skyfib]
                skyfit = CubicSpline(x=skylams,y=skyflux,extrapolate=False)

                a, b, c, d, e, f = calib_data[galfib]
                gallams = a + b + pixels + c * np.power(pixels, 2) + d * np.power(pixels, 3) + \
                       e * np.power(pixels, 4) + f * np.power(pixels, 5)

                outskyflux = skyfit(gallams)
                outskyflux[np.isnan(outskyflux)] = 0.
                sci_data[galfib] -= outskyflux
            self.all_hdus[(cam, sci_filnum, 'science', None)] = fits.BinTableHDU(data=sci_data,header=sci_hdu.header,name='flux')

    def fit_redshfits(self,cam):
        from zfit_testing import fit_redshifts
        sci_hdu = self.all_hdus[(cam,'master','science',None)]
        sci_data = sci_hdu.data
        results = fit_redshifts(sci_data,mask_name=self.filemanager.maskname, run_auto=True)
        self.fit_data[cam] = results


class ReductionController:
    def __init__(self,fielddata,filemanager,instrumentsetup):
        self.field_data = fielddata
        self.file_manager = filemanager
        self.instrument_setup = instrumentsetup

        self.field_data.load(step='bias')
        self.field_data.run_step(step='bias')
        self.field_data.save_data()
        self.field_data.proceed_to(step='remove_crs')
        self.field_data.check_data_consistent_for_current_step(step='remove_crs')
        self.field_data.run_step('remove_crs')
        self.field_data.save_data()
        self.field_data.proceed_to(step='sddsf')





