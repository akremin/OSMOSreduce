import os
from astropy.io import fits
#from quickreduce_funcs import print_data_neatly
import numpy as np
from astropy.table import Table

from inputoutput import FileManager
from instrument import InstrumentState
from calibrations import Calibrations
from collections import OrderedDict
from scipy.interpolate import CubicSpline,UnivariateSpline
from scipy.signal import medfilt,find_peaks
import matplotlib.pyplot as plt

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
        cal1_pair_index_lookup = OrderedDict()
        cal2_pair_index_lookup = OrderedDict()
        if self.obs_pairing_strategy == 'unique':
            print("Unique observation pairing isn't yet implemented, defaulting to 'nearest'")
            self.obs_pairing_strategy = 'nearest'

        if self.obs_pairing_strategy == 'user':
            for ii,sci,flat,calib1,calib2 in enumerate(zip(scis,flats,calib1s,calib2s)):
                calibration_pairs[ii] = (calib1,calib2)
                observation_sets[ii] = (sci,flat,calib1,calib2)

        if self.obs_pairing_strategy == 'nearest':
            if self.two_step_calib:
                for ii,cal2 in enumerate(calib2s):
                    nearest_calib1 = calib1s[np.argmin(np.abs(calib1s - cal2))]
                    calibration_pairs[ii] = (nearest_calib1,cal2)
                    cal1_pair_index_lookup[nearest_calib1] = ii
                    cal2_pair_index_lookup[cal2] = ii
            else:
                for ii, cal1 in enumerate(calib1s):
                    calibration_pairs[ii] = (cal1, None)
                    cal1_pair_index_lookup[cal1] = ii
                    cal2_pair_index_lookup[cal1] = ii
            for ii, sci in enumerate(scis):
                nearest_calib2 = calib2s[np.argmin(np.abs(calib2s - sci))]
                nearest_calib_pair_ind = cal2_pair_index_lookup[nearest_calib2]
                cal1,cal2 = calibration_pairs[nearest_calib_pair_ind]
                observation_sets[ii] = (sci, cal1,cal2,nearest_calib_pair_ind)
                obs_set_index_lookup[sci] = ii

        self.observations = observation_sets
        self.calibration_pairs = calibration_pairs
        self.obs_set_index_lookup = obs_set_index_lookup
        self.cal1_pair_index_lookup = cal1_pair_index_lookup
        self.cal2_pair_index_lookup = cal2_pair_index_lookup

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


        ## Dictionary of hdus
        ## Key is tuple: (camera,filnum,imtype,opamp)
        ## Value is the hdu for that tuple of information
        ## an aastropy.fits hdu with a header and data attribute
        self.step = None
        self.observations = Observations(self.filenumbers,obs_pairing_strategy)
        self.calibrations = {}
        self.fit_data = {}
        self.targeting_sky_pairs = {}
        self.update_step(startstep)
        self.all_hdus = {}
        self.read_all_filedata()
        if self.reduction_order[startstep]>self.reduction_order['apcut']:
            self.populate_calibrations()
        self.mtlz = self.filemanager.get_matched_target_list()

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
            self.filenumbers['bias'] = np.array(['master'])
        if numeric_step_value > self.reduction_order['stitch']:
            self.instrument.opamps = [None]
        if numeric_step_value > self.reduction_order['apcut']:
            self.filenumbers['fibmap'] = np.array(['master'])

        if numeric_step_value > self.reduction_order['wavecalib']:
            pass
        if numeric_step_value > self.reduction_order['flat']:
            self.filenumbers['flat'] = np.array(['master'])
        if numeric_step_value > self.reduction_order['combine']:
            self.filenumbers['science'] = np.array(['combined_fluxes','combined_wavelengths'])
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

        self.all_hdus = self.filemanager.read_all_filedata(self.filenumbers,self.instrument,\
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
            if self.filenumbers['bias'][0] != 'master':
                return False
        if numeric_step ==  self.reduction_order['apcut']: # for app cutting
            if (self.instrument.cameras[0],'master','fibmap',None) not in self.all_hdus.keys():
                return False
        if numeric_step >  self.reduction_order['apcut']: # after app cutting
            if self.observations.nobs == 0:
                return False
        if numeric_step >  self.reduction_order['wavecalib']: # after wavelength calibration
            pass
        if numeric_step >  self.reduction_order['flat']: # after flattening
            if self.filenumbers['flat'][0] != 'master':
                return False

        return True


    def run_step(self,step=None):
        if step is not None:
            self.proceed_to(step)

        if self.step not in self.reduction_order.keys():
            print("Step \"{}\" doesn't have an action assosciated with it".format(self.step))

        if self.step == 'bias':
            from bias_subtract import bias_subtract
            self.all_hdus = bias_subtract(self.all_hdus, self.filemanager.date_timestamp, strategy=self.debias_strategy, \
                                          convert_adu_to_e=self.convert_adu_to_e)
        elif self.step == 'stitch':
            from stitch import stitch_all_images
            self.all_hdus = stitch_all_images(self.all_hdus,self.filemanager.date_timestamp)
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
            if len(self.calibrations.keys()) == 0:
                self.populate_calibrations()
            for camera in self.instrument.cameras:
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
            calib1s = {filnum:self.all_hdus[(camera,filnum,'first_calib',None)] for filnum in self.observations.calib1s}
            if self.observations.two_step_calib:
                calib2s = {filnum: self.all_hdus[(camera, filnum, 'second_calib', None)] for filnum in self.observations.calib2s}
            else:
                calib2s = None
            calib = Calibrations(camera, self.caliblamps1, self.caliblamps2, calib1s, self.filemanager, \
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
            all_flats.append(Table(hdu.data))
        flat_table_one = all_flats[0]
        for key in flat_table_one.colnames:
            for flat in all_flats[1:]:
                flat_table_one[key] = flat_table_one[key] + flat[key]

        self.all_hdus[(camera,'master','flat',None)] = fits.BinTableHDU(data=flat_table_one,header=flat_header,name='flux')
        if return_table:
            return flat_table_one

    def flatten_sciences(self,cam):
        from flatten import flatten_data
        fiber_fluxes = self.combine_flats(camera = cam, return_table=True)
        if len(self.calibrations[cam].final_calibrated_hdulists.keys()) == 0:
            self.calibrations[cam].load_final_calib_hdus()
        calib_table = self.calibrations[cam].create_calibration_default(save=False)
        final_table, final_flux_array = flatten_data(fiber_fluxes=fiber_fluxes,waves=calib_table)

        import matplotlib.pyplot as plt
        for filnum in self.filenumbers['science']:
            hdu = self.all_hdus[(cam,filnum,'science',None)]
            data = hdu.data
            hdr = hdu.header
            #data_arr = np.ndarray(shape=(len(final_table.colnames),len(final_table.columns[0])))
            #plt.figure()
            for ii,col in enumerate(final_table.colnames):
                data[col] /= final_table[col]
                #plt.plot(np.arange(len(data[col])),data[col])
                #data_arr[ii,:] = data[col]
            #plt.show()
            # plt.figure()
            # plt.imshow(data_arr, 'gray', aspect='auto', origin='lowerleft')
            # plt.colorbar()
            # plt.show()
            self.all_hdus[(cam,filnum,'science',None)] = fits.BinTableHDU(data=data,header=hdr,name='flux')

    def combine_science_observations(self, cam):
        if len(self.calibrations[cam].final_calibrated_hdulists.keys()) == 0:
            self.calibrations[cam].load_final_calib_hdus()
        observation_keys = list(self.observations.observations.keys())
        nobs = len(observation_keys)
        middle_obs = observation_keys[nobs//2]

        ref_science_filnum,throw,throw1,calib_ind = self.observations.observations[middle_obs]
        ref_sci_hdu = self.all_hdus.pop((cam,ref_science_filnum,'science',None))
        ref_sci_data = ref_sci_hdu.data
        ref_calibs = self.calibrations[cam].second_calibration_coefs[calib_ind]
        ref_wavearrays = {}
        pixels = np.arange(len(ref_sci_data)).astype(np.float64)
        pix2 = np.power(pixels, 2)
        pix3 = np.power(pixels, 3)
        pix4 = np.power(pixels, 4)
        pix5 = np.power(pixels, 5)
        combined_data = ref_sci_data.copy()
        for col in ref_calibs.colnames:
            a,b,c,d,e,f = ref_calibs[col]
            lams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
            ref_wavearrays[col] = lams
        for obs in observation_keys:
            if obs == middle_obs:
                continue
            sci_filnum, throw, throw1, calibind = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))
            sci_data = sci_hdu.data
            sci_calibs = self.calibrations[cam].second_calibration_coefs[calibind]
            for col in sci_calibs.colnames:
                a, b, c, d, e, f = sci_calibs[col]
                lams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                flux = sci_data[col]
                scifit = CubicSpline(x=lams,y=flux)
                outwave = ref_wavearrays[col]
                outflux = scifit(outwave)
                combined_data[col] += outflux
        import matplotlib.pyplot as plt
        plt.figure()
        for col in sci_calibs.colnames:
            plt.plot(ref_wavearrays[col],combined_data[col])
        plt.show()
        ref_wave_table = Table(ref_wavearrays)
        self.all_hdus[(cam,'combined_fluxes','science',None)] = fits.BinTableHDU(data=combined_data,header=ref_sci_hdu.header,name='flux')
        self.all_hdus[(cam,'combined_wavelengths','science',None)] = fits.BinTableHDU(data=ref_wave_table)

    def match_skies(self,cam):
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        first_obs = list(self.observations.observations.keys())[0]
        sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
        sci_hdu = self.all_hdus[(cam, sci_filnum, 'science', None)]
        sci_data = Table(sci_hdu.data)
        header = sci_hdu.header

        skies, scis = {},{}
        for key in sci_data.colnames:
            # camera,tet,fib = key[0],int(key[1]),int(key[2:])
            # if camera.lower() != cam.lower():
            #     raise(ValueError,"Camera didn't match the fiber camera designation")
            # fibid = "{s}{d}{:02d}".format(camera,tet+1,fib+1)
            fibid = key.replace(cam,'FIBER')
            objname = header[fibid]
            if 'SKY' in objname:
                skies[key] = objname
            elif 'GAL' in objname:
                scis[key] = objname
        ##hack!
        #skies.pop('r202')
        #skies.pop('r315')
        if self.mtlz is None:
            skynums,skynames = [],[]
            for fibname in skies.keys():
                skynames.append(fibname)
                number_str = fibname.lstrip('rb')
                numeric = 8*int(number_str[0])+int(number_str[1:])
                skynums.append(numeric)
            skynums = np.array(skynums)
            target_sky_pair = {}
            for fibname, objname in scis.items():
                number_str = fibname.lstrip('rb')
                galnum = 8*int(number_str[0])+int(number_str[1:])
                minsepind = np.argmin(np.abs(skynums-galnum))
                target_sky_pair[fibname] = skynames[minsepind]
        else:
            skyloc_array = []
            skynames = []
            for fibname,objname in skies.items():
                fibrow = np.where(self.mtlz['FIBNAME']==fibname)[0][0]
                objrow = np.where(self.mtlz['ID']==objname)[0][0]
                if fibrow!=objrow:
                    print("Fiber and object matched to different rows!")
                    print(fibname,objname,fibrow,objrow)
                    raise()
                if self.mtlz['RA'].mask[fibrow]:
                    ra,dec = self.mtlz['RA_drilled'][fibrow],self.mtlz['DEC_drilled'][fibrow]
                else:
                    ra,dec = self.mtlz['RA'][fibrow],self.mtlz['DEC'][fibrow]
                skyloc_array.append([ra,dec])
                skynames.append(fibname)
            sky_coords = SkyCoord(skyloc_array,unit=u.deg)

            target_sky_pair = {}
            for fibname,objname in scis.items():
                fibrow = np.where(self.mtlz['FIBNAME']==fibname)[0][0]
                objrow = np.where(self.mtlz['ID']==objname)[0][0]
                if fibrow!=objrow:
                    print("Fiber and object matched to different rows!")
                    print(fibname,objname,fibrow,objrow)
                    raise()
                if self.mtlz['RA'].mask[fibrow]:
                    ra,dec = self.mtlz['RA_drilled'][fibrow],self.mtlz['DEC_drilled'][fibrow]
                else:
                    ra,dec = self.mtlz['RA'][fibrow],self.mtlz['DEC'][fibrow]
                coord = SkyCoord(ra,dec,unit=u.deg)
                seps = coord.separation(sky_coords)
                minsepind = np.argmin(seps)
                target_sky_pair[fibname] = skynames[minsepind]

        self.targeting_sky_pairs[cam] = target_sky_pair


    def subtract_skies(self,cam):
        if len(self.calibrations[cam].final_calibrated_hdulists.keys()) == 0:
            self.calibrations[cam].load_final_calib_hdus()
        observation_keys = list(self.observations.observations.keys())
        first_obs = observation_keys[0]
        sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
        npixels = len(self.all_hdus[(cam, sci_filnum, 'science', None)].data)
        pixels = np.arange(npixels).astype(np.float64)
        pix2 = np.power(pixels, 2)
        pix3 = np.power(pixels, 3)
        pix4 = np.power(pixels, 4)
        pix5 = np.power(pixels, 5)
        target_sky_pair = self.targeting_sky_pairs[cam]
        ##hack!
        def plot_skies(axi,minlam,maxlam):
            skys = Table.read(r'C:\Users\kremin\Github\M2FSreduce\lamp_linelists\gident_UVES_skylines.csv',format='ascii.csv')
            sky_lines = skys['WAVE_AIR']
            sky_fluxes = skys['FLUX']
            shortlist_skylines = ((sky_lines>minlam)&(sky_lines<maxlam))
            shortlist_fluxes = (sky_fluxes > 0.2)
            shortlist_bool = (shortlist_fluxes & shortlist_skylines)

            select_table = skys[shortlist_bool]
            from calibrations import air_to_vacuum
            shortlist_skylines = air_to_vacuum(select_table['WAVE_AIR'])
            fluxes = select_table['FLUX']
            log_flux = np.log(fluxes-np.min(fluxes)+1.01)
            max_log_flux = np.max(log_flux)
            for vlin,flux in zip(shortlist_skylines,log_flux):
                axi.axvline(vlin, ls='-.', alpha=0.2, c='black',lw=0.4+4*flux/max_log_flux)

        def smooth_and_dering(outskyflux):
            outskyflux[np.isnan(outskyflux)] = 0.
            smthd_outflux = medfilt(outskyflux, 3)
            peak_inds, peak_props = find_peaks(outskyflux, height=(1500, 100000), width=(0, 20))
            heights = peak_props['peak_heights']
            ringing_factor = 1 + (heights // 1000)
            ring_lefts = (peak_inds - ringing_factor * (peak_inds - peak_props['left_bases'])).astype(int)
            peak_lefts = (peak_props['left_bases']).astype(int)
            ring_rights = (peak_inds + ringing_factor * (peak_props['right_bases'] - peak_inds)).astype(int)
            peak_rights = (peak_props['right_bases']).astype(int)

            corrected = smthd_outflux.copy()
            for rleft, pleft in zip(ring_lefts, peak_lefts):
                corrected[rleft:pleft] = smthd_outflux[rleft:pleft]
            for rright, pright in zip(ring_rights, peak_rights):
                corrected[pright:rright] = smthd_outflux[pright:rright]
            return corrected

        for obs in observation_keys[1:]:
            sci_filnum, throw, throw2, calib_ind = self.observations.observations[obs]
            sci_hdu = self.all_hdus[(cam, sci_filnum, 'science', None)]
            calib_data = self.calibrations[cam].second_calibration_coefs[calib_ind]
            sci_data = sci_hdu.data

            skyfibs = np.unique(list(target_sky_pair.values()))

            a, b, c, d, e, f = calib_data['r214']
            skyllams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
            master_skies = []
            skyfits = {}
            for skyfib in skyfibs:
                a, b, c, d, e, f = calib_data[skyfib]
                skylams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                skyflux = medfilt(sci_data[skyfib]-medfilt(sci_data[skyfib],371),3)
                #plt.plot(skylams,skyflux,label=skyfib)
                skyfit = CubicSpline(x=skylams, y=skyflux, extrapolate=False)
                skyfits[skyfib] = skyfit
                outskyflux = skyfit(skyllams)
                corrected = smooth_and_dering(outskyflux)

                master_skies.append(corrected)

            master_sky = np.median(master_skies,axis=0)

            masterfit = CubicSpline(x=skyllams, y=master_sky, extrapolate=False)

            for galfib,skyfib in target_sky_pair.items():
                a, b, c, d, e, f = calib_data[galfib]
                gallams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                galflux = sci_data[galfib]
                galflux[np.isnan(galflux)] = 0.
                continuum = medfilt(sci_data[galfib], 371)
                gal_contsubd = medfilt(galflux - continuum, 3)

                # outskyflux = skyfit(gallams)
                # outskyflux[np.isnan(outskyflux)] = 0.
                # smthd_outflux = medfilt(outskyflux, 5)
                # peak_inds, peak_props = find_peaks(outskyflux, height=(1500, 100000), width=(0, 20))
                # heights = peak_props['peak_heights']
                # ringing_factor = 1 + (heights // 1000)
                # ring_lefts = (peak_inds - ringing_factor * (peak_inds - peak_props['left_bases'])).astype(int)
                # peak_lefts = (peak_props['left_bases'] + 1).astype(int)
                # ring_rights = (peak_inds + ringing_factor * (peak_props['right_bases'] - peak_inds)).astype(int)
                # peak_rights = (peak_props['right_bases'] + 1).astype(int)
                #
                # corrected = outskyflux.copy()
                # for rleft, pleft in zip(ring_lefts, peak_lefts):
                #     corrected[rleft:pleft] = smthd_outflux[rleft:pleft]
                # for rright, pright in zip(ring_rights, peak_rights):
                #     corrected[pright:rright] = smthd_outflux[pright:rright]

                # outsky = outskyflux - medfilt(outskyflux, 371)

                master_interp = masterfit(gallams)
                master_interp[np.isnan(master_interp)] = 0.

                ## Find the skylines
                priminence_threshold = np.max(master_interp) // 120
                peak_inds, peak_props = find_peaks(master_interp, height=(None, None), width=(1, 8), \
                                                   threshold=(None, None), prominence=(priminence_threshold, None),
                                                   wlen=24)
                heights = peak_props['peak_heights']  # peak_props['prominences']
                peak_lefts = (peak_props['left_bases']).astype(int)
                peak_rights = (peak_props['right_bases']).astype(int)

                ## Some doublets and triplets will be nested, decouple them
                for ii in range(len(peak_lefts) - 1):
                    if peak_lefts[ii + 1] < peak_rights[ii]:
                        if peak_lefts[ii + 1] - peak_inds[ii] > 0:
                            peak_rights[ii] = peak_lefts[ii + 1]
                        else:
                            peak_lefts[ii + 1] = peak_rights[ii]

                ## Look for peaks in the galaxy spectrum
                gpeak_inds, gpeak_props = find_peaks(gal_contsubd, height=(None, None), width=(1, 8), \
                                                     threshold=(None, None), prominence=(priminence_threshold, None),
                                                     wlen=24)
                gheights = gpeak_props['peak_heights']  # peak_props['prominences']
                gpeak_lefts = (gpeak_props['left_bases']).astype(int)
                gpeak_rights = (gpeak_props['right_bases']).astype(int)

                ## As with the sky spectrum, decouple nested doublets and triplets
                for ii in range(len(gpeak_lefts) - 1):
                    if gpeak_lefts[ii + 1] < gpeak_rights[ii]:
                        if gpeak_lefts[ii + 1] - gpeak_inds[ii] > 0:
                            gpeak_rights[ii] = gpeak_lefts[ii + 1]
                        else:
                            gpeak_lefts[ii + 1] = gpeak_rights[ii]

                ## For each sky peak, look to see if it exists in the galaxy spectrum
                ## If it exists, scale the peak flux to match the galaxy and subtract that line
                ## if line doesn't exist, do nothing
                subd_galflux = galflux - continuum
                mean_ratio = np.mean(gal_contsubd) / np.mean(master_interp)
                for ii, peak in enumerate(peak_inds):
                    match = np.where(np.abs(gpeak_inds - peak) < 1.1)[0]
                    if len(match) > 0:
                        if len(match) > 1:
                            match = match[np.argmax(gheights[match])]
                        sky_height = heights[ii]
                        left = peak_lefts[ii]
                        right = peak_rights[ii]
                        if (gpeak_rights[match] - right) > 4:
                            if np.abs(gpeak_rights[match] - peak_rights[ii + 1]) < 4:
                                right = peak_rights[ii + 1]
                                sky_height = np.max([heights[ii], heights[ii + 1]])
                        elif (left - gpeak_lefts[match]) > 4:
                            if np.abs(peak_lefts[ii - 1] - gpeak_lefts[match]) < 4:
                                left = peak_lefts[ii - 1]
                                sky_height = np.max([heights[ii], heights[ii - 1]])
                        gal_height = gheights[match]
                        ratio = gal_height / sky_height
                        if (ratio > 0.1 * mean_ratio) and (ratio < 10 * mean_ratio):
                            subd_galflux[left:right] = medfilt(
                                gal_contsubd[left:right] - ratio * master_interp[left:right], 5)

                ## Some skylines don't appear in the data. But low level sky is still there
                ## remove all identified peaks that we checked above, and then subtract the residual from the galaxy spectrum
                master_sub = master_interp.copy()
                for left, right in zip(peak_lefts, peak_rights):
                    master_sub[left:right] = master_sub[left:right] - master_interp[left:right]

                ## Look for peaks in the galaxy spectrum
                doctored = subd_galflux - master_sub
                priminence_threshold = np.max(doctored) // 60
                gpeak_inds, gpeak_props = find_peaks(doctored, height=(None, None), width=(1, 8), \
                                                     threshold=(None, None), prominence=(priminence_threshold, None),
                                                     wlen=24)
                gheights = gpeak_props['peak_heights']  # peak_props['prominences']
                height_sort_ind = np.argsort(gheights)[::-1]
                if len(gheights)<14:
                    ntops = len(gheights)
                else:
                    ntops = 14
                height_sort_inds = height_sort_ind[:ntops]
                top_peak_lefts =  gpeak_props['left_bases'][height_sort_inds]#).astype(int)
                top_peak_rights = gpeak_props['right_bases'][height_sort_inds]#).astype(int)
                for peakl,peakr in zip(top_peak_lefts,top_peak_rights):
                    doctored[peakl:peakr] = ((doctored[peakr]-doctored[peakl])/(peakr-peakl))*np.arange(peakr-peakl) + doctored[peakl]

                doctored += continuum
                fig, ax = plt.subplots()
                plt.title(skyfib)
                plt.plot(gallams+10,subd_galflux - master_sub + continuum + 100, label='before mask')
                plt.plot(gallams, doctored, label='after mask')
                plt.plot(gallams, master_interp, label='master')
                # for sky,skyfit in skyfits.items():
                #     outskyflux = skyfit(gallams)
                #     corrected = smooth_and_dering(outskyflux)
                #     plt.plot(gallams,corrected,label=sky)
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                plt.tight_layout()
                plt.legend(loc='best')
                plot_skies(ax, np.min(gallams), np.max(gallams))
                plt.show()

                sci_data[galfib] = subd_galflux - master_sub
            self.all_hdus[(cam, sci_filnum, 'science', None)] = fits.BinTableHDU(data=sci_data,header=sci_hdu.header,name='flux')

    def fit_redshfits(self,cam):
        from zfit_testing import fit_redshifts
        waves = Table(self.all_hdus[(cam,'combined_wavelengths','science',None)].data)
        fluxes = Table(self.all_hdus[(cam, 'combined_fluxes', 'science', None)].data)
        if (cam, 'combined_mask', 'science', None) in self.all_hdus.keys():
            mask = Table(self.all_hdus[(cam, 'combined_mask', 'science', None)].data)
        else:
            mask = None

        sci_data = OrderedDict()
        for ap in waves.colnames:
            if mask is None:
                current_mask = np.ones(len(waves[ap])).astype(bool)
            else:
                current_mask = mask[ap]
            sci_data[ap] = (waves[ap],fluxes[ap],current_mask)
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





