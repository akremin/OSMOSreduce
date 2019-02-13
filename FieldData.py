from astropy.io import fits
import numpy as np
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt, find_peaks

from calibrations import Calibrations
from observations import Observations


class FieldData:
    def __init__(self, filenumbers, filemanager, instrument,
                     startstep='bias', pipeline_options={} ):

        self.obs_pairing_strategy = pipeline_options['pairing_strategy']
        self.twod_to_oned = pipeline_options['twod_to_oned_strategy']
        self.debias_strategy = pipeline_options['debias_strategy']
        self.convert_adu_to_e = (str(pipeline_options['convert_adu_to_e']).lower()=='true')
        self.check_parameter_flags()

        self.twostep_wavecomparc = (len(list(filenumbers['fine_comp']))>0)
        self.filemanager=filemanager
        self.instrument=instrument
        # self.targeting_data = targeting_data
        self.comparc_lampsc = instrument.coarse_lamp_names
        self.comparc_lampsf = instrument.fine_lamp_names

        self.filenumbers = filenumbers

        self.data_stitched = False
        self.fibersplit = False
        self.current_data_saved = False
        self.current_data_from_disk = False

        self.reduction_order = {None: 0, 'raw': 0, 'bias': 1, 'stitch': 2, 'remove_crs': 3,   \
                                'apcut': 4, 'wavecalib': 5,'flatten': 6, 'skysub': 7, 'combine': 8,\
                                'zfit': 9}


        ## Dictionary of hdus
        ## Key is tuple: (camera,filnum,imtype,opamp)
        ## Value is the hdu for that tuple of information
        ## an aastropy.fits hdu with a header and data attribute
        self.step = None
        self.observations = Observations(self.filenumbers,self.obs_pairing_strategy)
        self.comparcs = {}
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
        self.fibersplit    = numeric_step_value>self.reduction_order['apcut']


        if numeric_step_value > self.reduction_order['bias']:
            self.filenumbers['bias'] = np.array(['master'])
        if numeric_step_value > self.reduction_order['stitch']:
            self.instrument.opamps = [None]
        if numeric_step_value > self.reduction_order['apcut']:
            self.filenumbers['fibermap'] = np.array(['master'])
        #
        # if numeric_step_value > self.reduction_order['wavecalib']:
        #     pass
        if numeric_step_value > self.reduction_order['flatten']:
            self.filenumbers['twiflat'] = np.array(['master'])
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
            if (self.instrument.cameras[0],'master','fibermap',None) not in self.all_hdus.keys():
                return False
        if numeric_step >  self.reduction_order['apcut']: # after app cutting
            if self.observations.nobs == 0:
                return False
        if numeric_step >  self.reduction_order['wavecalib']: # after wavelength comparc_
            pass
        if numeric_step >  self.reduction_order['flatten']: # after flattening
            if self.filenumbers['twiflat'][0] != 'master':
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
                readfile = self.filemanager.get_read_filename(camera=camera, imtype=imtype,\
                                                        filenum=filenum, amp=opamp)
                writefile = self.filemanager.get_write_filename(camera=camera, imtype=imtype,\
                                                        filenum=filenum, amp=opamp)
                maskfile = writefile.replace('.fits', '.crmask.fits')
                print("\nFor image type: {}, shoe: {},   filenum: {}".format(imtype, camera, filenum))

                PyCosmic.detCos(readfile, maskfile, writefile, rdnoise='ENOISE',parallel=False,\
                                                              sigma_det=8, gain='EGAIN', verbose=True, return_data=False)
            self.proceed_to('apcut')
            self.read_all_filedata()
        elif self.step == 'apcut':
            for camera in self.instrument.cameras:
                self.combine_fibermaps(camera, return_table=False)
            from aperture_detector import cutout_all_apperatures
            self.all_hdus = cutout_all_apperatures(self.all_hdus,self.instrument.cameras,\
                                                   deadfibers=self.instrument.deadfibers,summation_preference=self.twod_to_oned)
        elif self.step == 'wavecalib':
            if len(self.comparcs.keys()) == 0:
                self.populate_calibrations()
            for camera in self.instrument.cameras:
               self.comparcs[camera].run_initial_calibrations()
            ##hack
            # outdata, thetype = self.filemanager.locate_comparc_dict('basic-HgAr-NeAr-Xe', 'r', '11C', 628, locate_type='basic')
            # self.comparcs['r'].first_comparc_coefs[0] = Table(outdata)
            for camera in self.instrument.cameras:
                self.comparcs[camera].run_final_calibrations()

        elif self.step == 'flatten':
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
        comparc_pairs = self.observations.return_comparc_pairs()
        self.comparcs = {}
        for camera in self.instrument.cameras:
            comparc_cs = {filnum:self.all_hdus[(camera,filnum,'coarse_comp',None)] for filnum in self.observations.comparc_cs}
            if self.observations.two_step_comparc:
                comparc_fs = {filnum: self.all_hdus[(camera, filnum, 'fine_comp', None)] for filnum in self.observations.comparc_fs}
            else:
                comparc_fs = None
            comparc = Calibrations(camera, self.comparc_lampsc, self.comparc_lampsf, comparc_cs, self.filemanager, \
                                 config=self.instrument.configuration, fine_calibrations=comparc_fs,
                                 pairings=comparc_pairs, load_history=True, trust_after_first=False)

            self.comparcs[camera] = comparc

    def run_initial_calibrations(self):
        for camera in self.instrument.cameras:
            self.comparcs[camera].run_initial_calibrations()

    def run_final_calibrations(self):
        for camera in self.instrument.cameras:
            self.comparcs[camera].run_final_calibrations()

    def combine_fibermaps(self,camera,return_table=False):
        filnums = self.filenumbers['fibermap']
        master_fibmap = []
        fibmap_header = None
        for filnum in filnums:
            hdu = self.all_hdus.pop((camera, filnum, 'fibermap', None))
            if filnum == filnums[0]:
                fibmap_header = hdu.header
                master_fibmap = hdu.data
            else:
                master_fibmap += hdu.data

        self.all_hdus[(camera,'master','fibermap',None)] = fits.ImageHDU(data=master_fibmap,header=fibmap_header,name='flux')
        if return_table:
            return master_fibmap

    def combine_flats(self,camera,return_table=False):
        filnums = self.filenumbers['twiflat']
        all_flats = []
        flat_header = None
        for filnum in filnums:
            hdu = self.all_hdus.pop((camera, filnum, 'twiflat', None))
            if filnum == filnums[0]:
                flat_header = hdu.header
            all_flats.append(Table(hdu.data))
        flat_table_one = all_flats[0]
        for key in flat_table_one.colnames:
            for flat in all_flats[1:]:
                flat_table_one[key] = flat_table_one[key] + flat[key]

        self.all_hdus[(camera,'master','twiflat',None)] = fits.BinTableHDU(data=flat_table_one,header=flat_header,name='flux')
        if return_table:
            return flat_table_one

    def flatten_sciences(self,cam):
        from flatten import flatten_data
        fiber_fluxes = self.combine_flats(camera = cam, return_table=True)
        if len(self.comparcs[cam].final_comparc_rated_hdulists.keys()) == 0:
            self.comparcs[cam].load_final_comparc_hdus()
        comparc_table = self.comparcs[cam].create_comparc_default(save=False)
        final_table, final_flux_array = flatten_data(fiber_fluxes=fiber_fluxes,waves=comparc_table)

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
        if len(self.comparcs[cam].final_comparc_rated_hdulists.keys()) == 0:
            self.comparcs[cam].load_final_comparc_hdus()
        observation_keys = list(self.observations.observations.keys())
        nobs = len(observation_keys)
        middle_obs = observation_keys[nobs//2]

        ref_science_filnum,throw,throw1,comparc_ind = self.observations.observations[middle_obs]
        ref_sci_hdu = self.all_hdus.pop((cam,ref_science_filnum,'science',None))
        ref_sci_data = Table(ref_sci_hdu.data)
        ref_calibrations = self.comparcs[cam].second_comparc_coefs[comparc_ind]
        ref_wavearrays = {}
        pixels = np.arange(len(ref_sci_data)).astype(np.float64)
        pix2 = np.power(pixels, 2)
        pix3 = np.power(pixels, 3)
        pix4 = np.power(pixels, 4)
        pix5 = np.power(pixels, 5)
        combined_data = ref_sci_data.copy()
        for col in ref_sci_data.colnames:
            a,b,c,d,e,f = ref_calibrations[col]
            lams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
            ref_wavearrays[col] = lams
        for obs in observation_keys:
            if obs == middle_obs:
                continue
            sci_filnum, throw, throw1, comparc_ind = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))
            sci_data = sci_hdu.data
            sci_calibrations = self.comparcs[cam].second_comparc_coefs[comparc_ind]
            for col in ref_sci_data.colnames:
                a, b, c, d, e, f = sci_calibrations[col]
                lams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                flux = sci_data[col]
                scifit = CubicSpline(x=lams,y=flux)
                outwave = ref_wavearrays[col]
                outflux = scifit(outwave)
                combined_data[col] += outflux

        plt.figure()
        for col in ref_sci_data.colnames:
            plt.plot(ref_wavearrays[col],combined_data[col],alpha=0.4)
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
            fibid = key.replace(cam,'FIBER')
            objname = header[fibid]
            if 'SKY' in objname:
                skies[key] = objname
            elif 'GAL' in objname:
                scis[key] = objname

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
        from quickreduce_funcs import smooth_and_dering
        if len(self.comparcs[cam].final_comparc_rated_hdulists.keys()) == 0:
            self.comparcs[cam].load_final_comparc_hdus()
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
        scis = {}
        # plt.figure()
        for obs in observation_keys:
            sci_filnum, throw, throw2, comparc_ind = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))
            comparc_data = self.comparcs[cam].second_comparc_coefs[comparc_ind]
            sci_data = Table(sci_hdu.data)
            out_sci_data = Table()
            sci_lams = {}
            skyfibs = np.unique(list(target_sky_pair.values()))
            a, b, c, d, e, f = comparc_data['{}214'.format(cam)]
            skyllams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5

            master_skies = []
            skyfits = {}
            for skyfib in skyfibs:
                a, b, c, d, e, f = comparc_data[skyfib]
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
                a, b, c, d, e, f = comparc_data[galfib]
                gallams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                galflux = sci_data[galfib]
                galflux[np.isnan(galflux)] = 0.
                continuum = medfilt(sci_data[galfib], 371)
                medgal_contsubd = medfilt(galflux - continuum, 3)
                subd_galflux = galflux - continuum

                skyfit = skyfits[skyfib]
                outskyflux = skyfit(gallams)
                outskyflux[np.isnan(outskyflux)] = 0.
                sky_contsubd = medfilt(outskyflux, 5)

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
                gpeak_inds, gpeak_props = find_peaks(medgal_contsubd, height=(None, None), width=(1, 8), \
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

                ## Look for peaks in the sky spectrum
                speak_inds, speak_props = find_peaks(sky_contsubd, height=(None, None), width=(1, 8), \
                                                     threshold=(None, None), prominence=(priminence_threshold, None),
                                                     wlen=24)
                sheights = speak_props['peak_heights']  # peak_props['prominences']
                speak_lefts = (speak_props['left_bases']).astype(int)
                speak_rights = (speak_props['right_bases']).astype(int)

                ## As with the median sky spectrum, decouple nested doublets and triplets
                for ii in range(len(speak_lefts) - 1):
                    if speak_lefts[ii + 1] < speak_rights[ii]:
                        if speak_lefts[ii + 1] - speak_inds[ii] > 0:
                            speak_rights[ii] = speak_lefts[ii + 1]
                        else:
                            speak_lefts[ii + 1] = speak_rights[ii]

                ## For each sky peak, look to see if it exists in the galaxy spectrum
                ## If it exists, scale the peak flux to match the galaxy and subtract that line
                ## if line doesn't exist, do nothing
                doctored = subd_galflux.copy()
                mean_ratio = np.mean(gheights) / np.mean(heights)
                npeaks = len(peak_inds)
                for ii, peak in enumerate(peak_inds):
                    match = np.where(np.abs(gpeak_inds - peak) < 1.1)[0]
                    if len(match) > 0:
                        if len(match) > 1:
                            match = np.max(match)#match[np.argmax(gheights[match])]
                        sky_height = heights[ii]
                        left = peak_lefts[ii]
                        right = peak_rights[ii]
                        if ((gpeak_rights[match] - right) > 4) and (ii < npeaks-1):
                            if np.abs(gpeak_rights[match] - peak_rights[ii + 1]) < 4:
                                print("changed rights")
                                right = int(peak_rights[ii + 1])
                                sky_height = np.max([heights[ii], heights[ii + 1]])
                        elif ((left - gpeak_lefts[match]) > 4) and (ii>0):
                            if np.abs(peak_lefts[ii - 1] - gpeak_lefts[match]) < 4:
                                print("changed lefts")
                                left = int(peak_lefts[ii - 1])
                                sky_height = np.max([heights[ii], heights[ii - 1]])

                        if (right - gpeak_rights[match]) > 2:
                            right = int(gpeak_rights[match])
                        if (gpeak_lefts[match] - left) > 2:
                            left = int(gpeak_lefts[match])

                        gal_height = gheights[match]

                        ratio = np.mean(doctored[left:right]) / np.mean(master_interp[left:right])
                        if (ratio > 0.1 * mean_ratio) and (ratio < 10 * mean_ratio):
                            test_sub = doctored[left:right] - (ratio * master_interp[left:right])
                            if np.std(test_sub) > (6*np.std(doctored[200:200+4*(right-left)])):
                                #slope = ((doctored[right]-doctored[left])/(right-left))
                                #test_sub = np.arange(right-left)*slope+doctored[left]
                                test_sub = np.ones(right-left)*np.sort(doctored[left:right])[((right-left)//10)]
                            doctored[left:right] = test_sub

                out_sci_data.add_column(Table.Column(name=galfib,data=(doctored+continuum)))
                sci_lams[galfib] = gallams
                # plt.plot(gallams,sci_data[galfib],alpha=0.4)
                #plt.plot(gallams,doctored+continuum,alpha=0.4)
            scis[obs] = (out_sci_data.copy(),sci_lams.copy(),sci_filnum)
            self.all_hdus[(cam, sci_filnum, 'science', None)] = fits.BinTableHDU(data=out_sci_data,header=sci_hdu.header,name='flux')
            # plt.show()
        # plt.figure()
        # for obs,(scidict,lamdict,scifile) in scis.items():
        #     for nam in lamdict.keys():
        #         plt.plot(lamdict[nam],scidict[nam],alpha=0.4)
        # plt.show()



    def fit_redshfits(self,cam):
        from fit_redshifts import fit_redshifts
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