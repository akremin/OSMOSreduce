from astropy.io import fits
import numpy as np
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt, find_peaks
from collections import OrderedDict
from calibrations import Calibrations
from observations import Observations
import matplotlib.pyplot as plt

class FieldData:
    def __init__(self, filenumbers, filemanager, instrument,
                     startstep='bias', pipeline_options={} ):

        self.obs_pairing_strategy = pipeline_options['pairing_strategy']
        self.twod_to_oned = pipeline_options['twod_to_oned_strategy']
        self.debias_strategy = pipeline_options['debias_strategy']
        self.convert_adu_to_e = (str(pipeline_options['convert_adu_to_e']).lower()=='true')
        self.skip_coarse_calib = (str(pipeline_options['try_skip_coarse_calib']).lower()=='true')
        self.single_core = (str(pipeline_options['single_core']).lower()=='true')
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
        self.final_calibrated_hdulists = {}
        self.final_calibration_coefs = {}
        if self.reduction_order[startstep]>self.reduction_order['apcut']:
            self.get_final_wavelength_coefs()
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
            self.filenumbers['bias'] = np.array([])
        if numeric_step_value > self.reduction_order['stitch']:
            self.instrument.opamps = [None]
        if numeric_step_value > self.reduction_order['apcut']:
            self.filenumbers['fibermap'] = np.array([])
        if numeric_step_value > self.reduction_order['wavecalib']:
            self.filenumbers['coarse_comp'] = np.array([])
            self.filenumbers['fine_comp'] = np.array([])
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
            if len(self.filenumbers['bias']) > 0:
                return False
        if numeric_step ==  self.reduction_order['apcut']: # for app cutting
            if (self.instrument.cameras[0],'master','fibermap',None) not in self.all_hdus.keys():
                return False
        if numeric_step >  self.reduction_order['apcut']: # after app cutting
            if self.observations.nobs == 0:
                return False
            if len(self.filenumbers['fibermap']) > 0:
                return False
        if numeric_step >  self.reduction_order['wavecalib']: # after wavelength comparc_
            if len(self.filenumbers['coarse_comp']) > 0:
                return False
            if len(self.filenumbers['fine_comp']) > 0:
                return False
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
               self.comparcs[camera].run_initial_calibrations(skip_coarse = self.skip_coarse_calib,\
                                                              single_core = self.single_core)

            for camera in self.instrument.cameras:
                self.comparcs[camera].run_final_calibrations()
                self.comparcs[camera].create_calibration_default()

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

    def load_final_calib_hdus(self,camera):
        couldntfind = False
        if self.twostep_wavecomparc:
            filnum_ind = 1
        else:
            filnum_ind = 0
        if type(self.comparc_lampsf) in [list,np.ndarray]:
            name = 'full-' + '-'.join(self.comparc_lampsf)
        else:
            name = 'full-' + self.comparc_lampsf
        for pairnum, filnums in self.observations.comparc_pairs.items():
            filnum = filnums[filnum_ind]
            calib,thetype = self.filemanager.locate_calib_dict(name, camera, self.instrument.configuration,filnum,locate_type='full')
            if calib is None:
                couldntfind = True
                break
            elif thetype != 'full':
                print("Something went wrong when loading calibrations")
                print("Specified 'full' but got back {}".format(thetype))
                couldntfind = True
                break
            else:
                self.final_calibrated_hdulists[(camera,filnum)] = calib
                self.final_calibration_coefs[(camera,filnum)] = Table(calib['calib coefs'].data)
        if couldntfind:
            raise(IOError,"Couldn't find matching calibrations. Please make sure the step has been run fully")

    def get_final_wavelength_coefs(self):
        for camera in self.instrument.cameras:
            self.load_final_calib_hdus(camera)

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
        if len(self.final_calibration_coefs.keys()) == 0:
            self.get_final_wavelength_coefs()
        filnum = self.observations.comparc_fs[0]
        comparc_table = self.final_calibration_coefs[(cam,filnum)]
        #self.filemanager.load_calib_dict('default', cam, self.instrument.configuration)

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
        if len(self.final_calibration_coefs.keys()) == 0:
            self.get_final_wavelength_coefs()
        observation_keys = list(self.observations.observations.keys())
        nobs = len(observation_keys)
        middle_obs = observation_keys[nobs//2]

        ref_science_filnum,ccalib,fcalib,comparc_ind = self.observations.observations[middle_obs]
        ref_sci_hdu = self.all_hdus.pop((cam,ref_science_filnum,'science',None))
        ref_sci_data = Table(ref_sci_hdu.data)
        if self.twostep_wavecomparc:
            filnum = fcalib
        else:
            filnum = ccalib
        ref_calibrations = self.final_calibration_coefs[(cam,filnum)]
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
            sci_filnum, ccalib, fcalib, comparc_ind = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))
            sci_data = sci_hdu.data
            if self.twostep_wavecomparc:
                filnum = fcalib
            else:
                filnum = ccalib
            sci_calibrations = self.final_calibration_coefs[(cam,filnum)]
            for col in ref_sci_data.colnames:
                a, b, c, d, e, f = sci_calibrations[col]
                lams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                flux = np.array(sci_data[col])
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
        self.all_hdus[(cam,'combined_wavelengths','science',None)] = fits.BinTableHDU(data=ref_wave_table,name='wave')



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
        def gauss(lams, offset, mean, sig, amp):
            return offset + (amp * np.exp(-(lams - mean) * (lams - mean) / (2 * sig * sig))) / np.sqrt(
                2 * np.pi * sig * sig)

        def linear_gauss(lams, offset, linear, mean, sig, amp):
            return offset + linear * (lams - lams.min()) + (
                    amp * np.exp(-(lams - mean) * (lams - mean) / (2 * sig * sig))) / np.sqrt(
                2 * np.pi * sig * sig)

        def sumd_gauss(lams, offset, mean, sig1, sig2, amp1, amp2):
            return gauss(lams, offset, mean, sig1, amp1) + gauss(lams, offset, mean, sig2, amp2)

        def doublet_gauss(lams, offset, mean1, mean2, sig1, sig2, amp1, amp2):
            return gauss(lams, offset, mean1, sig1, amp1) + gauss(lams, offset, mean2, sig2, amp2)

        sigma_to_fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0))
        fwhm_to_sigma = 1.0 / sigma_to_fwhm
        nsigma = 4

        def to_sigma(s_right, s_left):
            return nsigma* fwhm_to_sigma * (s_right - s_left)
        from scipy.optimize import curve_fit
        from scipy.signal import medfilt

        if len(self.final_calibration_coefs.keys()) == 0:
            self.get_final_wavelength_coefs()
        observation_keys = list(self.observations.observations.keys())
        first_obs = observation_keys[0]
        sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
        npixels = len(self.all_hdus[(cam, sci_filnum, 'science', None)].data)
        pixels = np.arange(npixels).astype(np.float64)
        pix2 = pixels*pixels
        pix3 = pix2*pixels
        pix4 = pix3*pixels
        pix5 = pix4*pixels
        target_sky_pair = self.targeting_sky_pairs[cam]
        ##hack!
        scis = {}
        for obs in observation_keys:
            sci_filnum, ccalib, fcalib, comparc_ind = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))
            if self.twostep_wavecomparc:
                calib_filnum = fcalib
            else:
                calib_filnum = ccalib
            comparc_data = self.final_calibration_coefs[(cam,calib_filnum)]

            sci_data = Table(sci_hdu.data)
            out_sci_data = Table()
            sci_lams = {}
            skyfibs = list(target_sky_pair.values())

            cards = dict(sci_hdu.header)
            for key, val in cards.items():
                if key[:5].upper() == 'FIBER':
                    if val.strip(' \t').lower() == 'unplugged':
                        fib = '{}{}'.format(cam, key[5:])
                        if fib not in self.instrument.deadfibers:
                            skyfibs.append(fib)

            # fibnames = sci_data.colnames
            # fibernums = np.array([(int(fib[1])-1)*16+int(fib[2:]) for fib in fibnames])
            # first_fib = fibnames[np.argmin(fibernums)]
            # last_fib = fibnames[np.argmax(fibernums)]
            # midd_fibs = [fibnames[ind] for ind in np.argsort(np.abs(fibernums)-(5*16))[:2]]
            #
            # minlam = np.min([comparc_data[first_fib][0],comparc_data[last_fib][0]])
            #
            # a, b, c, d, e, f = comparc_data[midd_fibs[0]]
            # npixels = len(pixels)
            # maxlam1 = a + b * npixels + c * npixels*npixels + d * npixels*npixels*npixels + \
            #           e * npixels*npixels*npixels*npixels + f * npixels*npixels*npixels*npixels*npixels
            # a, b, c, d, e, f = comparc_data[midd_fibs[1]]
            # npixels = len(pixels)
            # maxlam2 = a + b * npixels + c * npixels*npixels + d * npixels*npixels*npixels + \
            #           e * npixels*npixels*npixels*npixels + f * npixels*npixels*npixels*npixels*npixels
            # maxlam = np.max([maxlam1,maxlam2])
            # del maxlam1,maxlam2

            skyfits = {}
            plt.figure()
            mins,maxs=[],[]
            for skyfib in skyfibs:
                a, b, c, d, e, f = comparc_data[skyfib]
                skylams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                skyflux = np.array(sci_data[skyfib])#medfilt(sci_data[skyfib] - medfilt(sci_data[skyfib], 371), 3)
                plt.plot(skylams, skyflux, label=skyfib,alpha=0.4)
                skyfit = CubicSpline(x=skylams, y=skyflux, extrapolate=False)
                skyfits[skyfib] = skyfit

                mins.append(skylams.min())
                maxs.append(skylams.max())

            skyllams = np.arange(np.min(mins),np.max(maxs),0.1).astype(np.float64)
            del mins,maxs

            master_skies = []
            meds = []
            for skyfib,skyfit in skyfits.items():
                outskyflux = skyfit(skyllams)
                # outskyflux[np.isnan(outskyflux)] = 0.
                med = np.nanmedian(outskyflux)
                meds.append(med)
                #corrected = smooth_and_dering(outskyflux)
                master_skies.append(outskyflux-med)

            # median_master_sky = np.median(master_skies, axis=0)
            # mean_master_sky = np.mean(master_skies, axis=0)
            # master_sky = np.zeros_like(mean_master_sky)
            # master_sky[:300] = mean_master_sky[:300]
            # master_sky[-300:] = mean_master_sky[-300:]
            # master_sky[300:-300] = median_master_sky[300:-300]
            master_sky = np.nanmedian(master_skies, axis=0)
            master_sky[np.isnan(master_sky)] = 0.
            medmed = np.median(meds)
            master_sky += medmed
            del meds,medmed

            masterfit = CubicSpline(x=skyllams, y=master_sky, extrapolate=False)
            plt.plot(skyllams, master_sky, 'k-',label='master',linewidth=4)
            plt.legend(loc='best')
            plt.show()
            nonzeros = np.where(master_sky>0.)[0]
            first_nonzero_lam = skyllams[nonzeros[0]]
            last_nonzero_lam = skyllams[nonzeros[-1]]
            del nonzeros

            for galfib,skyfib in target_sky_pair.items():
                a, b, c, d, e, f = comparc_data[galfib]
                gallams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                galflux = np.asarray(sci_data[galfib])
                galflux[np.isnan(galflux)] = 0.

                lamcut = np.where((gallams < first_nonzero_lam) | (gallams > last_nonzero_lam))[0]
                galflux[lamcut] = 0.
                galflux[np.isnan(galflux)] = 0.

                skyflux = masterfit(gallams)
                skyflux[np.isnan(skyflux)] = 0.
                skyflux[lamcut] = 0.

                gcont = medfilt(galflux, 371)
                scont = medfilt(skyflux, 371)
                gal_contsub = galflux - gcont
                sky_contsub = skyflux - scont

                # gal_contsub -= np.min(gal_contsub)
                # sky_contsub -= np.min(sky_contsub)
                s_peak_inds, s_peak_props = find_peaks(sky_contsub, height=(None, None), width=(0.5, 8), \
                                                       threshold=(None, None),
                                                       prominence=(sky_contsub.max() / 10, None), wlen=24)

                g_peak_inds, g_peak_props = find_peaks(gal_contsub, height=(30, None), width=(0.5, 8), \
                                                       threshold=(None, None),
                                                       prominence=(50, None), wlen=42)

                g_peak_inds_matched = []
                for peak in s_peak_inds:
                    ind = np.argmin(np.abs(gallams[g_peak_inds] - gallams[peak]))
                    g_peak_inds_matched.append(g_peak_inds[ind])

                g_peak_inds_matched = np.asarray(g_peak_inds_matched).astype(int)

                s_peak_fluxes = sky_contsub[s_peak_inds]
                g_peak_fluxes = gal_contsub[g_peak_inds_matched]
                # differences = g_peak_fluxes - s_peak_fluxes
                # normd_diffs = differences/s_peak_fluxes
                # median_normd_diff = np.median(normd_diffs)
                peak_ratio = g_peak_fluxes / s_peak_fluxes
                median_ratio = np.median(peak_ratio)
                # print(peak_ratio, median_ratio)

                ## Physically the gal+sky shouldn't be less than sky
                # if median_ratio < 1.0:
                #     median_ration = 1.0
                skyflux *= median_ratio
                scont = medfilt(skyflux, 371)
                sky_contsub = skyflux - scont


                s_peak_inds, s_peak_props = find_peaks(sky_contsub, height=(30, None), width=(0.5, 8), \
                                                       threshold=(None, None),
                                                       prominence=(50, None), wlen=42)


                remaining_sky = skyflux.copy()
                s_lefts = np.array(s_peak_props['left_ips']).astype(int)
                s_rights = np.array(s_peak_props['right_ips']).astype(int) + 1

                line_pairs = []
                skips = []
                for ii in range(len(s_lefts)):
                    if ii in skips:
                        continue
                    pair = {}

                    lam1 = gallams[s_peak_inds[ii]]
                    spacing1 = to_sigma(s_lefts[ii],s_rights[ii])
                    ind1 = np.argmin(np.abs(gallams[g_peak_inds] - lam1))

                    if np.abs(gallams[g_peak_inds[ind1]]-gallams[s_peak_inds[ii]])>3.0:
                        continue

                    if ii + 1 == len(s_lefts):
                        lam2 = None
                        spacing2 = None
                    else:
                        lam2 = gallams[s_peak_inds[ii + 1]]
                        spacing2 = to_sigma(s_rights[ii + 1] , s_lefts[ii + 1])

                    if lam2 is not None and (lam1+spacing1) > (lam2-spacing2):
                        pair['doublet'] = True

                        skips.append(ii+1)
                        ind2 = np.argmin(np.abs(gallams[g_peak_inds] - lam2))
                        pair['gal1'] = [g_peak_inds[ind1],g_peak_props['left_ips'][ind1],\
                                        g_peak_props['right_ips'][ind1] + 1, g_peak_props['peak_heights'][ind1],\
                                        g_peak_props['width_heights'][ind1]]

                        pair['gal2'] = [g_peak_inds[ind2],g_peak_props['left_ips'][ind2],\
                                        g_peak_props['right_ips'][ind2] + 1, g_peak_props['peak_heights'][ind2],\
                                        g_peak_props['width_heights'][ind2]]

                        pair['sky1'] = [s_peak_inds[ii], s_lefts[ii], \
                                        s_rights[ii] + 1, s_peak_props['peak_heights'][ii], \
                                        s_peak_props['width_heights'][ii]]

                        pair['sky2'] = [s_peak_inds[ii+1], s_lefts[ii+1], \
                                        s_rights[ii+1] + 1, s_peak_props['peak_heights'][ii+1], \
                                        s_peak_props['width_heights'][ii+1]]

                    else:
                        pair['doublet'] = False
                        pair['gal1'] = [g_peak_inds[ind1],g_peak_props['left_ips'][ind1],\
                                        g_peak_props['right_ips'][ind1] + 1, g_peak_props['peak_heights'][ind1],\
                                        g_peak_props['width_heights'][ind1]]
                        pair['sky1'] = [s_peak_inds[ii], s_lefts[ii], \
                                        s_rights[ii] + 1, s_peak_props['peak_heights'][ii], \
                                        s_peak_props['width_heights'][ii]]

                    line_pairs.append(pair)

                sky_smthd_contsub = np.convolve(sky_contsub,[1/15.,3/15.,7/15.,3/15.,1/15.],'same')
                gal_smthd_contsub = np.convolve(gal_contsub, [1 / 15., 3 / 15., 7 / 15., 3 / 15., 1 / 15.], 'same')
                for pair in line_pairs:
                    g1_peak, g1_left, g1_right, g1_height,g1_wh = pair['gal1']
                    s1_peak, s1_left, s1_right, s1_height,s1_wh = pair['sky1']
                    s_multi_sigma1 = to_sigma(s1_right,s1_left)
                    g_multi_sigma1 = to_sigma(g1_right,g1_left)
                    if int(g1_right)==len(pixels):
                        g1_right = len(pixels)-1
                    if int(s1_right) == len(pixels):
                        s1_right = len(pixels) - 1
                    if pair['doublet']:
                        g2_peak, g2_left, g2_right,g2_height,g2_wh = pair['gal2']
                        s2_peak, s2_left, s2_right,s2_height,s2_wh = pair['sky2']
                        s_multi_sigma2 = to_sigma(s2_right, s2_left)
                        g_multi_sigma2 = to_sigma(g2_right, g2_left)
                        if int(g2_right) == len(pixels):
                            g2_right = len(pixels) - 1
                        if int(s2_right) == len(pixels):
                            s2_right = len(pixels) - 1
                        lower_wave_ind = np.argmin(np.abs(gallams - (gallams[s1_peak] - s_multi_sigma1)))
                        upper_wave_ind = np.argmin(np.abs(gallams - (gallams[s2_peak] + s_multi_sigma2)))
                    else:
                        lower_wave_ind = np.argmin(np.abs(gallams - (gallams[s1_peak] - s_multi_sigma1)))
                        upper_wave_ind = np.argmin(np.abs(gallams - (gallams[s1_peak] + s_multi_sigma1)))+1


                    lams = gallams[lower_wave_ind:upper_wave_ind]
                    g_cutout = gal_contsub[lower_wave_ind:upper_wave_ind].copy()
                    s_cutout = sky_contsub[lower_wave_ind:upper_wave_ind].copy()
                    # nlams = len(lams)
                    # gauss_convolution = gauss(np.arange(nlams),0.,mean=nlams/2.,sig=nlams/4.,\
                    #                           amp=np.sqrt(2* np.pi * nlams * nlams / 16.))
                    # print(gauss_convolution)
                    nzeros = 5
                    g_cutout[:nzeros] = 0.+np.arange(nzeros)*np.median(g_cutout[nzeros:nzeros+2])/nzeros
                    g_cutout[-nzeros:] = np.median(g_cutout[-nzeros-2:-nzeros])-\
                                         np.arange(nzeros)*np.median(g_cutout[-nzeros-2:-nzeros])/nzeros
                    s_cutout[:nzeros] = 0. + np.arange(nzeros) * np.median(s_cutout[nzeros:nzeros + 2]) / nzeros
                    s_cutout[-nzeros:] = np.median(s_cutout[-nzeros - 2:-nzeros]) - \
                                         np.arange(nzeros) * np.median(s_cutout[-nzeros - 2:-nzeros]) / nzeros
                    g_cutout[np.isnan(g_cutout)] = 0.
                    s_cutout[np.isnan(s_cutout)] = 0.
                    if np.any(np.isinf(g_cutout)):
                        print("Infinite values detected in galaxy cutout!")
                        g_cutout[np.isinf(g_cutout)] = 1.0e5
                    if np.any(np.isinf(s_cutout)):
                        print("Infinite values detected in sky cutout!")
                        s_cutout[np.isinf(s_cutout)] = 1.0e5

                    #gauss_convolution
                    #s_cutout *= gauss_convolution
                    # print(s_p0)
                    # print(g_p0)
                    # print(bounds)

                    if pair['doublet']:
                        fitting_function = doublet_gauss

                        s_p0 = [0., gallams[s1_peak], gallams[s2_peak], s_multi_sigma1 / nsigma, \
                                s_multi_sigma2 / nsigma, s1_height, s2_height]
                        g_p0 = [0., gallams[g1_peak], gallams[g2_peak], g_multi_sigma1 / nsigma, \
                                g_multi_sigma2 / nsigma, g1_height, g2_height]
                        bounds = ([-100., min([gallams[int(s1_left)], gallams[int(g1_left)]]), \
                                   min([gallams[int(s2_left)], gallams[int(g2_left)]]), 0., 0., 0., 0.], \
                                  [min([0.8 * s1_height,0.8*s2_height]), \
                                   max([gallams[int(s1_right)], gallams[int(g1_right)]]), \
                                   max([gallams[int(s2_right)], gallams[int(g2_right)]]), \
                                   max(4., 1.2 * s_multi_sigma1 / nsigma, 1.2 * g_multi_sigma1 / nsigma), \
                                   max(4., 1.2 * s_multi_sigma2 / nsigma, 1.2 * g_multi_sigma2 / nsigma), \
                                   max([6 * s1_height, 6 * g1_height]), \
                                   max([6 * s2_height, 6 * g2_height])])
                    else:
                        fitting_function = gauss

                        s_p0 = [0., gallams[s1_peak], s_multi_sigma1 / nsigma, s1_height]
                        g_p0 = [0., gallams[g1_peak], g_multi_sigma1 / nsigma, g1_height]
                        bounds = ([-100., min([gallams[int(s1_left)], gallams[int(g1_left)]]), 0., 0.], \
                                  [0.8 * s1_height, \
                                   max([gallams[int(s1_right)], gallams[int(g1_right)]]), \
                                   max([4., 1.2 * s_multi_sigma1 / nsigma, 1.2 * g_multi_sigma1 / nsigma]), \
                                   max([6 * s1_height, 6 * g1_height])
                                   ])

                    b1,b2 = bounds
                    if np.any(np.array(s_p0)<np.array(b1)) or np.any(np.array(s_p0)>np.array(b2)):
                        print(b1)
                        print(b2)
                        print(s_p0)
                        print("stop")
                    if np.any(np.array(g_p0)<np.array(b1)) or np.any(np.array(g_p0)>np.array(b2)):
                        print(b1)
                        print(b2)
                        print(g_p0)
                        print("stop")
                    sfit_coefs, scovsing = curve_fit(fitting_function, lams, \
                                                     s_cutout, \
                                                     p0=s_p0,bounds=bounds, maxfev=10000)

                    gfit_coefs, gcovsing = curve_fit(fitting_function, lams, \
                                                     g_cutout, \
                                                     p0=g_p0, bounds=bounds, maxfev=10000)

                    s_normd_err = np.sqrt(np.sum(np.diagonal(scovsing)/(sfit_coefs*sfit_coefs)))
                    g_normd_err = np.sqrt(np.sum(np.diagonal(gcovsing) / (gfit_coefs * gfit_coefs)))
                    # print(len(lams),s_normd_err,sfit_coefs)

                    if s_normd_err > 20 and len(lams)>(nzeros*2+4+4):
                        szeros = nzeros+2
                        s_cutout[:szeros] = 0. + np.arange(szeros) * np.median(s_cutout[szeros:szeros + 2]) / szeros
                        s_cutout[-szeros:] = np.median(s_cutout[-szeros - 2:-szeros]) - \
                                             np.arange(szeros) * np.median(s_cutout[-szeros - 2:-szeros]) / szeros
                        sfit_coefs, scovsing = curve_fit(fitting_function, lams[2:-2], \
                                                         s_cutout[2:-2], \
                                                         p0=s_p0, bounds=bounds, maxfev=10000)
                        # print(len(lams), s_normd_err, sfit_coefs)
                    if g_normd_err > 20 and len(lams)>(nzeros*2+4+4):
                        gzeros = nzeros+2
                        g_cutout[:gzeros] = 0. + np.arange(gzeros) * np.median(g_cutout[gzeros:gzeros + 2]) / gzeros
                        g_cutout[-gzeros:] = np.median(g_cutout[-gzeros - 2:-gzeros]) - \
                                             np.arange(gzeros) * np.median(g_cutout[-gzeros - 2:-gzeros]) / nzeros

                        gfit_coefs, gcovsing = curve_fit(fitting_function, lams[2:-2], \
                                                         g_cutout[2:-2], \
                                                         p0=g_p0, bounds=bounds, maxfev=10000)

                    # transform = gfit_coefs  # use the fit from the galaxy spectrum
                    # transform[-1] = sfit_coefs[-1]  # switch amplitude to that of the sky model
                    # transform[0] = sfit_coefs[0]  # switch dc offset to that of sky model
                    # if pair['doublet']:
                    #     transform[-2] = sfit_coefs[-2]
                    # s_transformed_fit = fitting_function(lams, *transform)
                    #g1_peak, g1_left, g1_right, g1_height,g1_wh = pair['gal1']
                    #s1_peak, s1_left, s1_right, s1_height,s1_wh = pair['sky1']
                    # sleft = int(s1_peak)
                    # keep_going = True
                    # while keep_going:
                    #     if sky_smthd_contsub[sleft-1]<sky_smthd_contsub[sleft]:
                    #         sleft -= 1
                    #     elif sky_smthd_contsub[sleft-2]<sky_smthd_contsub[sleft]:
                    #         sleft -= 1
                    #     elif sky_smthd_contsub[sleft-3]<sky_smthd_contsub[sleft]:
                    #         sleft -= 1
                    #     else:
                    #         keep_going = False
                    #
                    # sright = int(s1_peak)
                    # keep_going = True
                    # while keep_going:
                    #     if sky_smthd_contsub[sright+1]<sky_smthd_contsub[sright]:
                    #         sright += 1
                    #     elif sky_smthd_contsub[sright+2]<sky_smthd_contsub[sright]:
                    #         sright += 1
                    #     elif sky_smthd_contsub[sright+3]<sky_smthd_contsub[sright]:
                    #         sright += 1
                    #     else:
                    #         keep_going = False
                    #
                    # gleft = int(g1_peak)
                    # keep_going = True
                    # while keep_going:
                    #     if gal_smthd_contsub[gleft - 1] < gal_smthd_contsub[gleft]:
                    #         gleft -= 1
                    #     elif gal_smthd_contsub[sleft - 2] < gal_smthd_contsub[gleft]:
                    #         gleft -= 1
                    #     elif gal_smthd_contsub[gleft - 3] < gal_smthd_contsub[gleft]:
                    #         gleft -= 1
                    #     else:
                    #         keep_going = False

                    #gleft = int(g1_peak)
                    itterleft = int(s1_peak)
                    keep_going = True
                    nextset = np.arange(1, 4).astype(int)
                    while keep_going:
                        if itterleft == 0:
                            g_select = False
                            s_select = False
                        elif itterleft > 2:
                            g_select = np.any(gal_smthd_contsub[itterleft - nextset] < gal_smthd_contsub[itterleft])
                            s_select = np.any(sky_smthd_contsub[itterleft - nextset] < sky_smthd_contsub[itterleft])
                        else:
                            to_start = itterleft
                            endcut = -3+to_start
                            g_select = np.any(
                                gal_smthd_contsub[itterleft - nextset[:endcut]] < gal_smthd_contsub[itterleft])
                            s_select = np.any(
                                sky_smthd_contsub[itterleft - nextset[:endcut]] < sky_smthd_contsub[itterleft])

                        over_zero_select = (
                                    (gal_smthd_contsub[itterleft] > -10.) & (sky_smthd_contsub[itterleft] > -10.))
                        if g_select and s_select and over_zero_select:
                            itterleft -= 1
                        else:
                            keep_going = False
                    sleft, gleft = int(itterleft), int(itterleft)

                    #gright = int(g1_peak)
                    itterright = int(s1_peak)
                    keep_going = True
                    nextset = np.arange(1, 4).astype(int)
                    while keep_going:
                        if (len(pixels) - itterright) == 1:
                            g_select = False
                            s_select = False
                        elif (len(pixels) - itterright) > 3:
                            g_select = np.any(gal_smthd_contsub[itterright+nextset] < gal_smthd_contsub[itterright])
                            s_select = np.any(sky_smthd_contsub[itterright+nextset] < sky_smthd_contsub[itterright])
                        else:
                            to_end = len(pixels) - itterright
                            endcut = -4+to_end
                            g_select = np.any(gal_smthd_contsub[itterright + nextset[:endcut]] < gal_smthd_contsub[itterright])
                            s_select = np.any(sky_smthd_contsub[itterright + nextset[:endcut]] < sky_smthd_contsub[itterright])

                        over_zero_select = ((gal_smthd_contsub[itterright] > -10.) & (sky_smthd_contsub[itterright] > -10.))
                        if g_select and s_select and over_zero_select:
                            itterright += 1
                        else:
                            keep_going = False
                    sright,gright = int(itterright),int(itterright)
                    #sleft, sright = sfit_coefs[1] - 2.5 * sfit_coefs[2], sfit_coefs[1] + 2.5 * sfit_coefs[2]
                    #gleft, gright = gfit_coefs[1] - 2.5 * gfit_coefs[2], gfit_coefs[1] + 2.5 * gfit_coefs[2]

                    # slower_wave_ind = np.argmin(np.abs(gallams - sleft))
                    # supper_wave_ind = np.argmin(np.abs(gallams - sright)) + 1
                    #
                    # glower_wave_ind = np.argmin(np.abs(gallams - gleft))
                    # gupper_wave_ind = np.argmin(np.abs(gallams - gright)) + 1

                    slower_wave_ind = sleft
                    supper_wave_ind = sright + 1

                    glower_wave_ind = gleft
                    gupper_wave_ind = gright + 1


                    if np.abs((sright-sleft)-(gright-gleft))> 6:
                        # if pair['doublet']:
                        #     speaks = [gallams[s1_peak],gallams[s2_peak]]
                        #     slefts = [gallams[int(s1_left)], gallams[int(s2_left)]]
                        #     srights = [gallams[int(s1_right)], gallams[int(s2_right)]]
                        #     sheights = [s1_height,s2_height]
                        #     swheights = [s1_wh, s2_wh]
                        #     gpeaks = [gallams[int(g1_peak)], gallams[int(g2_peak)]]
                        #     glefts = [gallams[int(g1_left)], gallams[int(g2_left)]]
                        #     grights = [gallams[int(g1_right)], gallams[int(g2_right)]]
                        #     gheights = [g1_height, g2_height]
                        #     gwheights = [g1_wh, g2_wh]
                        # else:
                        #     speaks,sheights,swheights = gallams[int(s1_peak)],s1_height,s1_wh
                        #     gpeaks, gheights, gwheights = gallams[int(g1_peak)], g1_height, g1_wh
                        #     slefts,glefts = gallams[int(s1_left)], gallams[int(g1_left)]
                        #     srights,grights = gallams[int(s1_right)], gallams[int(g1_right)]
                        # plt.figure()
                        # plt.plot(lams, gal_contsub[lower_wave_ind:upper_wave_ind],alpha=0.2,label='gal')
                        # plt.plot(gallams[slower_wave_ind:supper_wave_ind],sky_contsub[slower_wave_ind:supper_wave_ind],alpha=0.4,label='sky')
                        # plt.plot(lams, gal_smthd_contsub[lower_wave_ind:upper_wave_ind], alpha=0.2, label='smthd gal')
                        # plt.plot(gallams[slower_wave_ind:supper_wave_ind], sky_smthd_contsub[slower_wave_ind:supper_wave_ind],
                        #          alpha=0.4, label='smthd sky')
                        #
                        # plt.plot(speaks, sheights, 'k*', label='sky peak')
                        # plt.plot(slefts, swheights, 'k>')
                        # plt.plot(srights, swheights, 'k<')
                        #
                        # plt.plot(gpeaks, gheights, 'c*', label='gal peak')
                        # plt.plot(glefts, gwheights, 'c>')
                        # plt.plot(grights, gwheights, 'c<')
                        # # plt.plot(lams,gauss(lams,*gfit_coefs),label='galfit')
                        # plt.xlim(gallams[lower_wave_ind - 10], gallams[upper_wave_ind + 10])
                        # plt.legend(loc='best')
                        #
                        # plt.show()
                        print("bad")

                    g_distrib = gal_contsub[glower_wave_ind:gupper_wave_ind].copy()
                    min_g_distrib = g_distrib.min()
                    g_distrib = g_distrib-min_g_distrib+0.00001
                    # g_lams = gallams[glower_wave_ind:gupper_wave_ind]
                    # g_dlams = gallams[glower_wave_ind+1:gupper_wave_ind+1]-\
                    #           gallams[glower_wave_ind:gupper_wave_ind]
                    # integral_g = np.dot(g_distrib,g_dlams)
                    integral_g = np.sum(g_distrib)
                    normd_g_distrib = g_distrib / integral_g

                    s_distrib = sky_contsub[slower_wave_ind:supper_wave_ind].copy()
                    min_s_distrib = s_distrib.min()
                    s_distrib = s_distrib-min_s_distrib+0.00001
                    # s_lams = gallams[slower_wave_ind:supper_wave_ind]
                    # s_dlams = gallams[slower_wave_ind+1:supper_wave_ind+1]-\
                    #           gallams[slower_wave_ind:supper_wave_ind]
                    # integral_s = np.dot(s_distrib,s_dlams)
                    integral_s = np.sum(s_distrib)

                    if integral_s > (30.0+integral_g):
                        integral_s = (30.0+integral_g)

                    if integral_s > (1.1*integral_g):
                        integral_s = (1.1*integral_g)

                    if integral_g > (60.0+integral_s):
                        integral_s = integral_g - 60.0

                    sky_g_distrib = normd_g_distrib * integral_s
                    if len(sky_g_distrib)>3:
                        removedlineflux = np.convolve(gal_contsub[glower_wave_ind:gupper_wave_ind].copy() - sky_g_distrib,\
                                    [1 / 5., 3 / 5., 1 / 5.], 'same')
                    else:
                        removedlineflux = gal_contsub[glower_wave_ind:gupper_wave_ind].copy() - sky_g_distrib

                    if np.any((gal_contsub[glower_wave_ind:gupper_wave_ind] - sky_g_distrib)<(-60)) and \
                            np.all((gal_contsub[glower_wave_ind:gupper_wave_ind])>(-60)):
                        # if pair['doublet']:
                        #     speaks = [gallams[s1_peak],gallams[s2_peak]]
                        #     slefts = [gallams[int(s1_left)], gallams[int(s2_left)]]
                        #     srights = [gallams[int(s1_right)], gallams[int(s2_right)]]
                        #     sheights = [s1_height,s2_height]
                        #     swheights = [s1_wh, s2_wh]
                        #     gpeaks = [gallams[int(g1_peak)], gallams[int(g2_peak)]]
                        #     glefts = [gallams[int(g1_left)], gallams[int(g2_left)]]
                        #     grights = [gallams[int(g1_right)], gallams[int(g2_right)]]
                        #     gheights = [g1_height, g2_height]
                        #     gwheights = [g1_wh, g2_wh]
                        # else:
                        #     speaks,sheights,swheights = gallams[int(s1_peak)],s1_height,s1_wh
                        #     gpeaks, gheights, gwheights = gallams[int(g1_peak)], g1_height, g1_wh
                        #     slefts,glefts = gallams[int(s1_left)], gallams[int(g1_left)]
                        #     srights,grights = gallams[int(s1_right)], gallams[int(g1_right)]
                        # plt.figure()
                        # plt.plot(gallams[glower_wave_ind:gupper_wave_ind], gal_contsub[glower_wave_ind:gupper_wave_ind] - sky_g_distrib, label='new sub',
                        #          alpha=0.4)
                        # plt.plot(gallams[glower_wave_ind:gupper_wave_ind],sky_g_distrib,alpha=0.4,label='transformed sky')
                        # plt.plot(lams, gal_contsub[lower_wave_ind:upper_wave_ind],alpha=0.2,label='gal')
                        # plt.plot(gallams[slower_wave_ind:supper_wave_ind],s_distrib,alpha=0.4,label='sky')
                        #
                        # plt.plot(speaks, sheights, 'k*', label='sky peak')
                        # plt.plot(slefts, swheights, 'k>')
                        # plt.plot(srights, swheights, 'k<')
                        #
                        # plt.plot(gpeaks, gheights, 'c*', label='gal peak')
                        # plt.plot(glefts, gwheights, 'c>')
                        # plt.plot(grights, gwheights, 'c<')
                        # # plt.plot(lams,gauss(lams,*gfit_coefs),label='galfit')
                        # plt.xlim(gallams[lower_wave_ind - 10], gallams[upper_wave_ind + 10])
                        # plt.legend(loc='best')
                        #
                        # plt.show()
                        print("That didn't go well")


                    doplots = False
                    if doplots:
                        if pair['doublet']:
                            speaks = [gallams[s1_peak],gallams[s2_peak]]
                            slefts = [gallams[int(s1_left)], gallams[int(s2_left)]]
                            srights = [gallams[int(s1_right)], gallams[int(s2_right)]]
                            sheights = [s1_height,s2_height]
                            swheights = [s1_wh, s2_wh]
                            gpeaks = [gallams[int(g1_peak)], gallams[int(g2_peak)]]
                            glefts = [gallams[int(g1_left)], gallams[int(g2_left)]]
                            grights = [gallams[int(g1_right)], gallams[int(g2_right)]]
                            gheights = [g1_height, g2_height]
                            gwheights = [g1_wh, g2_wh]
                        else:
                            speaks,sheights,swheights = gallams[int(s1_peak)],s1_height,s1_wh
                            gpeaks, gheights, gwheights = gallams[int(g1_peak)], g1_height, g1_wh
                            slefts,glefts = gallams[int(s1_left)], gallams[int(g1_left)]
                            srights,grights = gallams[int(s1_right)], gallams[int(g1_right)]

                        sfit_flux = fitting_function(lams, *sfit_coefs)
                        plt.subplots(1, 3)

                        plt.subplot(131)
                        plt.plot(lams,s_cutout, label='sky', alpha=0.4)
                        plt.plot(speaks, sheights, 'k*', label='peaks')
                        plt.plot(slefts, swheights, 'k>')
                        plt.plot(srights, swheights, 'k<')
                        plt.plot(lams, sfit_flux, label='skyfit')
                        plt.xlim(gallams[lower_wave_ind ], gallams[upper_wave_ind])
                        plt.legend(loc='best')

                        gfit_flux = fitting_function(lams, *gfit_coefs)
                        plt.subplot(132)
                        plt.plot(lams,g_cutout, label='gal', alpha=0.4)
                        plt.plot(gpeaks, gheights, 'k*', label='peaks')
                        plt.plot(glefts, gwheights, 'k>')
                        plt.plot(grights, gwheights, 'k<')
                        plt.plot(lams, gfit_flux, label='galfit')
                        plt.xlim(gallams[lower_wave_ind], gallams[upper_wave_ind])
                        ymin,ymax = plt.ylim()
                        plt.legend(loc='best')

                        plt.subplot(133)
                        plt.plot(gallams[glower_wave_ind:gupper_wave_ind], gal_contsub[glower_wave_ind:gupper_wave_ind] - sky_g_distrib, label='new sub',
                                 alpha=0.4)
                        plt.plot(gallams[glower_wave_ind:gupper_wave_ind], removedlineflux, label='new smth sub',  alpha=0.4)

                        plt.plot(gallams[glower_wave_ind:gupper_wave_ind],sky_g_distrib,alpha=0.4,label='transformed sky')
                        plt.plot(lams, gal_contsub[lower_wave_ind:upper_wave_ind],alpha=0.2,label='gal')
                        plt.plot(gallams[slower_wave_ind:supper_wave_ind],s_distrib,alpha=0.4,label='sky')

                        plt.plot(speaks, sheights, 'k*', label='sky peak')
                        plt.plot(slefts, swheights, 'k>')
                        plt.plot(srights, swheights, 'k<')

                        plt.plot(gpeaks, gheights, 'c*', label='gal peak')
                        plt.plot(glefts, gwheights, 'c>')
                        plt.plot(grights, gwheights, 'c<')
                        # plt.plot(lams,gauss(lams,*gfit_coefs),label='galfit')
                        plt.xlim(gallams[lower_wave_ind - 10], gallams[upper_wave_ind + 10])
                        min1,min2 = plt.ylim()
                        plt.ylim(min1,ymax)
                        plt.legend(loc='best')

                        plt.show()
                        aaaa=2
                    # 1/np.sqrt(2*np.pi*sig*sig)
                    # print(*gfit_coefs,*sfit_coefs)
                    gal_contsub[glower_wave_ind:gupper_wave_ind] = removedlineflux
                    ## remove the subtracted sky from that remaining in the skyflux
                    remaining_sky[lower_wave_ind:upper_wave_ind] = scont[lower_wave_ind:upper_wave_ind]

                outgal = gal_contsub + gcont - remaining_sky
                if np.any(outgal > 1000.):
                    plt.subplots(1, 2)
                    plt.subplot(121)
                    plt.plot(gallams, outgal, label='output', alpha=0.4)
                    plt.plot(gallams, remaining_sky, label='remaining_sky', alpha=0.4)
                    plt.plot(gallams, gcont, label='gcont', alpha=0.4)
                    plt.legend(loc='best')
                    ymin, ymax = plt.ylim()
                    plt.subplot(122)
                    plt.plot(gallams, skyflux, label='sky', alpha=0.4)
                    plt.plot(gallams, galflux, label='gal', alpha=0.4)
                    plt.plot(gallams, outgal, label='outgal', alpha=0.4)
                    # plt.plot(gallams, gal_contsub, label='gal_contsub',alpha=0.4)
                    plt.legend(loc='best')
                    #plt.ylim(ymin, ymax)
                    plt.show()
                    print("bad")

                doplots = True
                if doplots:
                    plt.subplots(1,2)
                    plt.subplot(121)
                    plt.plot(gallams,outgal,label='output',alpha=0.4)
                    plt.plot(gallams, remaining_sky, label='remaining_sky',alpha=0.4)
                    plt.plot(gallams, gcont, label='gcont',alpha=0.4)
                    plt.legend(loc='best')
                    ymin,ymax = plt.ylim()
                    plt.subplot(122)
                    plt.plot(gallams,galflux-skyflux,label='basic',alpha=0.4)
                    plt.plot(gallams, outgal, label='gcont',alpha=0.4)
                    #plt.plot(gallams, gal_contsub, label='gal_contsub',alpha=0.4)
                    plt.legend(loc='best')
                    plt.ylim(ymin,ymax)
                    plt.show()
                doplots = False
                out_sci_data.add_column(Table.Column(name=galfib,data=outgal))
                sci_lams[galfib] = gallams


            scis[obs] = (out_sci_data.copy(),sci_lams.copy(),sci_filnum)
            self.all_hdus[(cam, sci_filnum, 'science', None)] = fits.BinTableHDU(data=out_sci_data,header=sci_hdu.header,name='flux')





    def subtract_skies_old(self, cam):
        from quickreduce_funcs import smooth_and_dering
        if len(self.final_calibration_coefs.keys()) == 0:
            self.get_final_wavelength_coefs()
        observation_keys = list(self.observations.observations.keys())
        first_obs = observation_keys[0]
        sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
        npixels = len(self.all_hdus[(cam, sci_filnum, 'science', None)].data)
        pixels = np.arange(npixels).astype(np.float64)
        pix2 = pixels * pixels
        pix3 = pix2 * pixels
        pix4 = pix3 * pixels
        pix5 = pix4 * pixels
        target_sky_pair = self.targeting_sky_pairs[cam]
        ##hack!
        scis = {}
        for obs in observation_keys:
            sci_filnum, ccalib, fcalib, comparc_ind = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))
            if self.twostep_wavecomparc:
                calib_filnum = fcalib
            else:
                calib_filnum = ccalib
            comparc_data = self.final_calibration_coefs[(cam, calib_filnum)]
            sci_data = Table(sci_hdu.data)
            out_sci_data = Table()
            sci_lams = {}
            skyfibs = np.unique(list(target_sky_pair.values()))
            a, b, c, d, e, f = comparc_data['{}214'.format(cam)]
            skyllams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5

            master_skies = []
            skyfits = {}
            plt.figure()
            for skyfib in skyfibs:
                a, b, c, d, e, f = comparc_data[skyfib]
                skylams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                skyflux = medfilt(sci_data[skyfib] - medfilt(sci_data[skyfib], 371), 3)
                plt.plot(skylams, skyflux, label=skyfib)
                skyfit = CubicSpline(x=skylams, y=skyflux, extrapolate=False)
                skyfits[skyfib] = skyfit
                outskyflux = skyfit(skyllams)
                corrected = smooth_and_dering(outskyflux)
                master_skies.append(corrected)

            master_sky = np.median(master_skies, axis=0)
            masterfit = CubicSpline(x=skyllams, y=master_sky, extrapolate=False)
            plt.plot(skyllams, master_sky, label='master')
            plt.legend(loc='best')

            for galfib, skyfib in target_sky_pair.items():
                a, b, c, d, e, f = comparc_data[galfib]
                gallams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                galflux = sci_data[galfib]
                galflux[np.isnan(galflux)] = 0.
                continuum = medfilt(sci_data[galfib], 371)
                medgal_contsubd = medfilt(galflux - continuum, 3)
                subd_galflux = galflux - continuum

                skyfit = skyfits[skyfib]
                outskyflux = masterfit(gallams)  # skyfit(gallams)
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
                            match = np.max(match)  # match[np.argmax(gheights[match])]
                        sky_height = heights[ii]
                        left = peak_lefts[ii]
                        right = peak_rights[ii]
                        if ((gpeak_rights[match] - right) > 4) and (ii < npeaks - 1):
                            if np.abs(gpeak_rights[match] - peak_rights[ii + 1]) < 4:
                                # print("changed rights")
                                right = int(peak_rights[ii + 1])
                                sky_height = np.max([heights[ii], heights[ii + 1]])
                        elif ((left - gpeak_lefts[match]) > 4) and (ii > 0):
                            if np.abs(peak_lefts[ii - 1] - gpeak_lefts[match]) < 4:
                                # print("changed lefts")
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
                            if np.std(test_sub) > (6 * np.std(doctored[200:200 + 4 * (right - left)])):
                                # slope = ((doctored[right]-doctored[left])/(right-left))
                                # test_sub = np.arange(right-left)*slope+doctored[left]
                                test_sub = np.ones(right - left) * np.sort(doctored[left:right])[((right - left) // 10)]
                            doctored[left:right] = test_sub

                out_sci_data.add_column(Table.Column(name=galfib, data=(doctored + continuum)))
                sci_lams[galfib] = gallams
                plt.figure()
                plt.plot(gallams, sci_data[galfib], alpha=0.4, label='orig')
                plt.plot(gallams, doctored + continuum, alpha=0.4, label='doctored')
                plt.plot(gallams, sci_data[galfib] - outskyflux, alpha=0.4, label='basic')
                plt.plot(gallams, outskyflux, alpha=0.4, label='sky')
                plt.legend(loc='best')
                plt.show()
            scis[obs] = (out_sci_data.copy(), sci_lams.copy(), sci_filnum)
            self.all_hdus[(cam, sci_filnum, 'science', None)] = fits.BinTableHDU(data=out_sci_data,
                                                                                 header=sci_hdu.header, name='flux')

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