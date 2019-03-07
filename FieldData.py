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
        if self.reduction_order[startstep]>self.reduction_order['wavecalib']:
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
            self.filenumbers['twiflat'] = np.array([])
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
        self.filemanager.write_all_filedata(self.fit_data,self.step)
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
            data = hdu.data.copy()
            hdr = hdu.header
            data_arr = np.ndarray(shape=(len(final_table.colnames),len(final_table.columns[0])))
            flt_data_arr = np.ndarray(shape=(len(final_table.colnames),len(final_table.columns[0])))
            flats_arr = np.ndarray(shape=(len(final_table.colnames),len(final_table.columns[0])))
            outdata = OrderedDict()
            #plt.figure()
            names = []
            for tet in range(1,9):
                for fib in range(1,17):
                    if cam == 'b':
                        name = '{}{:d}{:02d}'.format(cam,9-tet,fib)
                    else:
                        name = '{}{:d}{:02d}'.format(cam, tet, fib)
                    if name in final_table.colnames:
                        names.append(name)
            for ii,col in enumerate(names):
                outdata[col] = data[col]/final_table[col]
                #plt.plot(np.arange(len(data[col])),data[col])
                data_arr[ii, :] = data[col]
                flt_data_arr[ii,:] = outdata[col]
                flats_arr[ii,:] = final_table[col]
            #plt.show()
            plt.subplots(2,2)
            plt.subplot(221)
            im = plt.imshow(data_arr-data_arr.min()+1., aspect='auto', origin='lower-left')
            plt.colorbar()
            clow,chigh = im.get_clim()
            plt.title("Original cam:{} filnm:{}".format(cam, filnum))
            plt.subplot(222)
            plt.imshow(flats_arr-flats_arr.min()+1., aspect='auto', origin='lower-left')
            plt.title("Flat cam:{} filnm:{}".format(cam,filnum))

            plt.colorbar()
            plt.subplot(223)
            plt.imshow(flt_data_arr-flt_data_arr.min()+1., aspect='auto', origin='lower-left')
            plt.title("Flattened cam:{} filnm:{}".format(cam,filnum))
            plt.clim(clow, chigh)
            plt.colorbar()
            plt.subplot(224)
            plt.imshow(np.log(flt_data_arr-flt_data_arr.min()+1.), aspect='auto', origin='lower-left')
            plt.title("Log Flattened cam:{} filnm:{}".format(cam,filnum))
            plt.colorbar()
            plt.show()
            self.all_hdus[(cam,filnum,'science',None)] = fits.BinTableHDU(data=Table(outdata),header=hdr,name='flux')

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
        for obs in observation_keys[1:]:
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

            #HACK!
            skyfibs = comparc_data.colnames
            for skyfib in skyfibs:
                a, b, c, d, e, f = comparc_data[skyfib]
                skylams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                skyflux = np.array(sci_data[skyfib])#medfilt(sci_data[skyfib] - medfilt(sci_data[skyfib], 371), 3)
                skyflux[np.isnan(skyflux)] = 0.
                plt.plot(skylams, skyflux, label=skyfib,alpha=0.4)
                skyfit = CubicSpline(x=skylams, y=skyflux, extrapolate=False)
                skyfits[skyfib] = skyfit

                mins.append(skylams.min())
                maxs.append(skylams.max())

            skyllams = np.arange(np.min(mins),np.max(maxs),0.1).astype(np.float64)
            del mins,maxs

            master_skies = []
            meds = []
            for skyfib in skyfibs:
                skyfit = skyfits[skyfib]
                outskyflux = skyfit(skyllams)
                outskyflux[np.isnan(outskyflux)] = 0.
                # outskyflux[np.isnan(outskyflux)] = 0.
                med = np.nanmedian(outskyflux)
                meds.append(med)
                #corrected = smooth_and_dering(outskyflux)
                master_skies.append(outskyflux/med)

            # median_master_sky = np.median(master_skies, axis=0)
            # mean_master_sky = np.mean(master_skies, axis=0)
            # master_sky = np.zeros_like(mean_master_sky)
            # master_sky[:300] = mean_master_sky[:300]
            # master_sky[-300:] = mean_master_sky[-300:]
            # master_sky[300:-300] = median_master_sky[300:-300]
            master_sky = np.nanmedian(master_skies, axis=0)
            master_sky[np.isnan(master_sky)] = 0.
            medmed = np.median(meds)
            master_sky *= medmed
            del meds,medmed

            masterfit = CubicSpline(x=skyllams, y=master_sky, extrapolate=False)
            plt.plot(skyllams, master_sky, 'k-',label='master',linewidth=4)
            plt.legend(loc='best')
            plt.show()
            nonzeros = np.where(master_sky>0.)[0]
            first_nonzero_lam = skyllams[nonzeros[0]]
            last_nonzero_lam = skyllams[nonzeros[-1]]
            del nonzeros

            bin_by_tet = True
            if bin_by_tet:
                median_arrays1 = {int(i): [] for i in range(1, 9)}
                median_arrays2 = {int(i): [] for i in range(1, 9)}

                for ii, name in enumerate(skyfibs):
                    tet = int(name[1])
                    fib = int(name[2:4])
                    if fib > 8:
                        median_arrays2[tet].append(master_skies[ii])
                    else:
                        median_arrays1[tet].append(master_skies[ii])

                plt.figure()
                for (key1, arr1), (key2, arr2) in zip(median_arrays1.items(), median_arrays2.items()):
                    if len(arr1)>0:
                        med1 = np.nanmedian(np.asarray(arr1), axis=0)
                        plt.plot(skyllams, med1, label="{}_1".format(key1), alpha=0.4)
                    if len(arr2)>0:
                        med2 = np.nanmedian(np.asarray(arr2), axis=0)
                        plt.plot(skyllams, med2, label="{}_2".format(key2), alpha=0.4)
                plt.legend(loc='best')
                plt.show()

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


                s_peak_inds, s_peak_props = find_peaks(sky_contsub, height=(sky_contsub.max() / 10, None), width=(0.5, 8), \
                                                       threshold=(None, None),
                                                       prominence=(sky_contsub.max() / 5, None), wlen=24)

                g_peak_inds, g_peak_props = find_peaks(sky_contsub, height=(gal_contsub.max() / 10, None), width=(0.5, 8), \
                                                       threshold=(None, None),
                                                       prominence=(gal_contsub.max() / 5, None), wlen=24)

                g_peak_inds_matched = []
                for peak in s_peak_inds:
                    ind = np.argmin(np.abs(gallams[g_peak_inds] - gallams[peak]))
                    g_peak_inds_matched.append(g_peak_inds[ind])

                g_peak_inds_matched = np.asarray(g_peak_inds_matched).astype(int)

                # s_peak_fluxes = sky_contsub[s_peak_inds]
                # g_peak_fluxes = gal_contsub[g_peak_inds_matched]
                s_peak_fluxes = skyflux[s_peak_inds]
                g_peak_fluxes = galflux[g_peak_inds_matched]
                # differences = g_peak_fluxes - s_peak_fluxes
                # normd_diffs = differences/s_peak_fluxes
                # median_normd_diff = np.median(normd_diffs)
                peak_ratio = g_peak_fluxes / s_peak_fluxes
                median_ratio = np.median(peak_ratio)
                # print(peak_ratio, median_ratio)

                ## Physically the gal+sky shouldn't be less than sky
                # if median_ratio < 1.0:
                #     median_ration = 1.0
                sky_contsub *= median_ratio

                s_peak_inds, s_peak_props = find_peaks(sky_contsub, height=(30, None), width=(0.1, 10), \
                                                       threshold=(None, None),
                                                       prominence=(10, None), wlen=101)
                g_peak_inds, g_peak_props = find_peaks(gal_contsub, height=(30, None), width=(0.1, 10), \
                                                       threshold=(None, None),
                                                       prominence=(10, None), wlen=101)

                continuum_ratio = np.median(gcont[300:800]/scont[300:800])-0.1
                #continuum_ratio = np.median(galflux[:400] / skyflux[:400])-0.1
                remaining_sky = medfilt(continuum_ratio*skyflux.copy(),3)
                scont = continuum_ratio*scont
                outgal = gal_contsub.copy()
                line_pairs = []
                for ii in range(len(s_peak_inds)):
                    pair = {}

                    lam1 = gallams[s_peak_inds[ii]]
                    ind1 = np.argmin(np.abs(gallams[g_peak_inds] - lam1))

                    if np.abs(gallams[g_peak_inds[ind1]]-gallams[s_peak_inds[ii]])>3.0:
                        continue

                    pair['gal'] = {'peak':g_peak_inds[ind1],'left':g_peak_props['left_ips'][ind1],\
                                    'right':g_peak_props['right_ips'][ind1] + 1, 'height':g_peak_props['peak_heights'][ind1],\
                                    'wheight':g_peak_props['width_heights'][ind1]}
                    pair['sky'] = {'peak':s_peak_inds[ii], 'left':s_peak_props['left_ips'][ii], \
                                    'right':s_peak_props['right_ips'][ii] + 1, 'height':s_peak_props['peak_heights'][ii], \
                                    'wheight':s_peak_props['width_heights'][ii]}

                    line_pairs.append(pair)

                sky_smthd_contsub = np.convolve(sky_contsub,[1/15.,3/15.,7/15.,3/15.,1/15.],'same')
                gal_smthd_contsub = np.convolve(gal_contsub, [1 / 15., 3 / 15., 7 / 15., 3 / 15., 1 / 15.], 'same')
                for pair in line_pairs:
                    g1_peak = pair['gal']['peak']
                    s1_peak = pair['sky']['peak']

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

                        over_zero_select = ((gal_smthd_contsub[itterleft] > -10.) & (sky_smthd_contsub[itterleft] > -10.))
                        if g_select and s_select and over_zero_select:
                            itterleft -= 1
                        else:
                            keep_going = False

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

                    slower_wave_ind = int(itterleft)
                    supper_wave_ind = int(itterright) + 1
                    extended_lower_ind = np.clip(slower_wave_ind - 10, 0, npixels - 1)
                    extended_upper_ind = np.clip(supper_wave_ind + 10, 0, npixels - 1)

                    g_distrib = gal_contsub[slower_wave_ind:supper_wave_ind].copy()
                    min_g_distrib = g_distrib.min()
                    g_distrib = g_distrib-min_g_distrib+0.00001
                    # g_lams = gallams[slower_wave_ind:supper_wave_ind]
                    # g_dlams = gallams[slower_wave_ind+1:supper_wave_ind+1]-\
                    #           gallams[slower_wave_ind:supper_wave_ind]
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
                        removedlineflux = np.convolve(gal_contsub[slower_wave_ind:supper_wave_ind].copy() - sky_g_distrib,\
                                    [1 / 5., 3 / 5., 1 / 5.], 'same')
                    else:
                        removedlineflux = gal_contsub[slower_wave_ind:supper_wave_ind].copy() - sky_g_distrib

                    doplots = False
                    dips_low = np.any((gal_contsub[slower_wave_ind:supper_wave_ind] - sky_g_distrib) < (-60))
                    all_above = np.all((gal_contsub[slower_wave_ind:supper_wave_ind]) > (-60))
                    if doplots:# or (dips_low and all_above):
                        speaks,sheights,swheights = gallams[int(s1_peak)],pair['sky']['height'],pair['sky']['wheight']
                        gpeaks, gheights, gwheights = gallams[int(g1_peak)], pair['gal']['height'], pair['gal']['wheight']
                        slefts,glefts = gallams[int(pair['sky']['left'])], gallams[int(pair['gal']['left'])]
                        srights,grights = gallams[int(pair['sky']['right'])], gallams[int(pair['gal']['right'])]

                        plt.subplots(1, 3)

                        plt.subplot(131)
                        plt.plot(gallams[extended_lower_ind:extended_upper_ind],\
                                 sky_contsub[extended_lower_ind:extended_upper_ind], label='sky', alpha=0.4)
                        plt.plot(speaks, sheights, 'k*', label='peaks')
                        plt.plot(slefts, swheights, 'k>')
                        plt.plot(srights, swheights, 'k<')
                        plt.xlim(gallams[extended_lower_ind],gallams[extended_upper_ind])

                        ymin,ymax = plt.ylim()
                        plt.vlines(gallams[slower_wave_ind],ymin,ymax)
                        plt.vlines(gallams[supper_wave_ind-1],ymin,ymax)
                        plt.legend(loc='best')

                        plt.subplot(132)
                        plt.plot(gallams[extended_lower_ind:extended_upper_ind],\
                                 gal_contsub[extended_lower_ind:extended_upper_ind], label='gal', alpha=0.4)
                        plt.plot(gpeaks, gheights, 'k*', label='peaks')
                        plt.plot(glefts, gwheights, 'k>')
                        plt.plot(grights, gwheights, 'k<')
                        plt.xlim(gallams[extended_lower_ind], gallams[extended_upper_ind])
                        ymin,ymax = plt.ylim()
                        plt.vlines(gallams[slower_wave_ind],ymin,ymax)
                        plt.vlines(gallams[supper_wave_ind-1],ymin,ymax)
                        ymin,ymax = plt.ylim()
                        plt.legend(loc='best')

                        plt.subplot(133)
                        plt.plot(gallams[slower_wave_ind:supper_wave_ind], gal_contsub[slower_wave_ind:supper_wave_ind] - sky_g_distrib, label='new sub',
                                 alpha=0.4)
                        plt.plot(gallams[slower_wave_ind:supper_wave_ind], removedlineflux, label='new smth sub',  alpha=0.4)

                        plt.plot(gallams[slower_wave_ind:supper_wave_ind],sky_g_distrib,alpha=0.4,label='transformed sky')
                        plt.plot(gallams[extended_lower_ind:extended_upper_ind], gal_contsub[extended_lower_ind:extended_upper_ind],alpha=0.2,label='gal')
                        plt.plot(gallams[slower_wave_ind:supper_wave_ind],s_distrib,alpha=0.4,label='sky')

                        plt.plot(speaks, sheights, 'k*', label='sky peak')
                        plt.plot(slefts, swheights, 'k>')
                        plt.plot(srights, swheights, 'k<')

                        plt.plot(gpeaks, gheights, 'c*', label='gal peak')
                        plt.plot(glefts, gwheights, 'c>')
                        plt.plot(grights, gwheights, 'c<')
                        # plt.plot(lams,gauss(lams,*gfit_coefs),label='galfit')
                        plt.xlim(gallams[extended_lower_ind],gallams[extended_upper_ind])
                        ymin,ymax = plt.ylim()
                        plt.vlines(gallams[slower_wave_ind],ymin,ymax)
                        plt.vlines(gallams[supper_wave_ind-1],ymin,ymax)
                        #min1,min2 = plt.ylim()
                        #plt.ylim(min1,ymax)
                        plt.legend(loc='best')

                        plt.show()
                        if (dips_low and all_above):
                            print("That didn't go well")
                    # 1/np.sqrt(2*np.pi*sig*sig)
                    # print(*gfit_coefs,*sfit_coefs)
                    outgal[slower_wave_ind:supper_wave_ind] = removedlineflux
                    ## remove the subtracted sky from that remaining in the skyflux
                    remaining_sky[slower_wave_ind:supper_wave_ind] = scont[slower_wave_ind:supper_wave_ind]

                outgal = outgal + gcont - remaining_sky
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

                doplots = False
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
                    plt.plot(gallams, outgal, label='outgal',alpha=0.4)
                    #plt.plot(gallams, gal_contsub, label='gal_contsub',alpha=0.4)
                    plt.legend(loc='best')
                    #plt.ylim(ymin,ymax)
                    plt.show()
                doplots = False
                out_sci_data.add_column(Table.Column(name=galfib,data=outgal))
                sci_lams[galfib] = gallams


            scis[obs] = (out_sci_data.copy(),sci_lams.copy(),sci_filnum)
            self.all_hdus[(cam, sci_filnum, 'science', None)] = fits.BinTableHDU(data=out_sci_data,header=sci_hdu.header,name='flux')

    def investigate_app_naming(self, cam):

        from scipy.signal import medfilt

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
        for obs in observation_keys[1:]:
            sci_filnum, ccalib, fcalib, comparc_ind = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))

            if obs == observation_keys[1]:
                combined_table = Table(sci_hdu.data).copy()
                if self.twostep_wavecomparc:
                    calib_filnum = fcalib
                else:
                    calib_filnum = ccalib
                comparc_data = self.final_calibration_coefs[(cam, calib_filnum)]
                header = sci_hdu.header
            else:
                itter_tab = Table(sci_hdu.data)
                for col in itter_tab.colnames:
                    combined_table[col] += itter_tab[col]


        sci_data = combined_table
        skyfibs = list(np.unique((list(target_sky_pair.values()))))

        cards = dict(header)
        for key, val in cards.items():
            if key[:5].upper() == 'FIBER':
                if val.strip(' \t').lower() == 'unplugged':
                    fib = '{}{}'.format(cam, key[5:])
                    if fib not in self.instrument.deadfibers:
                        skyfibs.append(fib)

        skyfits = {}
        plt.figure()
        mins, maxs = [], []

        # HACK!
        skyfibs = comparc_data.colnames
        for skyfib in skyfibs:
            a, b, c, d, e, f = comparc_data[skyfib]
            skylams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
            skyflux = np.array(
                sci_data[skyfib])  # medfilt(sci_data[skyfib] - medfilt(sci_data[skyfib], 371), 3)
            skyflux[np.isnan(skyflux)] = 0.
            plt.plot(skylams, skyflux, label=skyfib, alpha=0.4)
            skyfit = CubicSpline(x=skylams, y=skyflux, extrapolate=False)
            skyfits[skyfib] = skyfit

            mins.append(skylams.min())
            maxs.append(skylams.max())

        skyllams = np.arange(np.min(mins), np.max(maxs), 0.1).astype(np.float64)
        del mins, maxs

        master_skies = []
        meds = []
        for skyfib in skyfibs:
            skyfit = skyfits[skyfib]
            outskyflux = skyfit(skyllams)
            outskyflux[np.isnan(outskyflux)] = 0.
            # outskyflux[np.isnan(outskyflux)] = 0.
            med = np.nanmedian(outskyflux)
            meds.append(med)
            # corrected = smooth_and_dering(outskyflux)
            master_skies.append(outskyflux)

        # median_master_sky = np.median(master_skies, axis=0)
        # mean_master_sky = np.mean(master_skies, axis=0)
        # master_sky = np.zeros_like(mean_master_sky)
        # master_sky[:300] = mean_master_sky[:300]
        # master_sky[-300:] = mean_master_sky[-300:]
        # master_sky[300:-300] = median_master_sky[300:-300]
        master_sky = np.nanmedian(master_skies, axis=0)
        master_sky[np.isnan(master_sky)] = 0.
        #medmed = np.median(meds)
        #master_sky *= medmed
        #del meds, medmed

        masterfit = CubicSpline(x=skyllams, y=master_sky, extrapolate=False)
        plt.plot(skyllams, master_sky, 'k-', label='master', linewidth=4)
        plt.legend(loc='best')
        plt.show()
        nonzeros = np.where(master_sky > 0.)[0]
        first_nonzero_lam = skyllams[nonzeros[0]]
        last_nonzero_lam = skyllams[nonzeros[-1]]
        del nonzeros

        bin_by_tet = True
        if bin_by_tet:
            median_arrays1 = {int(i): [] for i in range(1, 9)}
            median_arrays2 = {int(i): [] for i in range(1, 9)}

            for ii, name in enumerate(skyfibs):
                tet = int(name[1])
                fib = int(name[2:4])
                if fib > 8:
                    median_arrays2[tet].append(master_skies[ii])
                else:
                    median_arrays1[tet].append(master_skies[ii])

            plt.figure()
            for (key1, arr1), (key2, arr2) in zip(median_arrays1.items(), median_arrays2.items()):
                if len(arr1) > 0:
                    med1 = np.nanmedian(np.asarray(arr1), axis=0)
                    plt.plot(skyllams, med1, label="{}_1".format(key1), alpha=0.4)
                if len(arr2) > 0:
                    med2 = np.nanmedian(np.asarray(arr2), axis=0)
                    plt.plot(skyllams, med2, label="{}_2".format(key2), alpha=0.4)
            plt.legend(loc='best')
            plt.show()

        print("\n\n\n\n\n")
        sums = []
        gal_is_sky = 0.
        sky_is_gal = 0.
        unp_is_sky = 0.
        unp_is_gal = 0.
        correct_s, correct_g = 0., 0.
        cutlow = 3161840
        cuthigh = 3161840
        for skyfib, spec in zip(skyfibs, master_skies):
            subs = spec[((skyllams > 5660) & (skyllams < 5850))]
            # subs = subs[np.bitwise_not(np.isnan(subs))]
            # subs = subs[subs>0.]
            summ = np.sum(subs)
            sums.append(summ)
            truename = "FIBER{}{:02d}".format(9 - int(skyfib[1]), 17 - int(skyfib[2:]))
            objname = sci_hdu.header[truename]
            if summ < cutlow:
                print(truename, skyfib, summ, 'sky', objname)
                if 'GAL' == objname[:3]:
                    gal_is_sky += 1
                elif objname[0] == 'u':
                    unp_is_sky += 1
                else:
                    correct_s += 1
            elif summ > cuthigh:
                print(truename, skyfib, summ, 'gal', objname)
                if 'SKY' == objname[:3]:
                    sky_is_gal += 1
                elif objname[0] == 'u':
                    unp_is_gal += 1
                else:
                    correct_g += 1

        sums = np.array(sums)
        plt.figure()
        plt.hist(sums, bins=60)
        plt.show()
        print(len(sums))
        print(np.where(sums < cutlow)[0].size, np.where(sums > cuthigh)[0].size)
        print(len(np.unique(list(target_sky_pair.values()))))
        print(gal_is_sky, unp_is_sky, correct_s, gal_is_sky + unp_is_sky + correct_s)
        print(sky_is_gal, unp_is_gal, correct_g, sky_is_gal + unp_is_gal + correct_g)
        for galfib, skyfib in target_sky_pair.items():
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



def fit_redshfits(self,cam):
    from fit_redshifts import fit_redshifts
    waves = Table(self.all_hdus[(cam,'combined_wavelengths','science',None)].data)
    fluxes = Table(self.all_hdus[(cam, 'combined_fluxes', 'science', None)].data)
    if (cam, 'combined_mask', 'science', None) in self.all_hdus.keys():
        mask = Table(self.all_hdus[(cam, 'combined_mask', 'science', None)].data)
    else:
        mask = None

    sci_data = OrderedDict()
    for ap in fluxes.colnames:
        if mask is None:
            current_mask = np.ones(len(waves[ap])).astype(bool)
        else:
            current_mask = mask[ap]
        sci_data[ap] = (waves[ap],fluxes[ap],current_mask)

    header = self.all_hdus[(cam, 'combined_fluxes', 'science', None)].header
    results = fit_redshifts(sci_data,mask_name=self.filemanager.maskname, run_auto=True)
    self.fit_data[cam] = fits.BinTableHDU(data=results, header=header, name='zfits')

