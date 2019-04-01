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
from sky_subtraction import subtract_sky
import matplotlib.pyplot as plt

class FieldData:
    def __init__(self, filenumbers, filemanager, instrument,
                     startstep='bias', pipeline_options={}, obj_info={} ):

        self.obs_pairing_strategy = pipeline_options['pairing_strategy']
        self.twod_to_oned = pipeline_options['twod_to_oned_strategy']
        self.debias_strategy = pipeline_options['debias_strategy']
        self.skysub_strategy = pipeline_options['skysub_strategy']

        self.convert_adu_to_e = (str(pipeline_options['convert_adu_to_e']).lower()=='true')
        self.skip_coarse_calib = (str(pipeline_options['try_skip_coarse_calib']).lower()=='true')
        self.single_core = (str(pipeline_options['single_core']).lower()=='true')
        self.save_plots = (str(pipeline_options['save_plots']).lower()=='true')
        self.show_plots = (str(pipeline_options['show_plots']).lower()=='true')
        self.only_peaks_in_coarse_cal = (str(pipeline_options['only_peaks_in_coarse_cal']).lower()=='true')


        self.check_parameter_flags()

        self.twostep_wavecomparc = (len(list(filenumbers['fine_comp']))>0)
        self.filemanager=filemanager
        self.instrument=instrument
        # self.targeting_data = targeting_data
        self.comparc_lampsc = instrument.coarse_lamp_names
        self.comparc_lampsf = instrument.fine_lamp_names

        self.filenumbers = filenumbers

        self.swapped_fibers_corrected = False
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
        self.observations.set_astronomical_object(obj_info)
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
        self.mtl = self.filemanager.get_matched_target_list()
        self.all_skies = {}
        self.all_gals = {}
        if self.reduction_order[startstep] > self.reduction_order['stitch']:
            self.update_headers()

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
        # if numeric_step_value > self.reduction_order['sky_sub']:
        #     for cam in self.instrument.cameras:
        #         self.instrument.full_fibs[self.camera].tolist()
        if numeric_step_value > self.reduction_order['combine']:
            self.filenumbers['science'] = np.array(['combined_fluxes','combined_wavelengths'])
        self.filemanager.update_templates_for(step)

    def proceed_to(self,step):
        if step != self.step:
            self.update_step(step)

    def update_headers(self):
        self.correct_swapped_fibers()
        self.add_target_information()

    def add_target_information(self):
        object = self.observations.object
        scifil = self.filenumbers['science'][0]
        header = self.all_hdus[(self.instrument.cameras[0], scifil, 'science', None)].header.copy()

        tests = []
        keys,vals = [],[]
        for key,val in object.items():
            outkey = key
            if 'TARG' not in key.upper():
                outkey = 'TARG'+key
            if outkey not in header.keys():
                keys.append(outkey)
                vals.append(val)

        if len(keys)==0:
            return

        for outkey,val in zip(keys,vals):
            for (camera, filenum, imtype, opamp), outhdu in self.all_hdus.items():
                if imtype in ['fibmap', 'bias']:
                    continue
                elif outkey in outhdu.header.keys():
                    continue
                else:
                    outhdu.header[outkey] = val

    def correct_swapped_fibers(self):
        if self.swapped_fibers_corrected or len(self.instrument.swapped_fibers)==0:
            self.swapped_fibers_corrected = True
            return

        scifil = self.filenumbers['science'][0]
        self.swapped_fibers_corrected = True
        for camera in self.instrument.cameras:
            header = self.all_hdus[(camera, scifil, 'science', None)].header
            if 'swpdfibs' in header.keys() and header['swpdfibs'].lower() == 'true':
                continue
            else:
                self.swapped_fibers_corrected = False

        if self.swapped_fibers_corrected:
            return

        deads = np.asarray(self.instrument.deadfibers)
        swaps = np.asarray(self.instrument.swapped_fibers)
        fiber_targets = {}

        for camera in self.instrument.cameras:
            header = self.all_hdus[(camera, scifil, 'science', None)].header.copy()
            for key,val in dict(header).items():
                if 'FIBER' in key:
                    outkey = key.replace('FIBER',camera)
                    fiber_targets[outkey] = val
                    if outkey in swaps and 'unplugged' not in val:
                        raise(TypeError,"The swapped fiber was originally assigned another target! {}: {}".format(outkey,val))

        fibermask = []
        for swap,dead in zip(swaps,deads):
            bad = False
            if dead not in fiber_targets.keys():
                print("The dead fiber wasn't in the header target list: {}".format(dead))
                bad=True
            if swap not in fiber_targets.keys():
                print("The dead fiber wasn't in the header target list: {}".format(dead))
                bad=True
            fibermask.append(bad)

        fibermask = np.asarray(fibermask)
        deads,swaps = deads[fibermask],swaps[fibermask]

        for (camera, filenum, imtype, opamp), outhdu in self.all_hdus.items():
            if imtype in ['fibmap','bias']:
                continue

            header_keys = list(outhdu.header.keys())
            if 'swpdfibs' in header_keys and outhdu.header['swpdfibs'].lower() == 'true':
                continue
            else:
                for dead,new in zip(deads,swaps):
                    if camera in new:
                        hnew = new.replace(camera, 'FIBER')
                        outhdu.header[hnew] = fiber_targets[dead]
                    if camera in dead:
                        hdead = dead.replace(camera, 'FIBER')
                        outhdu.header[hdead] = 'dead'

                outhdu.header['swpdfibs'] = 'True'
        self.swapped_fibers_corrected = True


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
        self.filemanager.write_all_filedata(self.all_hdus,self.step)
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
                                          convert_adu_to_e=self.convert_adu_to_e,save_plots=True,show_plots=False,\
                                          savetemplate=self.filemanager.get_saveplot_template)
        elif self.step == 'stitch':
            from stitch import stitch_all_images
            self.all_hdus = stitch_all_images(self.all_hdus,self.filemanager.date_timestamp)
            self.instrument.opamps = [None]
            self.data_stitched = True
            for (camera, filenum, imtype, opamp),outhdu in self.all_hdus.items():
                if imtype == 'twiflat' and filenum == self.filenumbers['twiflat'][0]:
                    plt.figure()
                    if self.show_plots or self.save_plots:
                        outarr = np.array(outhdu.data)
                        plt.imshow(outarr, origin='lower-left')
                        plt.clim(outarr.min(),2*np.mean(outarr))
                        plt.title("Stitched {} {} {}".format(camera, filenum, imtype))
                    if self.save_plots:
                        plt.savefig(self.filemanager.get_saveplot_template(cam=camera,ap='',imtype='twilight',step='stitch',comment='_'+str(filenum)),dpi=200)
                    if self.show_plots:
                        plt.show()
                    plt.close()
        elif self.step == 'remove_crs':
            if self.current_data_saved or self.current_data_from_disk:
                pass
            else:
                self.write_all_filedata()
            import PyCosmic
            if self.show_plots or self.save_plots:
                from quickreduce_funcs import format_plot
            for (camera, filenum, imtype, opamp) in self.all_hdus.keys():
                readfile = self.filemanager.get_read_filename(camera=camera, imtype=imtype,\
                                                        filenum=filenum, amp=opamp)
                writefile = self.filemanager.get_write_filename(camera=camera, imtype=imtype,\
                                                        filenum=filenum, amp=opamp)
                maskfile = writefile.replace('.fits', '.crmask.fits')
                print("\nFor image type: {}, shoe: {},   filenum: {}".format(imtype, camera, filenum))

                if self.single_core:
                    PyCosmic.detCos(readfile, maskfile, writefile, rdnoise='ENOISE',parallel=False,\
                                                              sigma_det=8, gain='EGAIN', verbose=True, return_data=False)
                else:
                    PyCosmic.detCos(readfile, maskfile, writefile, rdnoise='ENOISE',parallel=True,\
                                                              sigma_det=8, gain='EGAIN', verbose=True, return_data=False)

                if imtype == 'science':
                    maskdata = fits.getdata(maskfile)
                    plt.figure()
                    if self.show_plots or self.save_plots:
                        plt.imshow(maskdata, origin='lower-left')
                        plt.title("CR Mask {} {} {}".format(camera, filenum, imtype))
                    if self.save_plots:
                        plt.savefig(self.filemanager.get_saveplot_template(\
                            cam=camera,ap='',imtype='crmask',step='cr_removal',comment='_'+str(filenum)))
                    if self.show_plots:
                        plt.show()
                    plt.close()

            self.proceed_to('apcut')
            self.read_all_filedata()
        elif self.step == 'apcut':
            for camera in self.instrument.cameras:
                self.combine_fibermaps(camera, return_table=False)
            from aperture_detector import cutout_all_apperatures
            self.all_hdus = cutout_all_apperatures(self.all_hdus,self.instrument.cameras,\
                                                   deadfibers=self.instrument.deadfibers,summation_preference=self.twod_to_oned,\
                                                   show_plots=self.show_plots,save_plots=self.save_plots,save_template=self.filemanager.get_saveplot_template)
        elif self.step == 'wavecalib':
            if len(self.comparcs.keys()) == 0:
                self.populate_calibrations()

            for camera in self.instrument.cameras:
               self.comparcs[camera].run_initial_calibrations(skip_coarse = self.skip_coarse_calib,\
                                                              only_use_peaks=self.only_peaks_in_coarse_cal)

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
            zfit_hdus = {}
            for camera in self.instrument.cameras:
                zfit_hdus[(camera,None,'zfits',None)] = self.fit_redshfits(cam=camera)
            self.all_hdus = zfit_hdus

        if self.step not in ['remove_crs','wave_calib']:
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
            comparc = Calibrations(camera, self.instrument, self.comparc_lampsc, self.comparc_lampsf, comparc_cs, self.filemanager, \
                                 config=self.instrument.configuration, fine_calibrations=comparc_fs,\
                                 pairings=comparc_pairs, load_history=True, trust_after_first=False,\
                                 single_core=self.single_core,show_plots=self.show_plots,\
                                 save_plots=self.save_plots,savetemplate_funcs=self.filemanager.get_saveplot_template)

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

    ## TODO: Check if the same wavecal is used. If so, add before interpolation
    ## TODO: Rework to use same code as the combine_sciences?
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
            if self.save_plots or self.show_plots:
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
            if self.save_plots:
                plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='sci',step='flatten',comment='_'+str(filnum)))
            if self.show_plots:
                plt.show()
            plt.close()
            self.all_hdus[(cam,filnum,'science',None)] = fits.BinTableHDU(data=Table(outdata),header=hdr,name='flux')


    def subtract_skies(self, cam):
        if len(self.final_calibration_coefs.keys()) == 0:
            self.get_final_wavelength_coefs()
        if cam not in self.targeting_sky_pairs.keys():
            self.match_skies(cam)

        target_sky_pair = self.targeting_sky_pairs[cam]
        skyfibs = np.array(list(self.all_skies[cam].keys()))

        observation_keys = list(self.observations.observations.keys())
        first_obs = observation_keys[0]
        sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
        npixels = len(self.all_hdus[(cam, sci_filnum, 'science', None)].data)
        pixels = np.arange(npixels).astype(np.float64)
        pix2 = pixels * pixels
        pix3 = pix2 * pixels
        pix4 = pix3 * pixels
        pix5 = pix4 * pixels

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

            skyfits = {}
            plt.figure()
            mins, maxs = [], []

            for skyfib in skyfibs:
                a, b, c, d, e, f = comparc_data[skyfib]
                skylams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                skyflux = np.array(sci_data[skyfib])  # medfilt(sci_data[skyfib] - medfilt(sci_data[skyfib], 371), 3)
                skyflux[np.isnan(skyflux)] = 0.
                if self.save_plots or self.show_plots:
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
                med = np.nanmedian(outskyflux)
                meds.append(med)
                master_skies.append(outskyflux)

            master_sky = np.nanmedian(master_skies, axis=0)
            master_sky[np.isnan(master_sky)] = 0.
            #medmed = np.median(meds)
            #master_sky *= medmed
            del meds

            masterfit = CubicSpline(x=skyllams, y=master_sky, extrapolate=False)
            if self.save_plots or self.show_plots:
                plt.plot(skyllams, master_sky, 'k-', label='master', linewidth=4)
                plt.legend(loc='best')
            if self.save_plots:
                plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='science',step='skysub',comment='_allskies_'+str(sci_filnum)))
            if self.show_plots:
                plt.show()
            plt.close()

            nonzeros = np.where(master_sky > 0.)[0]
            first_nonzero_lam = skyllams[nonzeros[0]]
            last_nonzero_lam = skyllams[nonzeros[-1]]
            del nonzeros



            if self.save_plots or self.show_plots:
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
                    arr1 = np.array(arr1)
                    arr2 = np.array(arr2)
                    if arr1.shape[0] != 0:
                        med1 = np.nanmedian(np.asarray(arr1), axis=0)
                        plt.plot(skyllams, med1, label="{}_1".format(key1), alpha=0.4)
                    if arr2.shape[0] != 0:
                        med2 = np.nanmedian(np.asarray(arr2), axis=0)
                        plt.plot(skyllams, med2, label="{}_2".format(key2), alpha=0.4)

                if self.save_plots or self.show_plots:
                    plt.legend(loc='best')
                if self.save_plots:
                    plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='science',step='skysub',comment='_tetmedskies_'+str(sci_filnum)))
                if self.show_plots:
                    plt.show()
                plt.close()

                del median_arrays1,median_arrays2

            for galfib, skyfib in target_sky_pair.items():
                a, b, c, d, e, f = comparc_data[galfib]
                gallams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                galflux = np.asarray(sci_data[galfib])

                if self.skysub_strategy == 'nearest':
                    skyfit = skyfits[skyfib]
                    skyflux = skyfit(gallams)
                elif self.skysub_strategy == 'median':
                    skyflux = masterfit(gallams)

                lamcut = np.where((gallams < first_nonzero_lam) | (gallams > last_nonzero_lam))[0]
                galflux[lamcut] = 0.
                skyflux[lamcut] = 0.

                outgal,remaining_sky,gcont = subtract_sky(galflux=galflux,skyflux=skyflux,gallams=gallams)

                plt.subplots(1, 2)
                if self.save_plots or self.show_plots:
                    plt.subplot(121)
                    plt.plot(gallams, outgal, label='output', alpha=0.4)
                    plt.plot(gallams, remaining_sky, label='remaining_sky', alpha=0.4)
                    plt.plot(gallams, gcont, label='gcont', alpha=0.4)
                    plt.legend(loc='best')
                    ymin, ymax = plt.ylim()
                    plt.subplot(122)
                    plt.plot(gallams, galflux - skyflux, label='basic', alpha=0.4)
                    plt.plot(gallams, outgal, label='outgal', alpha=0.4)
                    # plt.plot(gallams, gal_contsub, label='gal_contsub',alpha=0.4)
                    plt.legend(loc='best')
                    # plt.ylim(ymin,ymax)
                if self.save_plots:
                    plt.savefig(self.filemanager.get_saveplot_template(cam='',ap=galfib,imtype='science',step='skysub',comment='_result_'+str(sci_filnum)))
                if self.show_plots:
                    plt.show()
                plt.close()

                out_sci_data.add_column(Table.Column(name=galfib, data=outgal))
                sci_lams[galfib] = gallams

            self.all_hdus[(cam, sci_filnum, 'science', None)] = fits.BinTableHDU(data=out_sci_data,
                                                                                 header=sci_hdu.header,
                                                                                 name='flux')

    ## TODO: Check if the same wavecal is used. If so, add before interpolation
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
        if self.save_plots or self.show_plots:
            for col in ref_sci_data.colnames:
                plt.plot(ref_wavearrays[col],combined_data[col],alpha=0.2)
        if self.save_plots:
            plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='science',step='combine',comment='_all_1d'))
        if self.show_plots:
            plt.show()
        plt.close()
        ref_wave_table = Table(ref_wavearrays)
        self.all_hdus[(cam,'combined_fluxes','science',None)] = fits.BinTableHDU(data=combined_data,header=ref_sci_hdu.header,name='flux')
        self.all_hdus[(cam,'combined_wavelengths','science',None)] = fits.BinTableHDU(data=ref_wave_table,name='wave')


    def match_skies(self ,cam):
        from astropy.coordinates import SkyCoord
        import astropy.units as u

        first_obs = list(self.observations.observations.keys())[0]
        sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
        sci_hdu = self.all_hdus[(cam, sci_filnum, 'science', None)]
        header = sci_hdu.header

        skies, scis = {}, {}
        extra_skies = {}
        cards = dict(header)
        for fibid, objname in cards.items():
            if fibid[:5].upper() == 'FIBER':
                camid = fibid.replace("FIBER",cam)
                if camid not in self.instrument.deadfibers:
                    objname = objname.strip(' \t')
                    if 'SKY' in objname.upper():
                        skies[camid] = objname
                    elif objname.lower() == 'unplugged':
                        extra_skies[camid] = objname
                    else:# elif 'GAL' in objname.upper():
                        scis[camid] = objname

        all_skies = skies.copy()
        for key,val in extra_skies.items():
            all_skies[key] = val

        if self.mtl is None:
            skynums ,skynames = [] ,[]
            for fibname in all_skies.keys():
                skynames.append(fibname)
                number_str = fibname.lstrip('rb')
                if cam == 'r':
                    numeric = 16* int(number_str[0]) + int(number_str[1:])
                else:
                    numeric = 16 * (9- int(number_str[0])) + int(number_str[1:])
                skynums.append(numeric)
            skynums = np.array(skynums)
            target_sky_pair = {}
            for fibname, objname in scis.items():
                number_str = fibname.lstrip('rb')
                if cam == 'r':
                    galnum = 16* int(number_str[0]) + int(number_str[1:])
                else:
                    galnum = 16 * (9- int(number_str[0])) + int(number_str[1:])
                minsepind = np.argmin(np.abs(skynums - galnum))
                target_sky_pair[fibname] = skynames[minsepind]
        else:
            skyloc_array = []
            skynames = []
            for fibname, objname in skies.items():
                fibrow = np.where(self.mtl['FIBNAME'] == fibname)[0][0]
                objrow = np.where(self.mtl['ID'] == objname)[0][0]
                if fibrow != objrow:
                    print("Fiber and object matched to different rows!")
                    print(fibname, objname, fibrow, objrow)
                    raise ()
                if self.mtl['RA'].mask[fibrow]:
                    ra, dec = self.mtl['RA_targeted'][fibrow], self.mtl['DEC_targeted'][fibrow]
                else:
                    ra, dec = self.mtl['RA'][fibrow], self.mtl['DEC'][fibrow]
                skyloc_array.append([ra, dec])
                skynames.append(fibname)
            sky_coords = SkyCoord(skyloc_array, unit=u.deg)

            target_sky_pair = {}
            for fibname, objname in scis.items():
                fibrow = np.where(self.mtl['FIBNAME'] == fibname)[0][0]
                objrow = np.where(self.mtl['ID'] == objname)[0][0]
                if fibrow != objrow:
                    print("Fiber and object matched to different rows!")
                    print(fibname, objname, fibrow, objrow)
                    raise ()
                if self.mtl['RA'].mask[fibrow]:
                    ra, dec = self.mtl['RA_targeted'][fibrow], self.mtl['DEC_targeted'][fibrow]
                else:
                    ra, dec = self.mtl['RA'][fibrow], self.mtl['DEC'][fibrow]
                coord = SkyCoord(ra, dec, unit=u.deg)
                seps = coord.separation(sky_coords)
                minsepind = np.argmin(seps)
                target_sky_pair[fibname] = skynames[minsepind]

        self.targeting_sky_pairs[cam] = target_sky_pair
        self.all_skies[cam] = all_skies
        self.all_gals[cam] = scis


    def fit_redshfits(self,cam):
        from fit_redshifts import fit_redshifts_wrapper
        waves = Table(self.all_hdus[(cam,'combined_wavelengths','science',None)].data)
        fluxes = Table(self.all_hdus[(cam, 'combined_fluxes', 'science', None)].data)
        if (cam, 'combined_mask', 'science', None) in self.all_hdus.keys():
            mask = Table(self.all_hdus[(cam, 'combined_mask', 'science', None)].data)
        else:
            mask = None

        header = self.all_hdus[(cam, 'combined_fluxes', 'science', None)].header

        if self.single_core:
            sci_data = OrderedDict()
            for fib in fluxes.colnames:
                if mask is None:
                    current_mask = np.ones(len(waves[fib])).astype(bool)
                else:
                    current_mask = mask[fib]
                sci_data[fib] = (waves[fib], fluxes[fib], current_mask)
            obs1 = {
                'sky_subd_sciences':sci_data, 'mask_name': self.filemanager.maskname, 'savetemplate_func': self.filemanager.get_saveplot_template, 'run_auto': True
            }
            results = fit_redshifts_wrapper(obs1)
        else:
            sci_data1 = OrderedDict()
            fib1s = self.instrument.lower_half_fibs[cam]

            for fib in fib1s:
                if fib in fluxes.colnames:
                    if mask is None:
                        current_mask = np.ones(len(waves[fib])).astype(bool)
                    else:
                        current_mask = mask[fib]
                    sci_data1[fib] = (waves[fib], fluxes[fib], current_mask)

            sci_data2 = OrderedDict()
            fib2s = self.instrument.upper_half_fibs[cam]
            for fib in fib2s:
                if fib in fluxes.colnames:
                    if mask is None:
                        current_mask = np.ones(len(waves[fib])).astype(bool)
                    else:
                        current_mask = mask[fib]
                    sci_data2[fib] = (waves[fib], fluxes[fib], current_mask)

            obs1 = {
                'sky_subd_sciences':sci_data1, 'mask_name': self.filemanager.maskname, 'savetemplate_func': self.filemanager.get_saveplot_template, 'run_auto': True
            }
            obs2 = {
                'sky_subd_sciences':sci_data2, 'mask_name': self.filemanager.maskname, 'savetemplate_func': self.filemanager.get_saveplot_template, 'run_auto': True
            }

            all_obs = [obs1, obs2]
            NPROC = len(all_obs)

            from multiprocessing import Pool
            with Pool(NPROC) as pool:
                tabs = pool.map(fit_redshifts_wrapper, all_obs)

            from astropy.table import vstack
            results = vstack([tabs[0], tabs[1]])

        return fits.BinTableHDU(data=results, header=header, name='zfits')

