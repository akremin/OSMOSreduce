from collections import OrderedDict
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import CubicSpline

from calibrations import Calibrations
from observations import Observations
from sky_subtraction import subtract_sky_loop_wrapper, replace_nans


class FieldData:
    def __init__(self, filenumbers, filemanager, instrument,
                     startstep='bias', pipeline_options={}, obj_info={} ):

        self.obs_pairing_strategy = pipeline_options['pairing_strategy']
        self.twod_to_oned = pipeline_options['twod_to_oned_strategy']
        self.debias_strategy = pipeline_options['debias_strategy']
        self.skysub_strategy = pipeline_options['skysub_strategy']
        self.initial_calib_priors = pipeline_options['initial_calib_priors']

        self.convert_adu_to_e = (str(pipeline_options['convert_adu_to_e']).lower()=='true')
        self.skip_coarse_calib = (str(pipeline_options['try_skip_coarse_calib']).lower()=='true')
        self.skip_fine_calib = (str(pipeline_options['debug_skip_fine_calib']).lower() == 'true')
        self.single_core = (str(pipeline_options['single_core']).lower()=='true')
        self.save_plots = (str(pipeline_options['save_plots']).lower()=='true')
        self.show_plots = (str(pipeline_options['show_plots']).lower()=='true')
        self.only_peaks_in_coarse_cal = (str(pipeline_options['only_peaks_in_coarse_cal']).lower()=='true')
        self.use_selected_calib_lines = (str(pipeline_options['use_selected_calib_lines']).lower()=='true')
        self.use_history_calibs = (str(pipeline_options['use_history_calibs']).lower()=='true')

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
        self.observations = Observations(self.filenumbers.copy(),self.obs_pairing_strategy)
        self.observations.set_astronomical_object(obj_info)
        self.comparcs = {}
        self.fit_data = {}
        self.targeting_sky_pairs = {}
        self.update_step(startstep)
        self.all_hdus = {}
        self.read_all_filedata()
        # self.final_calibrated_hdulists = {}
        # self.final_calibration_coefs = {}
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

        if self.initial_calib_priors not in ['medians','defaults','parametrized']:
            print("\n\n ---> Only medians, defaults, parametrized utilizations of the coarse calibrations is permitted. Defaulting to 'defaults'\n\n")
            self.initial_calib_priors = 'defaults'

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
        if numeric_step_value > self.reduction_order['skysub']:
            self.filenumbers['masks'] = self.filenumbers['science']
        #     for cam in self.instrument.cameras:
        #         self.instrument.full_fibs[self.camera].tolist()
        if numeric_step_value > self.reduction_order['combine']:
            self.filenumbers['science'] = np.array(['combined'])
            self.filenumbers['wavelengths'] = np.array(['combined'])
            self.filenumbers['masks'] = np.array(['combined'])
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
        if numeric_step > self.reduction_order['skysub']:
            if 'masks' not in self.filenumbers.keys():
                return False
        if numeric_step > self.reduction_order['combine']:
            if 'wavelengths' not in self.filenumbers.keys():
                return False
            if 'masks' not in self.filenumbers.keys():
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
                        plt.savefig(self.filemanager.get_saveplot_template(cam=camera,ap='',imtype='twilight',step='stitch',comment='_'+str(filenum)),dpi=600)
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
                pass
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
                            cam=camera,ap='',imtype='crmask',step='cr_removal',comment='_'+str(filenum)),dpi=600)
                    if self.show_plots:
                        plt.show()
                    plt.close()

            self.proceed_to('apcut')
            self.read_all_filedata()
        elif self.step == 'apcut':
            for camera in self.instrument.cameras:
                self.combine_fibermaps(camera, return_table=False)
            from aperture_detector import cutout_all_apperatures
            outhdus = cutout_all_apperatures(self.all_hdus,self.instrument.cameras,\
                                                   deadfibers=self.instrument.deadfibers,summation_preference=self.twod_to_oned,\
                                                   show_plots=self.show_plots,save_plots=self.save_plots,\
                                                   save_template=self.filemanager.get_saveplot_template)
            self.all_hdus = outhdus
        elif self.step == 'wavecalib':
            if len(self.comparcs.keys()) == 0:
                self.populate_calibrations(try_loading_finals=False)

            for camera in self.instrument.cameras:
               self.comparcs[camera].run_initial_calibrations(skip_coarse = self.skip_coarse_calib, \
                                                              use_history_calibs = self.use_history_calibs,\
                                                              only_use_peaks = self.only_peaks_in_coarse_cal)
            if self.skip_fine_calib:
                print("Skipping fine calibration as the flag was set. Note if previous calibrations don't exist the next steps will fail")
            else:
                for camera in self.instrument.cameras:
                    self.comparcs[camera].run_final_calibrations(initial_priors=self.initial_calib_priors)
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
            self.filenumbers['masks'] = self.filenumbers['science']

        elif self.step == 'zfit':
            zfit_hdus = {}
            for camera in self.instrument.cameras:
                zfit_hdus[(camera,None,'zfits',None)] = self.fit_redshfits(cam=camera)
            self.all_hdus = zfit_hdus
            self.filenumbers['science'] = np.array(['combined'])
            self.filenumbers['wavelengths'] = np.array(['combined'])
            self.filenumbers['masks'] = np.array(['combined'])
        if self.step not in ['remove_crs','wave_calib']:
            self.current_data_saved = False
            self.current_data_from_disk = False


    def populate_calibrations(self,try_loading_finals=False):
        comparc_pairs = self.observations.return_comparc_pairs()
        self.comparcs = {}
        for camera in self.instrument.cameras:
            if try_loading_finals:
                comparc_cs = {}
                comparc_fs = {}
            else:
                comparc_cs = {filnum:self.all_hdus[(camera,filnum,'coarse_comp',None)] for filnum in self.observations.comparc_cs}
                if self.observations.two_step_comparc:
                    comparc_fs = {filnum: self.all_hdus[(camera, filnum, 'fine_comp', None)] for filnum in self.observations.comparc_fs}
                else:
                    comparc_fs = None
            comparc = Calibrations(camera, self.instrument, comparc_cs, self.filemanager, fine_calibrations=comparc_fs,\
                                 pairings=comparc_pairs, load_history=True, trust_after_first=False,\
                                 single_core=self.single_core,show_plots=self.show_plots,\
                                 save_plots=self.save_plots,use_selected_calib_lines=self.use_selected_calib_lines,
                                try_load_finals=try_loading_finals)

            self.comparcs[camera] = comparc



    def get_final_wavelength_coefs(self):
        self.populate_calibrations(try_loading_finals=True)

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
        if len(self.comparcs) == 0:
            self.populate_calibrations(try_loading_finals=True)
        filnum = self.observations.comparc_fs[0]
        comparc_table = self.comparcs[cam].generate_time_interpolated_calibration(self.all_hdus[(cam,'master','twiflat',None)])
        return
        # comparc_table = self.final_calibration_coefs[(cam,filnum)]
        #self.filemanager.load_calib_dict('default', cam, self.instrument.configuration)

        final_table, final_flux_array = flatten_data(fiber_fluxes=fiber_fluxes,waves=comparc_table)

        orig_flats_arr = np.ndarray(shape=(len(fiber_fluxes.colnames), len(fiber_fluxes.columns[0])))
        flats_arr = np.ndarray(shape=(len(final_table.colnames), len(final_table.columns[0])))
        orig_flattened_arr = np.ndarray(shape=(len(fiber_fluxes.colnames), len(fiber_fluxes.columns[0])))

        ## Ensure proper ordering so the spliced images look similar to the original uncut images
        names = []
        for tet in range(1, 9):
            for fib in range(1, 17):
                if cam == 'b':
                    name = '{}{:d}{:02d}'.format(cam, 9 - tet, fib)
                else:
                    name = '{}{:d}{:02d}'.format(cam, tet, fib)
                if name in final_table.colnames:
                    names.append(name)
        for ii, col in enumerate(names):
            orig_flats_arr[ii,:] = fiber_fluxes[col]
            flats_arr[ii, :] = final_table[col]
            orig_flattened_arr[ii,:] = fiber_fluxes[col]/final_table[col]

        plt.subplots(2, 2)
        if self.save_plots or self.show_plots:
            plt.subplot(221)
            im = plt.imshow(orig_flats_arr - orig_flats_arr.min() + 1., aspect='auto', origin='lower-left')
            plt.colorbar()
            clow, chigh = im.get_clim()
            plt.title("Original twiflat cam:{}".format(cam))
            plt.subplot(222)
            plt.imshow(flats_arr - flats_arr.min() + 1., aspect='auto', origin='lower-left')
            plt.title("Resulting Aperture Flat cam:{}".format(cam))

            plt.colorbar()
            plt.subplot(223)
            plt.imshow(orig_flattened_arr - orig_flattened_arr.min() + 1., aspect='auto', origin='lower-left')
            plt.title("Flattened Twiflats cam:{}".format(cam))
            plt.clim(clow, chigh)
            plt.colorbar()
            plt.subplot(224)
            plt.imshow(np.log(orig_flattened_arr - orig_flattened_arr.min() + 1.), aspect='auto', origin='lower-left')
            plt.title("Log(Flattened Twiflats) cam:{}".format(cam))
            plt.colorbar()
            plt.tight_layout()
        if self.save_plots:
            fig_comment = '-'.join([str(fnum) for fnum in self.filenumbers['twiflat']])
            plt.savefig(self.filemanager.get_saveplot_template(cam=cam, ap='', imtype='twiflat', step='flatten',\
                                                               comment='_'+fig_comment), dpi=600)
        if self.show_plots:
            plt.show()
        plt.close()

        for filnum in self.filenumbers['science']:
            hdu = self.all_hdus[(cam,filnum,'science',None)]
            data = hdu.data.copy()
            hdr = hdu.header
            data_arr = np.ndarray(shape=(len(final_table.colnames),len(final_table.columns[0])))
            flt_data_arr = np.ndarray(shape=(len(final_table.colnames),len(final_table.columns[0])))
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
                data_arr[ii, :] = data[col]
                flt_data_arr[ii,:] = outdata[col]
            #plt.show()
            plt.subplots(2,2)
            if self.save_plots or self.show_plots:
                plt.subplot(221)
                im = plt.imshow(data_arr-data_arr.min()+1., aspect='auto', origin='lower-left')
                plt.colorbar()
                clow,chigh = im.get_clim()
                plt.title("Orig. Science cam:{} filnm:{}".format(cam, filnum))
                plt.subplot(222)
                plt.imshow(np.log(data_arr-data_arr.min()+1.), aspect='auto', origin='lower-left')
                plt.title("Log(Orig. Science) cam:{} filnm:{}".format(cam,filnum))

                plt.colorbar()
                plt.subplot(223)
                plt.imshow(flt_data_arr-flt_data_arr.min()+1., aspect='auto', origin='lower-left')
                plt.title("FlatScience cam:{} filnm:{}".format(cam,filnum))
                plt.clim(clow, chigh)
                plt.colorbar()
                plt.subplot(224)
                plt.imshow(np.log(flt_data_arr-flt_data_arr.min()+1.), aspect='auto', origin='lower-left')
                plt.title("Log(Flat Science) cam:{} filnm:{}".format(cam,filnum))
                plt.colorbar()
                plt.tight_layout()
            if self.save_plots:
                plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='science',step='flatten',comment='_'+str(filnum)),dpi=600)
            if self.show_plots:
                plt.show()
            plt.close()
            self.all_hdus[(cam,filnum,'science',None)] = fits.BinTableHDU(data=Table(outdata),header=hdr,name='FLUX')

    def subtract_skies(self, cam):
        wave_grid_step = 0.25
        wave_unit = 'A'
        wave_type = 'linear'
        if len(self.comparcs) == 0:
            self.populate_calibrations(try_loading_finals=True)
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
            nsecs_limit = 60*5.1
            if float(sci_hdu.header['EXPTIME']) < nsecs_limit:
                dosimple_subtraction = True
                print("A short exposure of less than {} detected.".format(nsecs_limit))
                print("Attempting to adjust the skyflux to match fiber throughputs, but then performing direct subtractions.")
            else:
                dosimple_subtraction = False
            # if self.twostep_wavecomparc:
            #     calib_filnum = fcalib
            # else:
            #     calib_filnum = ccalib
            # comparc_data = self.final_calibration_coefs[(cam, calib_filnum)]
            comparc_data = self.comparcs[cam].generate_time_interpolated_calibration(sci_hdu)

            sci_data = Table(sci_hdu.data)
            out_sci_data = Table()
            out_mask_data = Table()

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

            wave_grid = np.arange(np.ceil(np.min(mins)), np.floor(np.max(maxs)), wave_grid_step).astype(np.float64)
            self.wave_grid = wave_grid
            del mins, maxs

            master_skies = []
            meds = []
            for skyfib in skyfibs:
                skyfit = skyfits[skyfib]
                outskyflux = skyfit(wave_grid)
                med = np.nanmedian(outskyflux)
                if np.isnan(med):
                    outskyflux[np.isnan(outskyflux)] = 0.
                    med = np.nanmedian(outskyflux)
                meds.append(med)
                master_skies.append(outskyflux)

            master_sky = np.nanmedian(master_skies, axis=0)
            master_sky = replace_nans(master_sky)
            #medmed = np.median(meds)
            #master_sky *= medmed
            del meds

            #masterfit = CubicSpline(x=wave_grid, y=master_sky, extrapolate=False)
            if self.save_plots or self.show_plots:
                plt.plot(wave_grid, master_sky, 'k-', label='master', linewidth=2,alpha=0.4)
                plt.legend(loc='best')
            if self.save_plots:
                plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='science',step='skysub',comment='_allskies_'+str(sci_filnum)),dpi=600)
            if self.show_plots:
                plt.show()
            plt.close()

            nonzeros = np.where(master_sky > 0.)[0]
            first_nonzero_lam = wave_grid[nonzeros[0]]
            last_nonzero_lam = wave_grid[nonzeros[-1]]
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
                        plt.plot(wave_grid, med1, label="{}_1".format(key1), alpha=0.4)
                    if arr2.shape[0] != 0:
                        med2 = np.nanmedian(np.asarray(arr2), axis=0)
                        plt.plot(wave_grid, med2, label="{}_2".format(key2), alpha=0.4)

                if self.save_plots or self.show_plots:
                    plt.legend(loc='best')
                if self.save_plots:
                    plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='science',step='skysub',comment='_tetmedskies_'+str(sci_filnum)),dpi=600)
                if self.show_plots:
                    plt.show()
                plt.close()

                del median_arrays1,median_arrays2

            interpd_galfluxes, skyfluxes, galmasks = OrderedDict(),OrderedDict(),OrderedDict()
            for galfib, skyfib in target_sky_pair.items():
                a, b, c, d, e, f = comparc_data[galfib]
                gallams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
                galflux = np.asarray(sci_data[galfib])

                galspline = CubicSpline(x=gallams,y=galflux,extrapolate=False)
                interpd_galflux = galspline(wave_grid)
                galmask = np.isnan(interpd_galflux)
                # lamcut = ((gallams>=first_nonzero_lam)&(gallams<=last_nonzero_lam))
                # galmask = np.bitwise_not(lamcut)

                if self.skysub_strategy == 'nearest':
                    skyfit = skyfits[skyfib]
                    skyflux = skyfit(wave_grid)
                    skymask = np.isnan(skyflux)
                    galmask = (skymask | galmask)
                else:
                    skyflux = master_sky
                # elif self.skysub_strategy == 'median':
                #     skyflux = master_sky

                interpd_galfluxes[galfib], skyfluxes[galfib], galmasks[galfib] =  interpd_galflux, skyflux, galmask

            spectypes = ['gal','sky','gcont','scont','mask']
            if self.single_core:
                sky_inputs = { 'galfluxes':interpd_galfluxes, 'skyfluxes':skyfluxes, \
                         'wave_grid':wave_grid, 'galmasks':galmasks, 'quickreturn':dosimple_subtraction}
                outdict = subtract_sky_loop_wrapper(sky_inputs)
            else:
                sky_inputs1 = { 'galfluxes':OrderedDict(), 'skyfluxes':OrderedDict(), \
                         'wave_grid':wave_grid.copy(), 'galmasks':OrderedDict(), 'quickreturn':dosimple_subtraction }
                sky_inputs2 = { 'galfluxes':OrderedDict(), 'skyfluxes':OrderedDict(), \
                         'wave_grid':wave_grid.copy(), 'galmasks':OrderedDict(), 'quickreturn':dosimple_subtraction }

                for fib in self.instrument.lower_half_fibs[cam]:
                    if fib in target_sky_pair.keys():
                        sky_inputs1['galfluxes'][fib] = interpd_galfluxes[fib]
                        sky_inputs1['skyfluxes'][fib] = skyfluxes[fib]
                        sky_inputs1['galmasks'][fib] = galmasks[fib]
                for fib in self.instrument.upper_half_fibs[cam]:
                    if fib in target_sky_pair.keys():
                        sky_inputs2['galfluxes'][fib] = interpd_galfluxes[fib]
                        sky_inputs2['skyfluxes'][fib] = skyfluxes[fib]
                        sky_inputs2['galmasks'][fib] = galmasks[fib]

                all_sky_inputs = [sky_inputs1, sky_inputs2]
                NPROC = np.clip(len(all_sky_inputs),1,2)

                with Pool(NPROC) as pool:
                    outs = pool.map(subtract_sky_loop_wrapper, all_sky_inputs)
                outdict,outdict2 = outs

                for fib in self.instrument.upper_half_fibs[cam]:
                    if fib in target_sky_pair.keys():
                        for spectype in spectypes:
                            outdict[spectype][fib] = outdict2[spectype][fib]
            # galfluxes, skyfluxes, wave_grid, galmask
            # outdict = {'gal': outgals, 'sky': remaining_skies, 'gcont': gconts, 'scont': sconts, 'mask': maskeds}

            outgals, remaining_skies = outdict['gal'], outdict['sky']
            gconts, sconts = outdict['gcont'], outdict['scont']
            maskeds =  outdict['mask']
            for galfib in target_sky_pair.keys():
                outgal, remaining_sky = outgals[galfib],remaining_skies[galfib]
                gcont, scont = gconts[galfib], sconts[galfib]
                masked = maskeds[galfib]
                interpd_galflux = interpd_galfluxes[galfib]
                skyflux = skyfluxes[galfib]
                good_vals = np.bitwise_not(masked|np.isnan(outgal))
                plt.subplots(2, 2)
                if self.save_plots or self.show_plots:
                    plt.subplot(221)
                    plt.plot(wave_grid[good_vals], interpd_galflux[good_vals], alpha=0.3, label='Science')
                    plt.plot(wave_grid[good_vals], skyflux[good_vals], alpha=0.3, label='Sky')
                    plt.ylabel('Counts [Arbitrary]')
                    plt.xlabel("Wavelength [Angstroms]")
                    plt.title("Raw Flux")
                    plt.legend()
                    plt.subplot(222)
                    plt.plot(wave_grid[good_vals], gcont[good_vals], alpha=0.3, label='Science')
                    plt.plot(wave_grid[good_vals], scont[good_vals], alpha=0.3, label='Sky')
                    plt.ylabel('Counts [Arbitrary]')
                    plt.xlabel("Wavelength [Angstroms]")
                    plt.title("Continuum Fits")
                    plt.legend()
                    plt.subplot(223)
                    plt.plot(wave_grid[good_vals], interpd_galflux[good_vals]-gcont[good_vals], alpha=0.3, label='Science')
                    plt.plot(wave_grid[good_vals], skyflux[good_vals]-scont[good_vals], alpha=0.3, label='Sky')
                    plt.ylabel('Counts [Arbitrary]')
                    plt.xlabel("Wavelength [Angstroms]")
                    plt.title("Continuum Subtracted")
                    plt.legend()
                    plt.subplot(224)
                    plt.plot(wave_grid[good_vals], outgal[good_vals], alpha=0.4, label='Science')
                    plt.plot(wave_grid[good_vals], skyflux[good_vals], alpha=0.2, label='Orig. Sky')
                    plt.ylabel('Counts [Arbitrary]')
                    plt.xlabel("Wavelength [Angstroms]")
                    plt.title("Final Masked Output")
                    plt.legend()
                    plt.tight_layout()

                if self.save_plots:
                    plt.savefig(self.filemanager.get_saveplot_template(cam='',ap=galfib,imtype='science',step='skysub',comment='_result_'+str(sci_filnum)),dpi=600)
                if self.show_plots:
                    plt.show()
                plt.close()

                out_sci_data.add_column(Table.Column(name=galfib, data=outgal))
                out_mask_data.add_column(Table.Column(name=galfib, data=masked))

            out_header = sci_hdu.header.copy()
            out_header['wavemin'] = wave_grid.min()
            out_header['wavemax'] = wave_grid.max()
            out_header['wavestep'] = wave_grid_step
            out_header['waveunit'] = wave_unit
            out_header['wavetype'] = wave_type
            self.all_hdus[(cam, sci_filnum, 'science', None)] = fits.BinTableHDU(data=out_sci_data,
                                                                                 header=out_header,
                                                                                 name='FLUX')
            self.all_hdus[(cam, sci_filnum, 'masks', None)] = fits.BinTableHDU(data=out_mask_data,
                                                                                 header=out_header,
                                                                                 name='MASK')

    def generate_wave_grid(self,header):
        wavemin,wavemax = header['wavemin'], header['wavemax']
        wavetype = header['wavetype']
        if wavetype == 'log':
            if 'numwaves' in list(header.getHdrKeys()):
                nwaves = header['numwaves']
            else:
                nwaves = header['NAXIS1']
            if 'logbase' in list(header.getHdrKeys()):
                logbase = header['logbase']
            else:
                logbase = 10
            outwaves = np.logspace(wavemin,wavemax,num=nwaves,base=logbase)
        else:
            wavestep = header['wavestep']
            outwaves = np.arange(wavemin,wavemax+wavestep,wavestep)
        return outwaves

    def combine_science_observations(self, cam):
        from scipy.ndimage import gaussian_filter
        observation_keys = list(self.observations.observations.keys())
        nobs = len(observation_keys)

        logbase = None
        for obs in observation_keys:
            sci_filnum, ccalib, fcalib, comparc_ind = self.observations.observations[obs]
            sci_header = self.all_hdus[(cam, sci_filnum, 'science', None)].header
            if obs == 0:
                wavemin, wavemax = sci_header['wavemin'], sci_header['wavemax']
                wavestep = sci_header['wavestep']
                if 'logbase' in list(sci_header.keys()):
                    logbase = sci_header['logbase']
            else:
                if sci_header['wavemin'] < wavemin:
                    wavemin = sci_header['wavemin']
                if sci_header['wavemax'] > wavemax:
                    wavemax = sci_header['wavemax']
                if wavestep != sci_header['wavestep']:
                    print("Wave steps were different!!")
                    raise()
                if 'logbase' in list(sci_header.keys()) and logbase != sci_header['logbase']:
                    print("Wavelength log base was different!!")
                    raise()

        if logbase is not None:
            master_wave_grid = np.logspace(wavemin,wavemax,num=sci_header['NAXIS2'],base=logbase)
        else:
            master_wave_grid = np.arange(wavemin,wavemax+wavestep,wavestep)

        medians = { }
        n_arbitrary_counts_per_exposure = 10000
        for obs in observation_keys:
            sci_filnum, ccalib, fcalib, comparc_ind = self.observations.observations[obs]
            sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))
            sci_data = Table(sci_hdu.data)

            wave_grid = self.generate_wave_grid(sci_hdu.header)

            mask_hdu = self.all_hdus.pop((cam, sci_filnum, 'masks', None))
            mask_data = Table(mask_hdu.data)

            matching_lams = ((wave_grid.min() <= master_wave_grid) & (wave_grid.max() >= master_wave_grid))
            if np.sum(matching_lams) != len(sci_data):
                print("Flux length wasn't the number of matching wavelengths!")
                print(np.sum(matching_lams),"!=",len(sci_data))
                print("Obs {}: Wavemin={} wavemax={} wavestep={}".format(obs,wave_grid.min(), wave_grid.max(),
                                                                 wave_grid[1] - wave_grid[0]))
                print("Master: Wavemin={} wavemax={} wavestep={}".format(master_wave_grid.min(), master_wave_grid.max(),
                                                                 master_wave_grid[1] - master_wave_grid[0]))

            if obs == 0:
                all_fluxs = {col:[] for col in sci_data.colnames}
                all_masks = {col:[] for col in sci_data.colnames}
                medians =   {col: [] for col in sci_data.colnames}

            oned_array = np.zeros(shape=(len(master_wave_grid),))
            for col in sci_data.colnames:
                outflux = oned_array.copy()
                outmask = oned_array.copy()
                outflux[matching_lams] = sci_data[col].copy()
                outmask[matching_lams] = mask_data[col].copy()
                outmask[np.bitwise_not(matching_lams)] = True
                medians[col].append(np.nanmedian(outflux))
                outflux = outflux/np.nanmedian(outflux)
                all_fluxs[col].append(outflux)
                all_masks[col].append(outmask)

        out_data_table = Table()
        out_mask_table = Table()
        out_uncmult_table = Table()
        for col in sci_data.colnames:
            flux_arr = np.array(all_fluxs[col])*np.sum(medians[col])
            masked_arr = np.array(all_masks[col]).astype(bool)
            flux_arr[masked_arr] = np.nan
            ngood_obs_perpix = float(nobs)-np.sum(masked_arr,axis=0)
            masked = ((ngood_obs_perpix < int(np.ceil(float(nobs) / 2.))))
            from collections import Counter
            # print(col,len(masked),np.sum(masked),np.sum(np.bitwise_not(masked)),len(observation_keys),Counter(ngood_obs_perpix))

            ## Transform the array back to a realistic estimate of counts
            ngood_obs_perpix[ngood_obs_perpix==0] = np.nan
            nanmean_fluxes = np.nanmean(flux_arr,axis=0)
            ## poisson statistics
            unc_values = np.nanstd(np.sqrt(flux_arr),axis=0)

            masked = (masked | np.isnan(nanmean_fluxes) | np.isnan(unc_values) | np.isinf(unc_values) | np.isinf(nanmean_fluxes))

            unc_values[masked] = np.nan
            nanmean_fluxes[masked] = np.nan

            nansumd_fluxes = nanmean_fluxes

            # ## All masked values should be nan, get all masked values
            # alleged_nans = nanmean_fluxes[masked]
            # ## if they are all nans, then 'not' that will be all False. The sum of which should be 0
            # if np.sum(np.bitwise_not(np.isnan(alleged_nans)))!=np.sum(np.isnan(ngood_obs_perpix)):
            #     print("Mask assignment didn't match nanmean in combining fluxes!")
            #     print("Nmasked: {}, n_notenoughobs: {}, n_nanpix_tomask: {}".format(np.sum(masked),np.sum(np.bitwise_not(np.isnan(alleged_nans))),np.sum(np.isnan(ngood_obs_perpix))))
            if np.sum(masked) == len(masked):
                print("All values are masked in {}!".format(col))
                print(np.sum(masked),np.sum(np.isnan(ngood_obs_perpix)))

            from scipy.interpolate import CubicSpline
            masked = masked.astype(bool)
            nmaskbins = 55  ## must be odd
            start = (nmaskbins - 1)
            half = start // 2
            cutmask = masked[start:].copy()
            for ii in range(1, start + 1):
                cutmask = (cutmask | masked[(start - ii):-ii])
            cutmask = np.append(np.append([True] * half, cutmask), [True] * half)
            fitd_spectrum_func = CubicSpline(master_wave_grid[np.bitwise_not(cutmask)],nansumd_fluxes[np.bitwise_not(cutmask)],extrapolate=False)
            del cutmask
            ## Note we fit with a much larger mask, but we're only using that fit on the masked data
            nansumd_fluxes[masked] = fitd_spectrum_func(master_wave_grid[masked])
            cutnansumd_flux = gaussian_filter(nansumd_fluxes, sigma=0.66, order=0)

            unc = unc_values
            cutmask = ( masked[2:] | masked[1:-1] | masked[:-2] | np.isnan(cutnansumd_flux[1:-1]))
            cutmask = np.append(np.append([True],cutmask),[True])
            ## below is an approximation, it would be rigourously true for sigma = 0.5
            unc = np.sqrt( (unc[2:]*unc[2:]/(0.16*0.16)) + \
                                              (unc[1:-1]*unc[1:-1]/(0.68*0.68)) + \
                                              (unc[:-2]*unc[:-2]/(0.16*0.16)) )
            unc = unc / ((2/(0.16*0.16))+(1/(0.68*0.68)))
            cut_nansumd_unc = np.append(np.append(unc[0],unc),unc[-1])

            out_data_table.add_column(Table.Column(name=col, data=cutnansumd_flux[::2]))
            out_mask_table.add_column(Table.Column(name=col, data=cutmask[::2]))
            out_uncmult_table.add_column(Table.Column(name=col, data=cut_nansumd_unc[::2]))

        # plt.subplots(2,1)
        plt.figure()
        if self.save_plots or self.show_plots:
            for col in out_data_table.colnames:
                flux = out_data_table[col]
                unmasked = np.bitwise_not(out_mask_table[col].data)
                stds = out_uncmult_table[col]
                # plt.subplot(211)
                # plt.plot(ref_wavearrays[col][np.bitwise_not(out_mask_table[col])],out_data_table[col][np.bitwise_not(out_mask_table[col])],alpha=0.4,lw=1)
                # plt.title("Masked cam: {}".format(cam))
                # plt.subplot(212)
                if len(flux) == 0 or np.nansum(flux) == 0:
                    print("Empty flux in {}!".format(col))
                if np.any(np.isnan(stds[unmasked])):
                    print("There were nans that were unmasked in std in {}!".format(col))
                if np.any(np.isnan(flux[unmasked])):
                    print("There were nans that were unmasked in flux in {}!".format(col))
                plt.plot(master_wave_grid[::2][unmasked],flux[unmasked],alpha=0.1)
                #plt.fill_between(master_wave_grid[unmasked],flux[unmasked]-stds[unmasked],flux[unmasked]+stds[unmasked],alpha=0.2)
                plt.title("Bad Data Masked, cam: {}".format(cam))
        if self.save_plots:
            plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='science',step='combine',comment='_all_1d_unmasked'),dpi=600)
        if self.show_plots:
            plt.show()
        plt.close()

        plt.figure()
        if self.save_plots or self.show_plots:
            for col in out_data_table.colnames:
                flux = out_data_table[col]

                plt.plot(master_wave_grid[::2],flux,alpha=0.1)
                #plt.fill_between(master_wave_grid[unmasked],flux[unmasked]-stds[unmasked],flux[unmasked]+stds[unmasked],alpha=0.2)
                plt.title("All Data cam: {}".format(cam))
        if self.save_plots:
            plt.savefig(self.filemanager.get_saveplot_template(cam=cam,ap='',imtype='science',step='combine',comment='_all_1d_masked'),dpi=600)
        if self.show_plots:
            plt.show()
        plt.close()

        wavemax_new = master_wave_grid[::2][-1]
        sci_header['wavemin'],sci_header['wavemax'],sci_header['wavestep'] = wavemin,wavemax_new,(2.*wavestep)
        mask_header = mask_hdu.header
        mask_header['wavemin'], mask_header['wavemax'], mask_header['wavestep'] = wavemin, wavemax_new, (2.*wavestep)
        self.all_hdus[(cam,'combined','science',None)] = fits.BinTableHDU(data=out_data_table,header=sci_header,name='FLUX')
        self.all_hdus[(cam, 'combined', 'masks', None)] = fits.BinTableHDU(data=out_mask_table,header=mask_header, name='MASK')


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
                if 'RA_targeted' in self.mtl.colnames and (('RA' not in self.mtl.colnames) or (type(self.mtl['RA']) is Table.MaskedColumn and self.mtl['RA'].mask[fibrow])):
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
                if 'RA_targeted' in self.mtl.colnames and (('RA' not in self.mtl.colnames) or (type(self.mtl['RA']) is Table.MaskedColumn and self.mtl['RA'].mask[fibrow])):
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
        from scipy.ndimage import gaussian_filter

        from fit_redshifts import fit_redshifts_wrapper
        fluxes = Table(self.all_hdus[(cam, 'combined', 'science', None)].data)
        masks = Table(self.all_hdus[(cam, 'combined', 'masks', None)].data)
        # if (cam, 'combined_mask', 'science', None) in self.all_hdus.keys():
        #     mask = Table(self.all_hdus[(cam, 'combined_mask', 'science', None)].data)
        # else:
        #     mask = None

        header = self.all_hdus[(cam, 'combined', 'science', None)].header
        wave_grid = self.generate_wave_grid(header)
        if self.single_core:
            sci_data = OrderedDict()
            for fib in fluxes.colnames:
                sci_data[fib] = (wave_grid, fluxes[fib], masks[fib])
            obs1 = {
                'sky_subd_sciences':sci_data, 'mask_name': self.filemanager.maskname, 'savetemplate_func': self.filemanager.get_saveplot_template, 'run_auto': True
            }
            results = fit_redshifts_wrapper(obs1)
        else:
            sci_data1 = OrderedDict()
            fib1s = self.instrument.lower_half_fibs[cam]

            for fib in fib1s:
                if fib in fluxes.colnames:
                    sci_data1[fib] = (wave_grid, fluxes[fib], masks[fib])

            sci_data2 = OrderedDict()
            fib2s = self.instrument.upper_half_fibs[cam]
            for fib in fib2s:
                if fib in fluxes.colnames:
                    sci_data2[fib] = (wave_grid, fluxes[fib], masks[fib])

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

