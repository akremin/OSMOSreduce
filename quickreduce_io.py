import os

import numpy as np
from astropy.io import fits
from astropy.table import Table


class DirectoryManager:
    def __init__(self,conf, startstep='bias'):
        self.dirnames = dict(conf['DIRS'])
        PATHS = dict(conf['PATHS'])
        self.raw_data_loc = os.path.abspath(PATHS['raw_data_loc'])
        self.data_product_loc = os.path.abspath(PATHS['data_product_loc'])

        if not os.path.exists(self.raw_data_loc):
            print("The raw data directory doesn't exist. If you're performing bias subtracting, this will lead to an error.")
        if not os.path.exists(self.data_product_loc):
            os.makedirs(self.data_product_loc)

        ## Setup Defaults
        self.current_read_dir = None
        self.current_write_dir = None

        dirnms = self.dirnames
        self.catalog_path    = os.path.abspath(PATHS['catalog_loc'])
        self.lampline_dir    = os.path.abspath(PATHS['lampline'])

        self.calibration_dir = os.path.abspath(os.path.join(PATHS['data_product_loc'],dirnms['calibration']))
        if not os.path.exists(self.calibration_dir):
            os.makedirs(self.calibration_dir)

        self.default_calibration_dir = os.path.abspath(PATHS['default_calibration'])
        if not os.path.exists(self.default_calibration_dir):
            os.makedirs(self.default_calibration_dir)

        self.base_plot_dir = os.path.join(self.data_product_loc,dirnms['save_plots'])
        self.plot_dir = self.base_plot_dir
        if not os.path.exists(self.base_plot_dir):
            os.makedirs(self.base_plot_dir)

        self.mtl_path       = os.path.join(self.catalog_path,dirnms['mtl'])
        if not os.path.exists(self.mtl_path):
            os.makedirs(self.mtl_path)

        self.dirname_dict = {
                                'bias':        {'read':dirnms['raw'],      'write':dirnms['debiased']},\
                                'stitch':      {'read':dirnms['debiased'], 'write':dirnms['stitched']},\
                                'remove_crs':  {'read':dirnms['stitched'], 'write':dirnms['products']},\
                                'apcut':       {'read':dirnms['products'], 'write':dirnms['oned']}, \
                                'wavecalib':   {'read':dirnms['oned'],     'write':dirnms['oned']},\
                                'flatten':     {'read':dirnms['oned'],     'write':dirnms['calibd1d']}, \
                                'skysub':      {'read':dirnms['calibd1d'], 'write':dirnms['calibd1d']}, \
                                'combine':     {'read':dirnms['calibd1d'], 'write':dirnms['final1d']},\
                                'zfit':        {'read':dirnms['final1d'],  'write':dirnms['zfit']}\
                            }

        self.step = startstep
        #self.verify_files_exist()
        self.update_dirs_for()

        if not os.path.exists(self.calibration_dir):
            os.makedirs(self.calibration_dir)
            print("Calibration folder created: {}".format(self.calibration_dir))

    def verify_files_exist(self):
        for dirnam in self.dirnames.values():
            fullname = os.path.join(self.data_product_loc, dirnam)
            if not os.path.exists(fullname):
                os.mkdir(fullname)

    def update_dirs_for(self,step=None):
        if step is None:
            step = self.step
        else:
            self.step = step
            print("Setting internal dirs for step: {}".format(step))

        if step not in self.dirname_dict.keys():
            print("{} not understood. No directory updates performed.\nPossible steps: {}".format(step,self.dirname_dict.keys()))

        readdir = self.dirname_dict[step]['read']
        writedir = self.dirname_dict[step]['write']

        if step in ['wavecalib','skysub','zfit']:
            self.plot_dir = os.path.join(self.base_plot_dir,step)
            if not os.path.exists(self.plot_dir):
                os.makedirs(self.plot_dir)
        else:
            self.plot_dir = self.base_plot_dir

        if step == 'bias':
            if os.path.exists(self.raw_data_loc):
                self.current_read_dir = self.raw_data_loc
            else:
                print("Couldn't locate raw data directory: {},\nnow looking in: {}".format(
                    self.raw_data_loc,
                    os.path.join(self.data_product_loc, readdir)))
                self.current_read_dir = os.path.join(self.data_product_loc, readdir)
        else:
            self.current_read_dir = os.path.join(self.data_product_loc, readdir)
        self.current_write_dir =  os.path.join(self.data_product_loc, writedir)

        if not os.path.exists(self.current_read_dir):
            os.makedirs(self.current_read_dir)
            print("Read folder created: {}".format(self.current_read_dir))
            print("! -- > WARNING: This may mean we are about to try to read files from an empty directory, which will not end well")
        if not os.path.exists(self.current_write_dir):
            os.makedirs(self.current_write_dir)
            print("Write folder created: {}".format(self.current_write_dir))

        print("write location changed for step {} to:\n read={}\n write={}".format(step,self.current_read_dir,self.current_write_dir))



class FileManager:
    def __init__(self, conf, startstep='bias'):
        ## TODO  get the configuration values propogated to here
        GENERAL, SPECIALFILES = conf['GENERAL'], conf['SPECIALFILES']
        flnm_tmplt = conf['FILETEMPLATES']
        flnm_tags = conf['FILETAGS']

        self.date_timestamp = np.datetime_as_string(np.datetime64('today', 'D'))
        self.numeric_timestamp = np.datetime64('now' ,'m').astype(int )-np.datetime64('2018-08-01T00:00' ,'m').astype(int)

        self.directory = DirectoryManager(conf, startstep)
        self.maskname = GENERAL['mask_name']
        self.mtl_name = SPECIALFILES['mtl']

        ## Setup Defaults
        self.current_read_template = None
        self.current_write_template = None

        self.calibration_template = flnm_tmplt['calibration']
        self.default_calibration_template = flnm_tmplt['default_calibration']
        self.pickled_datadump_name =flnm_tmplt['pickled_datadump']
        self.lampline_template = flnm_tmplt['lampline']
        self.redshift_fit_template = flnm_tmplt['redshift_fits']
        self.save_plot_template = flnm_tmplt['save_plots']

        self.tempname_dict = {
                                'bias': {'read': flnm_tmplt['raw'], 'write': flnm_tmplt['debiased']},
                                'stitch': {'read': flnm_tmplt['debiased'], 'write': flnm_tmplt['stitched']},\
                                'remove_crs': {'read': flnm_tmplt['stitched'], 'write': flnm_tmplt['stitched']}, \
                                'apcut': {'read': flnm_tmplt['stitched'], 'write': flnm_tmplt['oneds']}, \
                                'wavecalib': {'read': flnm_tmplt['oneds'], 'write': flnm_tmplt['oneds']}, \
                                'flatten': {'read': flnm_tmplt['oneds'], 'write': flnm_tmplt['oneds']}, \
                                'skysub': {'read': flnm_tmplt['oneds'], 'write': flnm_tmplt['oneds']}, \
                                'combine': {'read': flnm_tmplt['oneds'], 'write': flnm_tmplt['combined']}, \
                                'zfit': {'read': flnm_tmplt['combined'], 'write': flnm_tmplt['combined']} \
                             }


        self.tags_dict =  {
                                'bias':        {'read': '', 'write': flnm_tags['debiased']}, \
                                'stitch':      {'read':flnm_tags['debiased'], 'write': flnm_tags['debiased']},\
                                'remove_crs':  {'read':flnm_tags['debiased'], 'write': flnm_tags['crrmvd']},\
                                'apcut':       {'read':flnm_tags['crrmvd'],   'write': flnm_tags['crrmvd']}, \
                                'wavecalib':   {'read':flnm_tags['crrmvd'],   'write': flnm_tags['crrmvd']},\
                                'flatten':     {'read':flnm_tags['crrmvd'],   'write': flnm_tags['flatnd']}, \
                                'skysub':      {'read':flnm_tags['flatnd'],   'write': flnm_tags['skysubd']}, \
                                'combine':     {'read':flnm_tags['skysubd'],  'write': flnm_tags['skysubd']},\
                                 'zfit':       {'read':flnm_tags['skysubd'],  'write': flnm_tags['skysubd']}\
                            }

        self.step = startstep
        self.update_templates_for()

    def update_templates_for(self, step=None):
        if step is None:
            step = self.step
        elif step not in self.tempname_dict.keys():
            print("{} not understood. No updates performed.\nPossible steps: {}".format(step,  self.tempname_dict.keys()))
            print("Using the currently saved step: {}".format(self.step))
            step = self.step
        elif step in self.tempname_dict.keys():
            self.step = step
            print("Setting internal file formats for step: {}".format(step))

        ## Update the directory
        self.directory.update_dirs_for(step)

        ## Update the templates
        self.current_read_template_base = self.tempname_dict[step]['read']
        self.current_write_template_base = self.tempname_dict[step]['write']
        self.current_read_tags = self.tags_dict[step]['read']
        self.current_write_tags = self.tags_dict[step]['write']

        self.current_read_template = self.current_read_template_base + self.current_read_tags + '.fits'
        self.current_write_template = self.current_write_template_base + self.current_write_tags + '.fits'

        print("Template changed for step {} to:\n read={}\n write={}".format(step, self.current_read_template,\
                                                                             self.current_write_template))

    def get_saveplot_template(self,cam='',ap='',imtype='',step='',comment=''):
        basic_template = self.save_plot_template.format(cam=cam,ap=ap,imtype=imtype,\
                                                        step=step,comment=comment)
        return os.path.join(self.directory.plot_dir,basic_template)

    def get_read_filename(self,camera, imtype, filenum, amp):
        inname = self.current_read_template.format(cam=camera, imtype=imtype, filenum=filenum,\
                                                       opamp=amp, maskname=self.maskname)
        filename = os.path.join(self.directory.current_read_dir, inname)
        return filename

    def get_write_filename(self,camera, imtype, filenum =None,amp=None):
        if imtype != 'zfit':
            outname = self.current_write_template.format(cam=camera, imtype=imtype, filenum=filenum,\
                                                             opamp=amp, maskname=self.maskname)
        else:
            outname = self.redshift_fit_template.format(cam=camera,maskname=self.maskname)
        filename = os.path.join(self.directory.current_write_dir, outname)
        return filename


    def write_hdu(self,outhdu, camera='r', filenum=999,imtype='comp',amp=None, step=None, history=None):
        if history is not None:
            outhdu.header.add_history(history)
        outhdu.header.add_history("wrote by M2FS reduce after step {} on {}".format(step,self.date_timestamp))

        filename = self.get_write_filename(camera=camera,imtype=imtype,filenum=filenum,amp=amp)

        outhdu.writeto(filename, overwrite=True)


    # def write_zfit(self,outhdu, camera='r', step=None, history=None):
    #     if history is not None:
    #         outhdu.header.add_history(history)
    #     outhdu.header.add_history("wrote by M2FS reduce after step {} on {}".format(step,self.date_timestamp))
    #
    #     filename = self.get_write_filename(camera=camera,imtype='zfit')
    #
    #     outhdu.writeto(filename, overwrite=True)


    def read_hdu(self,camera='r', filenum=999,imtype='comp',amp=None,fibersplit=False):
        filename = self.get_read_filename(camera=camera,imtype=imtype,filenum=filenum,amp=amp)
        if not os.path.exists(filename):
            print("\n\n\n\n\n------->Couldn't find: {}\n\n\n".format(filename))
        if imtype != 'masks' and os.path.getsize(filename) < 1e6:
            print("\n\n\n\n\n------->Small filesize: {},  {}\n\n\n".format(filename,os.path.getsize(filename)))
        inhdulist = fits.open(filename,memmap=False)
        if len(inhdulist)>1:
            if 'FLUX' in inhdulist:
                inhdu = inhdulist['FLUX']
            elif 'WAVE' in inhdulist:
                inhdu = inhdulist['WAVE']
            elif 'MASK' in inhdulist:
                inhdu = inhdulist['MASK']
            else:
                inhdu = inhdulist[1]
        else:
            inhdu = inhdulist[0]

        if not fibersplit:
            if inhdu.header['SHOE'].lower() != camera.lower():
                print("WARNING: camera didn't match the shoe name in the read header")
            if amp is not None and inhdu.header['OPAMP'] != amp:
                print("WARNING: opamp didn't match the name in the read header")
        return inhdu

    def load_calib_dict(self,fittype,cam,config,filenum=None,timestamp=None):
        if fittype == 'default':
            filename = self.default_calibration_template.format(cam=cam,config=config)
            fullpathname = os.path.join(self.directory.default_calibration_dir,filename)
            if os.path.exists(fullpathname):
                calib = Table.read(fullpathname,format='fits')
            else:
                calib = None
        elif 'basic' in fittype:
            filename = self.calibration_template.format(cam=cam, fittype=fittype, config=config, \
                                                        filenum=filenum, timestamp=timestamp)
            fullpathname = os.path.join(self.directory.calibration_dir, filename)
            calib = Table.read(fullpathname)
        else:
            filename = self.calibration_template.format(cam=cam, fittype=fittype, config=config, \
                                                                filenum=filenum, timestamp=timestamp)
            fullpathname = os.path.join(self.directory.calibration_dir,filename)
            calib = fits.open(fullpathname)

        return calib

    def save_basic_calib_dict(self, outtable, fittype, cam, config, filenum=None):
        if fittype == 'default':
            filename = self.default_calibration_template.format(cam=cam,config=config)
            fullpathname = os.path.join(self.directory.default_calibration_dir,filename)
            outtable.write(fullpathname,format='fits',overwrite=True)
        else:
            filename = self.calibration_template.format(cam=cam, fittype=fittype, config=config, \
                                                                filenum=filenum, timestamp=self.numeric_timestamp)
            fullpathname = os.path.join(self.directory.calibration_dir,filename)
            outtable.write(fullpathname,format='fits',overwrite=True)


    def save_full_calib_dict(self, outhdulist, fittype, cam, config, filenum=None):
        filename = self.calibration_template.format(cam=cam, fittype=fittype, config=config, \
                                                    filenum=filenum, timestamp=self.numeric_timestamp)
        fullpathname = os.path.join(self.directory.calibration_dir, filename)
        outhdulist.writeto(fullpathname, overwrite=True)


    def locate_calib_dict(self,fittype, camera, config, filenum,locate_type='any'):
        import re
        #if type(filenum) is not str:
        #    filenum = "{:04d}".format(int(filenum))
        if 'basic' in fittype:
            ftype = fittype.replace('basic','')
        elif 'full' in fittype:
            ftype = fittype.replace('full','')
        else:
            ftype = fittype
        if locate_type == 'basic':
            fittype = '(basic)' + ftype
            match_str = self.calibration_template.format(cam=camera, fittype=fittype, \
                                                     timestamp=r'(\d{6})', filenum=filenum,config=config)
        elif locate_type == 'full':
            fittype =  '(full)' + ftype
            match_str = self.calibration_template.format(cam=camera, fittype=fittype, \
                                                     timestamp=r'(\d{6})', filenum=filenum,config=config)
        else:
            fittype = '(full|basic)' + ftype
            match_str = self.calibration_template.format(cam=camera, fittype=fittype, \
                                                         timestamp=r'(\d{6})', filenum=filenum, config=config)
        matches,types=[],[]
        files = os.listdir(self.directory.calibration_dir)

        for fil in files:
            srch_res = re.search(match_str, fil)
            if srch_res:
                matches.append(int(srch_res.group(2)))
                types.append(str(srch_res.group(1)))

        calib,thetype = None,''
        if len(matches) > 0:
            newest = np.argmax(matches)
            newmatch = matches[newest]
            thetype = types[newest]
            calib = self.load_calib_dict(thetype+ftype, camera, config, filenum, newmatch)

        return calib, thetype

    def read_crashed_filedata(self):
        import pickle as pkl
        data_product_loc = self.directory.data_product_loc
        infile = os.path.join(data_product_loc, self.pickled_datadump_name)
        with open(infile,'rb') as crashdata:
            all_hdus = pkl.load(crashdata)
        return all_hdus

    def read_all_filedata(self,filenumber_dict,instrument,data_stitched=True,fibersplit=True):
        opamps = instrument.opamps
        cameras = instrument.cameras
        all_hdus = {}
        for imtype, filenums in filenumber_dict.items():
            for filnum in filenums:
                for camera in cameras:
                    for opamp in opamps:
                        all_hdus[(camera,filnum,imtype,opamp)] = self.read_hdu(camera=camera, filenum=filnum, imtype=imtype, amp=opamp, fibersplit=fibersplit)

        return all_hdus

    def write_all_filedata(self,all_hdus,step=''):
        # if step != 'zfit':
        for (camera,filnum,imtype,opamp),outhdu in all_hdus.items():
            self.write_hdu(outhdu=outhdu,camera=camera, filenum=filnum, imtype=imtype, amp=opamp, step=step)
        # else:
        #     for camera,hdu in all_hdus.items():
        #         self.write_zfit(outhdu=hdu,camera=camera,step=step)

    def load_calibration_lines_dict(self,cal_lamp,wavemincut=4000,wavemaxcut=10000,use_selected=False):
        """Assumes the format of the salt linelist csvs privuded with this package"""
        from calibration_funcs import air_to_vacuum
        #linelistdict = {}
        selectedlinesdict = {}
        print(('Using calibration lamps: ', cal_lamp))
        possibilities = ['Xe','Ar','HgNe','HgAr','NeAr','Hg','Ne','ThAr','Th']
        all_wms = []
        for lamp in possibilities:
            if lamp in cal_lamp:
                print(lamp)
                filname = self.lampline_template.format(mod='',lamp=lamp)
                sel_filname = self.lampline_template.format(mod='selected_',lamp=lamp)
                pathname = os.path.join(self.directory.lampline_dir,filname)
                sel_pathname = os.path.join(self.directory.lampline_dir,sel_filname)
                if use_selected and os.path.exists(sel_pathname):
                    tab = Table.read(sel_pathname,format='ascii.csv',dtypes=[float,float,str,str])
                else:
                    tab = Table.read(pathname, format='ascii.csv')
                fm = tab['Intensity'].data
                wm_vac = air_to_vacuum(tab['Wavelength'].data)
                boolean = np.array([val.lower()=='y' for val in tab['Use']]).astype(bool)
                ## sort lines by wavelength
                sortd = np.argsort(wm_vac)
                srt_wm_vac, srt_fm, srt_bl = wm_vac[sortd], fm[sortd],boolean[sortd]
                good_waves = np.where((srt_wm_vac>=wavemincut)&(srt_wm_vac<=wavemaxcut))[0]
                out_wm_vac,out_fm_vac,out_bl = srt_wm_vac[good_waves], srt_fm[good_waves],srt_bl[good_waves]
                #linelistdict[lamp] = (out_wm_vac,out_fm_vac)
                selectedlinesdict[lamp] = (out_wm_vac[out_bl],out_fm_vac[out_bl])
                all_wms.extend(out_wm_vac.tolist())

        #return linelistdict, selectedlinesdict, all_wms
        return selectedlinesdict, np.asarray(all_wms)

    def get_matched_target_list(self):
        full_name = os.path.join(self.directory.mtl_path,self.mtl_name)
        full_name = full_name + '_full.csv'
        if os.path.exists(full_name):
            try:
                mtl = Table.read(full_name, format='ascii.csv', \
                                  include_names=['ID','TARGETNAME','FIBNAME','sdss_SDSS12','RA','DEC',\
                                                 'RA_targeted','DEC_targeted','sdss_zsp','sdss_zph','sdss_rmag','MAG'])
                return mtl
            except:
                print("Failed to open merged target list, but it did exist")
                return None
        else:
            print("Could not find the merged target list file at: {}".format(full_name))
            return None
