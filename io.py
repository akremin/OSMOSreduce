import os
from astropy.io import fits
#from quickreduce_funcs import print_data_neatly
import numpy as np
from astropy.table import Table


class DirectoryManager:
    def __init__(self,raw_data_loc,data_product_loc):
        self.raw_data_loc = os.path.abspath(raw_data_loc)
        self.data_product_loc = os.path.abspath(data_product_loc)

        if not os.path.exists(self.raw_data_loc):
            raise(IOError,"The raw data directory doesn't exist")
        if not os.path.exists(self.data_product_loc):
            os.makedirs(self.data_product_loc)

        ## Setup Defaults
        self.current_read_dir = None
        self.current_write_dir = None
        self.calibration_dir = os.path.join(data_product_loc,'calibrations')
        self.calibration_dir = os.path.join(os.path.abspath('./'),'calibrations')

        self.dirname_dict = {
                                'stitch':      {'read':'raw_stitched','write':'data_products'},\
                                'bias':        {'read':'data_products','write':'data_products'},\
                                'remove_crs':  {'read':'data_products','write':'data_products'},\
                                'ffmerge':     {'read':'data_products','write':'data_products'},\
                                'apcut':       {'read':'data_products','write':'oneds'}, \
                                'wavecalib':   {'read':'oneds','write':'calibrated_oned'},\
                                'flat':        {'read':'calibrated_oned','write':'calibrated_oned'},\
                                'combine':     {'read':'calibrated_oned','write':'final_oned'},\
                                'zfit':        {'read':'final_oned','write':'zfits'}\
                            }

        self.step = 'stitch'
        self.update_dirs_for()

        if not os.path.exists(self.calibration_dir):
            os.makedirs(self.calibration_dir)
            print("Calibration folder created: {}".format(self.calibration_dir))

    def update_dirs_for(self,step=None):
        if step is None:
            step = self.step
        else:
            self.step = step
            print("Setting internal step to {}".format(step))

        if step not in self.dirname_dict.keys():
            print("{} not understood. No directory updates performed.\nPossible steps: {}".format(step,self.dirname_dict.keys()))

        readdir = self.dirname_dict[step]['read']
        writedir = self.dirname_dict[step]['write']

        self.current_read_dir =  os.path.join(self.data_product_loc, readdir)
        self.current_write_dir =  os.path.join(self.data_product_loc, writedir)

        if not os.path.exists(self.current_read_dir):
            os.makedirs(self.current_read_dir)
            print("write folder created: {}".format(self.current_read_dir))
        if not os.path.exists(self.current_write_dir):
            os.makedirs(self.current_write_dir)
            print("Write folder created: {}".format(self.current_write_dir))

        print("write location changed for step {} to:\n read={}\n write={}".format(step,self.current_read_dir,self.current_write_dir))



class FileManager:
    def __init__(self, raw_data_loc='./', data_product_loc='./', maskname='{maskname}'):
        self.date_timestamp = np.datetime_as_string(np.datetime64('today', 'D'))
        self.numeric_timestamp = np.datetime64('now' ,'m').astype(int ) -np.datetime64('2018-06-01T00:00' ,'m').astype(int)

        self.directory - DirectoryManager(raw_data_loc, data_product_loc)
        self.maskname = maskname

        ## Setup Defaults
        self.current_read_template = None
        self.current_write_template = None
        self.calibration_template = '{cam}_calibration_{fittype}_{config}_{filenum:04d}_{timestamp:05d}.fits'
        self.default_calibration_template = '{cam}_calibration_default_{config}.fits'

        flnm_tmplt = {}
        flnm_tmplt['raw'] = '{cam}{filenum:04d}c{opamp}.fits'
        flnm_tmplt['base'] = '{cam}_{imtype}_{filenum:04d}_' + maskname + '_'
        flnm_tmplt['stitched'] = flnm_tmplt['base'] + 'stitched'
        flnm_tmplt['twods'] = flnm_tmplt['base'] + '{fibername}_2d'
        flnm_tmplt['oneds'] = flnm_tmplt['base'] + '{fibername}_1d'
        flnm_tmplt['combined'] = flnm_tmplt['base'] + '{fibername}_combined_1d'
        flnm_tmplt['master'] = '{cam}_{imtype}_master_' + maskname

        self.filename_template = flnm_tmplt

        self.tempname_dict = {
                                'stitch': {'read': flnm_tmplt['raw'], 'write': flnm_tmplt['stitched']}, \
                                'bias': {'read': flnm_tmplt['stitched'], 'write': flnm_tmplt['stitched']}, \
                                'remove_crs': {'read': flnm_tmplt['stitched'], 'write': flnm_tmplt['stitched']}, \
                                'ffmerge': {'read': flnm_tmplt['stitched'], 'write': flnm_tmplt['stitched']}, \
                                'apcut': {'read': flnm_tmplt['stitched'], 'write': flnm_tmplt['oneds']}, \
                                'wavecalib': {'read': flnm_tmplt['oneds'], 'write': flnm_tmplt['oneds']}, \
                                'flat': {'read': flnm_tmplt['oneds'], 'write': flnm_tmplt['oneds']}, \
                                'combine': {'read': flnm_tmplt['oneds'], 'write': flnm_tmplt['combined']}, \
                                'zfit': {'read': flnm_tmplt['combined'], 'write': flnm_tmplt['combined']} \
                             }

        self.tags_dict =  {
                                'stitch':      {'read':'','write':''},\
                                'bias':        {'read':'','write':'_b'},\
                                'remove_crs':  {'read':'_b','write':'_bc'},\
                                'ffmerge':     {'read':'_bc','write':'_bc'},\
                                'apcut':       {'read':'_bc','write':'_bc'}, \
                                'wavecalib':   {'read':'_bc','write':'_bcw'},\
                                'flat':        {'read':'_bcw','write':'_bcwf'},\
                                'combine':     {'read':'_bcwf','write':'_bcwf'},\
                                'zfit':        {'read':'_bcwf','write':'_bcwf'}\
                            }

        self.step = 'stitch'
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
            print("Setting internal step to {}".format(step))

        ## Update the directory
        self.directory.update_dir_for(step)

        ## Update the templates
        self.current_read_template_base = self.tempname_dict[step]['read']
        self.current_write_template_base = self.tempname_dict[step]['write']
        self.current_read_tags = self.tags_dict[step]['read']
        self.current_write_tags = self.tags_dict[step]['write']

        self.current_read_template = self.current_read_template_base + self.current_read_tags + '.fits'
        self.current_write_template = self.current_write_template_base + self.current_write_tags + '.fits'

        print("Template changed for step {} to:\n read={}\n write={}".format(step, self.current_read_template,\
                                                                             self.current_write_template))

    def get_read_filename(self,camera, imtype, filenum, amp):
        if filenum!='master':
            inname = self.current_read_template.format(cam=camera, imtype=imtype,filenum=filenum,
                                                       amp=amp,maskname=self.maskname)
        else:
            inname = self.filename_template['master'].format(cam=camera, imtype=imtype, maskname=self.maskname)
            if imtype == 'bias':
                tag = ''
            else:
                tag = '_b'
            inname = inname + tag
        return inname

    def get_write_filename(self,camera, imtype, filenum,amp):
        outname = self.current_write_template.format(cam=camera, imtype=imtype,filenum=filenum,
                                                     amp=amp,maskname=self.maskname)
        filename = os.path.join(self.directory.current_write_dir, outname)
        return filename


    def write_hdu(self,outhdu, camera='r', filenum=999,imtype='comp',amp=None,history=None):
        if history is not None:
            outhdu.header.add_history(history)
        outhdu.header.add_history("writed by M2FS reduce on {}".format(self.date_timestamp))
        outname = self.current_write_template.format(cam=camera, imtype=imtype,filenum=filenum,
                                                     amp=amp,maskname=self.maskname)
        filename = os.path.join(self.directory.current_write_dir, outname)
        outhdu.writeto(filename, overwrite=True)

    def read_hdu(self,camera='r', filenum=999,imtype='comp',amp=None,fibersplit=False):
        if filenum!='master':
            inname = self.current_read_template.format(cam=camera, imtype=imtype,filenum=filenum,
                                                       amp=amp,maskname=self.maskname)
        else:
            inname = self.filename_template['master'].format(cam=camera, imtype=imtype, maskname=self.maskname)
            if imtype == 'bias':
                tag = ''
            else:
                tag = '_b'
            inname = inname + tag

        filename = os.path.join(self.directory.current_read_dir, inname)

        if fibersplit:
            inhdus = fits.open(filename)
            return inhdus
        else:
            inhdu = fits.open(filename)[0]

            if inhdu.header['SHOE'].lower() != camera.lower():
                print("WARNING: camera didn't match the shoe name in the read header")
            if amp is not None and inhdu.header['OPAMP'] != amp:
                print("WARNING: opamp didn't match the shoe name in the read header")
            return inhdu

    def load_calib_dict(self,fittype,cam,config,filenum=None,timestamp=None):
        if fittype == 'default':
            filename = self.default_calibration_template.format(cam=cam,config=config)
            fullpathname = os.path.join(self.directory.default_calibration_dir,filename)
            if os.path.exists(fullpathname):
                calib_tab = Table.read(fullpathname,format='fits')
            else:
                calib_tab = None
        else:
            filename = self.calibration_template.format(cam=cam, fittype=fittype, config=config, \
                                                                filenum=filenum, timestamp=timestamp)
            fullpathname = os.path.join(self.directory.calibration_dir,filename)
            calib_tab = Table.read(fullpathname,format='fits')

        return calib_tab

    def save_calib_dict(self, outtable, fittype, cam, config, filenum=None):
        if fittype == 'default':
            filename = self.default_calibration_template.format(cam=cam,config=config)
            fullpathname = os.path.join(self.directory.default_calibration_dir,filename)
            outtable.write(fullpathname,format='fits')
        else:
            filename = self.calibration_template.format(cam=cam, fittype=fittype, config=config, \
                                                                filenum=filenum, timestamp=self.numeric_timestamp)
            fullpathname = os.path.join(self.directory.calibration_dir,filename)
            outtable.write(fullpathname,format='fits')


    def locate_calib_dict(self,fittype, camera, config, filenum):
        import re
        calib_coef_table = None
        match_str = self.calibration_template.replace('{timestamp}.fits',r'(\d{5}).fits')
        files = os.listdir(self.directory.calibration_dir)
        matches = []
        for fil in files:
            srch_res = re.search(match_str, fil)
            if srch_res:
                matches.append(int(srch_res.group(1)))
            else:
                continue

        if len(matches) > 0:
            print(matches)
            newest = max(matches)
            calib_coef_table = self.load_calib_dict(fittype, camera, config, filenum, newest)
        else:
            calib_coef_table = None
        return calib_coef_table
