#!/C/Users/kremin/Anaconda3/python3.exe
# coding: utf-8

# Basic Walkthrough:
#      1) Define everything
#      2) Create master bias file, Save master bias file  
#      3) Open all other files, sub master bias, save  (*c?.b.fits)
#      4) Remove cosmics from all file types except bias  (*c?.bc.fits)
#      5) Open flats and create master skyflat file, save
#      6) Open all remainging types and divide out master flat, then save  (*c?.bcf.fits)
#      7) Open all remaining types and stitch together, save  (*full.bcf.fits)
#      8) Use fibermap files to determine aperatures
#      9) Use aperatures to cut out same regions in thar,comp,science
#      10) Save all 256 to files with their header tagged name in filename, along with fiber num
#      11) Assume no curvature within tiny aperature region; fit profile to 2d spec and sum to 1d
#      12) Fit the lines in comp spectra, save file and fit solution
#      13) Try to fit lines in thar spectra, save file and fit solution
#      14) Apply to science spectra

# In[478]:

import matplotlib
matplotlib.use('Qt5Agg')

import os
import sys
import pickle as pkl

from collections import OrderedDict
from quickreduce_funcs import digest_filenumbers

from FieldData import FieldData
from quickreduce_io import FileManager
from instrument import InstrumentState

import configparser

def pipeline(maskname=None,obs_config_name=None,io_config_name=None, pipe_config_name='./configs/pipeline.ini'):
    """
    Call the pipeline

    inputs: maskname=None
            obs_config_name=None    (becomes 'obs_{}.ini'.format(maskname) if None)
            io_config_name=None     (becomes 'io_{}.ini'.format(maskname) if None)
            pipe_config_name='pipeline_config.ini'

    *
      - If maskname specified, the filenames of obs_* and io_* must have the format
        mentioned above
      - If just filenames specified, maskname will be taken from config files
      - If both maskname and filenames are specified, it is up to you to ensure they
        are consistent
    * If pipeline_config_name is not specified, the default of "pipeline.ini"
       should be present

    """
    if '/configs' not in pipe_config_name and not os.path.exists(pipe_config_name):
        pipe_config_name = './configs/{}'.format(pipe_config_name)

    pipe_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    pipe_config.read(pipe_config_name)

    if maskname is not None:
        pipe_config['GENERAL']['mask_name'] = maskname
    else:
        try:
            maskname = pipe_config['GENERAL']['mask_name']
        except:
            raise (IOError, "I don't know the necessary configuration file information. Exiting")

    ## Check that the configs are there or defined in the pipeline conf file
    ## read in the confs if they are there, otherwise return none
    obs_config = read_obs_config(obs_config_name, pipe_config['CONFS'], maskname)
    io_config = read_io_config(io_config_name, pipe_config, maskname)

    ####         Beginning of Code
    ## Ingest steps and determine where to start
    steps, start, pipe_config = get_steps(pipe_config)

    ## Interpret the filenumbers specified in the configuration files
    str_filenumbers = OrderedDict(obs_config['FILENUMBERS'])
    obs_config.remove_section('FILENUMBERS')
    filenumbers = digest_filenumbers(str_filenumbers)

    ## Load the filemanager and instrument status based on the configuration files
    filemanager = FileManager( io_config )
    instrument = InstrumentState( obs_config )
    obj_info = obs_config['TARGET']

    ## Get specific pipeline options
    pipe_options = dict(pipe_config['PIPE_OPTIONS'])

    if boolify(pipe_options['make_mtl']) and \
            io_config['SPECIALFILES']['mtl'].lower() != 'none':
        from create_merged_target_list import create_mtl
        create_mtl(io_config,filenumbers['science'][0],vizier_catalogs=['sdss12'], \
                   overwrite_field=False, overwrite_redshifts = False)

    ## Load the data and instantiate the pipeline functions within the data class
    data = FieldData(filenumbers, filemanager=filemanager, instrument=instrument,
                     startstep=start, pipeline_options=pipe_options,obj_info=obj_info)

    ## For all steps marked true, run those steps on the data
    for step,do_this_step in steps.items():
        do_step_bool = (do_this_step.lower() == 'true')
        if not do_step_bool:
            print("\nSkipping {}".format(step))
            continue

        ## Get ready for the requested step
        print("\nPerforming {}:".format(step))
        data.proceed_to(step=step)

        ## Check that the currently loaded data is relevant to the
        ## requested step. If not, it will raise an error
        data.check_data_ready_for_current_step()#step=step)

        ## run the step
        data.run_step()#step=step)

        ## cosmic ray step autosaves during the process
        ## for other steps, save the results
        if step not in ['cr_remove','wave_calib']:
            try:
                data.write_all_filedata()#step=step)
            except:
                outfile = os.path.join(io_config['PATHS']['data_product_loc'], io_config['FILETEMPLATES']['pickled_datadump'])
                print("Save data failed to complete. Dumping data to {}".format(outfile))
                with open(outfile,'wb') as crashsave:
                    pkl.dump(data.all_hdus,crashsave)
                raise()
    ##HACK!!

    if boolify(pipe_options['make_mtlz']) and ((io_config['SPECIALFILES']['mtlz'].lower()) != 'none'):
        cams = instrument.cameras
        imtype = 'zfits'
        if not do_step_bool:
            step == 'zfit'
            data.proceed_to(step)
            write_dir = filemanager.directory.current_write_dir
            write_template = filemanager.current_write_template

            if os.path.exists(os.path.join(write_dir,write_template.format(imtype=imtype,cam=cams[0]))):
                hdus = {}
                from astropy.io import fits
                hdus = [fits.open(os.path.join(write_dir,write_template.format(imtype=imtype,cam=cams[0])))['ZFITS']]
                if len(cams)>1:
                    hdus.append(fits.open(os.path.join(write_dir,write_template.format(imtype=imtype,cam=cams[1])))['ZFITS'])
                else:
                    hdus.append(None)
            else:
                return
        else:
            hdus = [data.all_hdus[(cams[0], None, imtype, None)]]
            if len(cams) == 2:
                hdus.append(data.all_hdus[(cams[1], None, imtype, None)])
            else:
                hdus.append(None)

        find_extra_redshifts = boolify(pipe_options['find_extra_redshifts'])
        mtlz_path = os.path.join(io_config['PATHS']['catalog_loc'],io_config['DIRS']['mtl'])
        mtlz_name = io_config['SPECIALFILES']['mtlz']
        outfile = os.path.join(mtlz_path,mtlz_name)
        from create_merged_target_list import make_mtlz
        make_mtlz(data.mtl, hdus, find_extra_redshifts,outfile=outfile,\
                  vizier_catalogs = ['sdss12'])
    else:
        return



def get_steps(pipe_config):
    ## Ingest steps and determine where to start
    steps = OrderedDict(pipe_config['STEPS'])
    pipe_config.remove_section('STEPS')
    start = str(list(steps.keys())[-1])
    for key,val in steps.items():
        if boolify(val):
            start = key
            break
    return steps, start, pipe_config

def check_confname(confname, conftype, pipe_config_dict, maskname):
    if confname is None:
        if conftype in pipe_config_dict.keys():
            confname = pipe_config_dict[conftype]
        else:
            confname = './configs/{}_{}.ini'.format(conftype[:-4],maskname)
    if '/configs' not in confname and not os.path.exists(confname):
        confname = './configs/{}'.format(confname)
    return confname

def read_obs_config(obs_config_name, pipe_config_dict, maskname):
    obs_config_name = check_confname(obs_config_name, 'obsconf', pipe_config_dict, maskname)
    if os.path.exists(obs_config_name):
        obs_config = configparser.ConfigParser()
        obs_config.read(obs_config_name)
        return obs_config
    else:
        return None

def read_io_config(io_config_name, pipe_dict, maskname):
    io_config_name = check_confname(io_config_name, 'ioconf', pipe_dict['CONFS'], maskname)
    if os.path.exists(io_config_name):
        io_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        io_config.add_section('GENERAL')
        io_config['GENERAL']['mask_name'] = maskname
        if 'path_to_masks' in pipe_dict['GENERAL'].keys() and str(
                pipe_dict['GENERAL']['path_to_masks']).lower() != 'none':
            io_config.add_section('PATHS')
            io_config['PATHS']['path_to_masks'] = pipe_dict['GENERAL']['path_to_masks']
        if 'raw_data_loc' in pipe_dict['GENERAL'].keys() and str(
                pipe_dict['GENERAL']['raw_data_loc']).lower() != 'none':
            if 'PATHS' not in io_config.sections():
                io_config.add_section('PATHS')
            io_config['PATHS']['raw_data_loc'] = pipe_dict['GENERAL']['raw_data_loc']
        io_config.read(io_config_name)
        return io_config
    else:
        return None

def boolify(parameter):
    return (str(parameter).lower()=='true')

def parse_command_line(argv):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--maskname",
                      action="store", type="string", dest="maskname")
    parser.add_option("-i", "--iofile",
                      action="store", type="string", dest="io_config_name")
    parser.add_option("-o", "--obsfile",
                      action="store", type="string", dest="obs_config_name")
    parser.add_option("-p", "--pipefile",
                      action="store", type="string", dest="pipe_config_name")
    if argv is None:
        (options, args) = parser.parse_args()
    else:
        (options, args) = parser.parse_args(argv)

    return options.__dict__



if __name__ == '__main__':
    if len(sys.argv)>1:
        print("Detected input parameters: ",sys.argv[1:])
        input_variables = parse_command_line(sys.argv)
        nonvals = []
        for key,val in input_variables.items():
            if val is None:
                nonvals.append(key)
        for key in nonvals:
            input_variables.pop(key)
        print("Received input variables: ", input_variables)
    else:
        input_variables = {}
    pipeline(**input_variables)