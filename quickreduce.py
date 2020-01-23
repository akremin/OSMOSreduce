#!/C/Users/kremin/Anaconda3/python3.exe
# coding: utf-8

# Basic Walkthrough:
#     1) Define everything, load the bias files, generate any directories that don't exist.
#     2) Create master bias file, Save master bias file
#     3) Open all other files, subtract the master bias, save (*c?.b.fits)
#     4) Stitch the four opamps together.
#     5) Remove cosmics from all file types except bias (*c?.bc.fits)
#     6) Use fibermaps or flats to identify apertures of each spectrum.
#     7) Use fitted apertures to cut out 2-d spectra in thar,comp,science,twiflat
#     8) Collapse the 2-d spectra to 1-d.
#     9) Perform a rough calibration using low-res calibration lamps.
#     10) Use the rough calibrations to identify lines in a higher density calibration lamp (e.g. ThAr).
#     11) Fit a fifth order polynomial to every spectrum of each camera for each exposure.
#     12) Save fits for use as initial guesses of future fits.
#     13) Create master skyflat file, save.
#     14) Open all remaining types and divide out master flat, then save (*c?.bcf.fits)
#     15) Create master sky spectra by using a median fit of all sky fibers
#     16) Remove continuums of the science and sky spectra and iteratively remove each sky line.
#     17) Subtract the sky continuum from the science continuum and add back to the science.
#     18) Combine the multiple sky subtracted, wavelength calibrated spectra from the multiple exposures.
#     19) Fit the combined spectra to redshfit using cross-correlation with SDSS templates.

# In[478]:

import matplotlib
matplotlib.use('Qt5Agg')
import warnings
warnings.filterwarnings("ignore", module="matplotlib")

import os
import sys
import pickle as pkl

from collections import OrderedDict
from pyM2FS.pyM2FS_funcs import digest_filenumbers,boolify,read_io_config,\
    read_obs_config,get_steps,make_mtlz_wrapper

from pyM2FS.FieldData import FieldData
from pyM2FS.pyM2FS_io import FileManager
from pyM2FS.instrument import InstrumentState

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
        from pyM2FS.create_merged_target_list import make_mtl
        make_mtl(io_config,filenumbers['science'][0],vizier_catalogs=['sdss12'], \
                   overwrite_field=False, overwrite_redshifts = False)

    ## Load the data and instantiate the pipeline functions within the data class
    data = FieldData(filenumbers, filemanager=filemanager, instrument=instrument,
                     startstep=start, pipeline_options=pipe_options,obj_info=obj_info)

    ## For all steps marked true, run those steps on the data
    for step,do_this_step in steps.items():
        do_step_bool = boolify(do_this_step)
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

    if boolify(pipe_options['make_mtlz']) and ((io_config['SPECIALFILES']['mtlz'].lower()) != 'none'):
        cams = instrument.cameras
        make_mtlz_wrapper(data, filemanager, io_config, step, do_step_bool, pipe_options, cams)
    else:
        return



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