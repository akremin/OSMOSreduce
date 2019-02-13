#!/C/Users/kremin/Anaconda3/python.exe
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
import getopt
import numpy as np
import pickle as pkl

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.ndimage.filters import median_filter
from quickreduce_funcs import digest_filenumbers

from FieldData import FieldData
from inputoutput import FileManager
from instrument import InstrumentState

import configparser

def pipeline(maskname=None,obs_config_name=None,io_config_name=None, pipe_config_name='pipeline_config.ini'):
    """
    Call the pipeline

    inputs: maskname=None
            obs_config_name=None    (becomes 'obs_{}.ini'.format(maskname) if None)
            io_config_name=None     (becomes 'io_{}.ini'.format(maskname) if None)
            pipe_config_name='pipeline_config.ini'

    * Requires EITHER maskname or (obs_config_name and io_config_name)
      - If maskname specified, the filenames of obs_* and io_* must have the format
        mentioned above
      - If just filenames specified, maskname will be taken from config files
      - If both maskname and filenames are specified, it is up to you to ensure they
        are consistent
    * If pipeline_config_name is not specified, the default of "pipeline_config.ini
       should be present

    """
    if maskname is None and (obs_config_name is None or io_config_name is None):
        raise(IOError,"I don't know the necessary configuration file information. Exiting")
    if obs_config_name is None:
        obs_config_name = 'obs_{}.ini'.format(maskname)
    if io_config_name is None:
        io_config_name = 'io_{}.ini'.format(maskname)

    pipe_config = configparser.ConfigParser()
    pipe_config.read(pipe_config_name)

    obs_config = configparser.ConfigParser()
    obs_config.read(obs_config_name)

    io_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    io_config.read(io_config_name)

    # ###         Beginning of Code
    steps = OrderedDict(pipe_config['STEPS'])
    pipe_config.remove_section('STEPS')
    start = str(list(steps.keys())[0])
    for key,val in steps.items():
        if val.upper()=='TRUE':
            start = key
            break

    ## Interpret the filenumbers specified in the configuration files
    str_filenumbers = OrderedDict(obs_config['FILENUMBERS'])
    obs_config.remove_section('FILENUMBERS')
    filenumbers = digest_filenumbers(str_filenumbers)

    ## Load the filemanager and instrument status based on the configuration files
    filemanager = FileManager( io_config )
    instrument = InstrumentState( obs_config )

    ## Get specific pipeline options
    pipe_options = dict(pipe_config['PIPE_OPTIONS'])

    ## Load the data and instantiate the pipeline functions within the data class
    data = FieldData(filenumbers, filemanager=filemanager, instrument=instrument,
                     startstep=start, pipeline_options=pipe_options)

    ## For all steps marked true, run those steps on the data
    for step,do_this_step in steps.items():
        if do_this_step.lower()=='false':
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
        if step != 'cr_remove':
            try:
                data.write_all_filedata()#step=step)
            except:
                outfile = os.path.join(io_config['PATHS']['data_product_loc'], io_config['FILETEMPLATES']['pickled_datadump'])
                print("Save data failed to complete. Dumping data to {}".format(outfile))
                with open(outfile,'wb') as crashsave:
                    pkl.dump(data.all_hdus,crashsave)
                raise()





def parse_command_line(argv):
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option("-m", "--maskname",
                      action="store", type="string", dest="maskname")
    parser.add_option("-i", "--instfile",
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
        input_variables = parse_command_line()
    else:
        input_variables = {'maskname': 'A02'}
    pipeline(**input_variables)