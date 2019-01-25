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
import numpy as np
import pickle as pkl

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.ndimage.filters import median_filter

from FieldData import FieldData
from inputoutput import FileManager
from instrument import InstrumentState

import configparser

def pipeline(maskname=None,obs_config_name=None,io_config_name=None, pipe_config_name='pipeline_config.ini'):
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

    str_filenumbers = OrderedDict(obs_config['FILENUMBERS'])
    obs_config.remove_section('FILENUMBERS')
    filenumbers = digest_filenumbers(str_filenumbers)

    filemanager = FileManager( io_config )
    instrument = InstrumentState( obs_config )
    pipe_options = dict(pipe_config['PIPE_OPTIONS'])

    data = FieldData(filenumbers, filemanager=filemanager, instrument=instrument,
                     startstep=start, pipeline_options=pipe_options)

    for step,do_this_step in steps.items():
        if do_this_step.lower()=='false':
            print("\nSkipping {}".format(step))
            continue

        print("\nPerforming {}:".format(step))
        data.proceed_to(step=step)
        data.check_data_ready_for_current_step()#step=step)

        data.run_step()#step=step)

        if step != 'cr_remove':
            try:
                data.write_all_filedata()#step=step)
            except:
                outfile = os.path.join(io_config['PATHS']['data_product_loc'], io_config['FILETEMPLATES']['pickled_datadump'])
                print("Save data failed to complete. Dumping data to {}".format(outfile))
                with open(outfile,'wb') as crashsave:
                    pkl.dump(data.all_hdus,crashsave)
                raise()




def digest_filenumbers(str_filenumbers):
    out_dict = {}
    for key,strvals in str_filenumbers.items():
        out_num_list = []
        if ',' in strvals:
            vallist = str(strvals).strip('[]() \t').split(',')

            for strval in vallist:
                if '-' in strval:
                    start,end = strval.split('-')
                    for ii in range(int(start),int(end)+1):
                        out_num_list.append(int(ii))
                else:
                    out_num_list.append(int(strval))
        elif '-' in strvals:
            start,end = str(strvals).split('-')
            for ii in range(int(start),int(end)+1):
                out_num_list.append(int(ii))
        elif strvals.isnumeric:
            out_num_list.append(int(strvals))
        out_dict[key] = np.sort(out_num_list)

    return out_dict




if __name__ == '__main__':
    maskname = 'A02'
    pipeline(maskname=maskname)