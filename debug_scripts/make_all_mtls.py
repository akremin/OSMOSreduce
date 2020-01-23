import matplotlib
matplotlib.use('Qt5Agg')

import os

import sys
sys.path.append("..")

from collections import OrderedDict
from pyM2FS.pyM2FS_funcs import digest_filenumbers
from pyM2FS.create_merged_target_list import make_mtl

import configparser
from quickreduce import read_io_config, read_obs_config


overwrite_mtls = True


def main(maskname=None,do_overwrite=False, pipe_config_name = '../configs/pipeline.ini'):
    obs_config_name = '../configs/obs_{}.ini'.format(maskname)
    io_config_name = '../configs/io_{}.ini'.format(maskname)

    if '/configs' not in pipe_config_name and not os.path.exists(pipe_config_name):
        pipe_config_name = './configs/{}'.format(pipe_config_name)

    pipe_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    pipe_config.read(pipe_config_name)

    pipe_config['GENERAL']['mask_name'] = maskname


    ## Check that the configs are there or defined in the pipeline conf file
    ## read in the confs if they are there, otherwise return none
    obs_config = read_obs_config(obs_config_name, pipe_config['CONFS'], maskname)
    io_config = read_io_config(io_config_name, pipe_config, maskname)

    print("performing mask: {}  obs: {}  io: {}".format(maskname,obs_config_name,io_config_name))

    mtl_path = os.path.join(io_config['PATHS']['catalog_loc'],io_config['DIRS']['mtl'])
    mtl_name = io_config['SPECIALFILES']['mtl']+"_full.csv"

    if not os.path.exists(os.path.join(mtl_path,mtl_name)) or do_overwrite:
        ## Interpret the filenumbers specified in the configuration files
        str_filenumbers = OrderedDict(obs_config['FILENUMBERS'])
        filenumbers = digest_filenumbers(str_filenumbers)

        make_mtl(io_config,filenumbers['science'][0],vizier_catalogs=['sdss12'], \
                   overwrite_field=False, overwrite_redshifts = True)





if __name__ == '__main__':
    masks = []
    for fil in os.listdir('../configs/'):
        if 'io' == fil[:2] and 'A267' not in fil and 'A00' not in fil\
                and 'B01' not in fil:
            masks.append((fil.split('_')[1]).split('.')[0])
    for mask in masks:
        main(mask,do_overwrite=overwrite_mtls)