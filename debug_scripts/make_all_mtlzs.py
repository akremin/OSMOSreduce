import matplotlib
matplotlib.use('Qt5Agg')

import os



import sys
sys.path.append("..")

from collections import OrderedDict
from quickreduce_funcs import digest_filenumbers
from create_merged_target_list import make_mtl, make_mtlz
from astropy.table import Table
from astropy.io import fits
import configparser

def main(maskname=None):
    obs_config_name = '../configs/obs_{}.ini'.format(maskname)
    io_config_name = '../configs/io_{}.ini'.format(maskname)

    print("performing mask: {}  obs: {}  io: {}".format(maskname,obs_config_name,io_config_name))
    if '/configs' not in obs_config_name and not os.path.exists(obs_config_name):
        obs_config_name = '../configs/{}'.format(obs_config_name)
    if '/configs' not in io_config_name and not os.path.exists(io_config_name):
        io_config_name = '../configs/{}'.format(io_config_name)

    obs_config = configparser.ConfigParser()
    obs_config.read(obs_config_name)

    io_config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
    io_config.read(io_config_name)

    ## Interpret the filenumbers specified in the configuration files
    str_filenumbers = OrderedDict(obs_config['FILENUMBERS'])
    filenumbers = digest_filenumbers(str_filenumbers)

    mtl_path = os.path.join(io_config['PATHS']['catalog_loc'],io_config['DIRS']['mtl'])
    mtl_name = io_config['SPECIALFILES']['mtl']+"_full.csv"

    if not os.path.exists(os.path.join(mtl_path,mtl_name)):
        make_mtl(io_config,filenumbers['science'][0],vizier_catalogs=['sdss12'], \
                   overwrite_field=False, overwrite_redshifts = False)

    data_path = os.path.abspath(os.path.join(io_config['PATHS']['data_product_loc'], io_config['DIRS']['zfit']))
    dataname = io_config['FILETEMPLATES']['combined'].format(cam='{cam}', imtype = 'zfits')
    dataname = os.path.join(data_path,dataname + io_config['FILETAGS']['skysubd'] + '.fits')

    mtl_table_name = os.path.join(mtl_path,mtl_name)
    if os.path.exists(mtl_table_name):
        mtl_table = Table.read(mtl_table_name,format='ascii.csv')
    else:
        print("There was no mtl!")

    hdus = []
    for cam in ['r','b']:
        if os.path.exists(dataname.format(cam=cam)):
            hdus.append(fits.open(dataname.format(cam=cam))['ZFITS'])

    make_mtlz(mtl_table, hdus, find_more_redshifts = False, outfile = 'mtlz.csv', \
                                                            vizier_catalogs = ['sdss12'])

if __name__ == '__main__':
    masks = []
    for fil in os.listdir('../configs/'):
        if 'io' == fil[:2] and 'A267' not in fil and 'A00' not in fil\
                and 'B01' not in fil:
            masks.append((fil.split('_')[1]).split('.')[0])
    for mask in masks:
        main(mask)