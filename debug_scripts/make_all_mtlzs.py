import matplotlib
matplotlib.use('Qt5Agg')

import os


from collections import OrderedDict
from quickreduce_funcs import digest_filenumbers
from create_merged_target_list import make_mtl, make_mtlz

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

    mtl_path = os.path.join(io_config['PATHS']['catalog_loc'],io_config['DIRS']['mtl'])
    mtl_name = io_config['SPECIALFILES']['mtlz']+"_full.csv"

    if not os.path.exists(os.path.join(mtl_path,mtl_name)):
        ## Interpret the filenumbers specified in the configuration files
        str_filenumbers = OrderedDict(obs_config['FILENUMBERS'])
        filenumbers = digest_filenumbers(str_filenumbers)

        make_mtl(io_config,filenumbers['science'][0],vizier_catalogs=['sdss12'], \
                   overwrite_field=False, overwrite_redshifts = False)


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