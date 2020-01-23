import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.signal import medfilt, find_peaks
import os
from collections import OrderedDict
import configparser


def make_mtlz_wrapper(data,filemanager,io_config,step,do_step_bool,pipe_options,cams):
    imtype = 'zfits'
    if not do_step_bool:
        step == 'zfit'
        data.proceed_to(step)
        write_dir = filemanager.directory.current_write_dir
        write_template = filemanager.current_write_template

        if os.path.exists(os.path.join(write_dir,write_template.format(imtype=imtype,cam=cams[0]))):
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
    from pyM2FS.create_merged_target_list import make_mtlz
    make_mtlz(data.mtl, hdus, find_extra_redshifts,outfile=outfile,\
              vizier_catalogs = ['sdss12'])

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


def generate_wave_grid(header):
    wavemin, wavemax = header['wavemin'], header['wavemax']
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
        outwaves = np.logspace(wavemin, wavemax, num=nwaves, base=logbase)
    else:
        wavestep = header['wavestep']
        outwaves = np.arange(wavemin, wavemax + wavestep, wavestep)
    return outwaves


def format_plot(ax, title=None, xlabel=None, ylabel=None, labelsize=16, titlesize=None, ticksize=None, legendsize=None,
                legendloc=None):
    if titlesize is None:
        titlesize = labelsize + 2
    if legendsize is None:
        legendsize = labelsize - 2
    if title is not None:
        ax.set_title(title, fontsize=titlesize)
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=labelsize)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=labelsize)
    if legendloc is not None:
        ax.legend(loc=legendloc, fontsize=legendsize)

    if ticksize is not None:
        ax.set_xticklabels(labels=ax.get_xticklabels(), fontsize=ticksize)
        ax.set_yticklabels(labels=ax.get_yticklabels(), fontsize=ticksize)


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