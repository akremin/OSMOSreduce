import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import medfilt,find_peaks
from astropy.table import Table
# Non-standard dependencies
# import PyCosmic




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


def debug_plots(opamparray,camera, opamp, filetype='Bias',typical_sep=100,np_oper=np.median):
    master_bias = np_oper(opamparray,axis=0)
    plt.figure()
    plt.imshow(master_bias, origin='lowerleft')
    plt.colorbar()
    plt.title("Median {} {}{}".format(filetype,camera, opamp))
    plt.show()
    plt.figure(figsize=(16, 16))
    plt.subplot(121)
    subset = opamparray.copy()
    subset[subset <= 0] = 1e-4
    plt.imshow(np.log(np.std(subset, axis=0)), origin='lowerleft')
    plt.colorbar(fraction=0.05)
    plt.title("log(Std) {} {}{}".format(filetype,camera, opamp))
    plt.subplot(122)
    subset[np.abs(subset - typical_sep) > (typical_sep+1)] = np.median(subset)
    plt.imshow(np.std(subset, axis=0), origin='lowerleft')
    plt.colorbar(fraction=0.05)
    plt.title("Cut Std {} {}{}".format(filetype,camera, opamp))
    plt.show()

def plot_cr_images(pycosmask,outdat,maskfile,filename):
    ## Plot the image
    plt.figure()
    pycosmask = pycosmask - np.min(pycosmask) + 1e-4
    plt.imshow(np.log(pycosmask),'gray',origin='lowerleft')
    plt.savefig(maskfile.replace('.fits','.png'),dpi=1200)
    plt.close()
    ## Plot the image
    plt.figure()
    pycos = outdat - np.min(outdat) + 1e-4
    plt.imshow(np.log(pycos),'gray',origin='lowerleft')
    plt.savefig(filename.replace('.fits','.png'),dpi=1200)
    plt.close()


def plot_calibd(calib_coefs,dict_of_hdus):
    import seaborn
    def fifthorder(xs ,a ,b ,c ,d ,e ,f):
        return a+ b * xs + c * xs * xs + d * xs ** 3 + e * xs ** 4 + f * xs ** 5

    from quickreduce_funcs import format_plot

    fnum = {'comp': 628, 'thar': 627}
    cam = 'r'

    best_fits = {}
    best_fits['comp'] = calib_coefs['comp'][fnum['comp']][cam]
    best_fits['thar'] = calib_coefs['thar'][fnum['thar']][cam]

    for calib_cof in ['comp', 'thar']:
        for ap in best_fits[calib_cof].colnames:
            for calib_spec in ['comp', 'thar']:
                # calib_spec = calib_cof
                plt.figure()
                fig, ax = plt.subplots(1, figsize=(12, 8))
                flux = dict_of_hdus[calib_spec][cam][fnum[calib_spec]].data[ap]
                pixels = np.arange(flux.size).astype(np.float64)
                best_wave = fifthorder(pixels, *best_fits[calib_cof][ap].data)
                fline, = plt.plot(best_wave, flux / flux.max(), 'b')
                format_plot(ax,
                            title='Pixel to Vacuum Wavelegth fit coefs:{}, spec:{}, {}'.format(calib_cof, calib_spec, ap),
                            xlabel=r'${\lambda}_{vac}\quad\mathrm{[\AA]}$', ylabel='Normalized Intensity')


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



def plot_skies(axi,minlam,maxlam):
    skys = Table.read(r'C:\Users\kremin\Github\M2FSreduce\lamp_linelists\gident_UVES_skylines.csv',format='ascii.csv')
    sky_lines = skys['WAVE_AIR']
    sky_fluxes = skys['FLUX']
    shortlist_skylines = ((sky_lines>minlam)&(sky_lines<maxlam))
    shortlist_fluxes = (sky_fluxes > 0.2)
    shortlist_bool = (shortlist_fluxes & shortlist_skylines)

    select_table = skys[shortlist_bool]
    from calibration_funcs import air_to_vacuum
    shortlist_skylines = air_to_vacuum(select_table['WAVE_AIR'])
    fluxes = select_table['FLUX']
    log_flux = np.log(fluxes-np.min(fluxes)+1.01)
    max_log_flux = np.max(log_flux)
    for vlin,flux in zip(shortlist_skylines,log_flux):
        axi.axvline(vlin, ls='-.', alpha=0.2, c='black',lw=0.4+4*flux/max_log_flux)

def smooth_and_dering(outskyflux):
    outskyflux[np.isnan(outskyflux)] = 0.
    smthd_outflux = medfilt(outskyflux, 3)
    peak_inds, peak_props = find_peaks(outskyflux, height=(1500, 100000), width=(0, 20))
    heights = peak_props['peak_heights']
    ringing_factor = 1 + (heights // 1000)
    ring_lefts = (peak_inds - ringing_factor * (peak_inds - peak_props['left_bases'])).astype(int)
    peak_lefts = (peak_props['left_bases']).astype(int)
    ring_rights = (peak_inds + ringing_factor * (peak_props['right_bases'] - peak_inds)).astype(int)
    peak_rights = (peak_props['right_bases']).astype(int)

    corrected = smthd_outflux.copy()
    for rleft, pleft in zip(ring_lefts, peak_lefts):
        corrected[rleft:pleft] = smthd_outflux[rleft:pleft]
    for rright, pright in zip(ring_rights, peak_rights):
        corrected[pright:rright] = smthd_outflux[pright:rright]
    return corrected
