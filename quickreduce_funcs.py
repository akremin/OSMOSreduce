import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Non-standard dependencies
# import PyCosmic




def format_plot(ax, title=None, xlabel=None, ylabel=None, labelsize=16, titlesize=None, ticksize=None, legendsize=None,
                legendloc=None):
    if titlesize is None:
        titlesize = labelsize + 2
    if ticksize is None:
        ticksize = labelsize - 2
    if legendsize is None:
        legendsize = labelsize - 4
    if title is not None:
        plt.title(title, fontsize=titlesize)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=labelsize)
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=labelsize)
    if legendloc is not None:
        plt.legend(loc=legendloc, fontsize=legendsize)

    plt.setp(ax.get_xticklabels(), fontsize=ticksize)
    plt.setp(ax.get_yticklabels(), fontsize=ticksize)


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


def remove_bias_lines(cur_dat,cut_bias_cols=False,convert_adu_to_e=False):
    cur_dat_header = cur_dat.header
    cur_dat_data = cur_dat.data
    if cut_bias_cols:
        datasec = cur_dat.header['DATASEC'].strip('[]')
        (x1, x2), (y1, y2) = [[int(x) - 1 for x in va.split(':')] for va in datasec.split(',')]
        cur_dat_data = cur_dat.data[y1:y2 + 1, x1:x2 + 1]
        cur_dat_header = scrub_header(cur_dat_header, cur_dat_data.shape)

    if convert_adu_to_e:
        cur_dat_data = (cur_dat_data * cur_dat_header['EGAIN'])  # - cur_dat_header['ENOISE']
        cur_dat_header['EGAIN'] = 1.0
        # cur_dat_header['ENOISE'] = 0.0

    outhdu = fits.PrimaryHDU(data=cur_dat_data, header=cur_dat_header)
    return outhdu

def scrub_header(header,array_shape):
    header.remove('NOVERSCN')
    header.remove('NBIASLNS')
    header.remove('BIASSEC')
    header.remove('TRIMSEC')
    header.remove('DATASEC')
    header['NAXIS1'] = int(array_shape[0])
    header['NAXIS2'] = int(array_shape[1])
    return header

# def remove_cosmic_rays(filelist = [],readnoise='ENOISE',sigmadet=8,crgain='EGAIN',crverbose=True,crreturndata=False):
#     import PyCosmic
#     if crreturndata:
#         outdats_crs = []
#     for i,fil in enumerate(filelist):
#         rootfile = fil.split('.fits')[0]
#         savefile = rootfile+'c.fits'
#         maskfile = rootfile+'.crmask.fits'
#
#         if os.path.exists(savefile):
#             try:    os.remove(savefile)
#             except: pass
#         if os.path.exists(maskfile):
#             try:    os.remove(maskfile)
#             except: pass
#         if crreturndata:
#             outdat,pycosmask,pyheader = PyCosmic.detCos(fil,maskfile,savefile,rdnoise=readnoise,sigma_det=sigmadet,
#                                                        gain=crgain,verbose=crverbose,return_data=crreturndata)
#             outdats_crs.append(outdat)
#         else:
#             PyCosmic.detCos(rootfile, maskfile, savefile, rdnoise=readnoise, sigma_det=sigmadet, gain=crgain,
#                             verbose=crverbose, return_data=crreturndata)
#     if crreturndata:
#         return np.asarray(outdats_crs)
#     else:
#         print("Cosmic ray removal successful")


def plot_calibd()
    import seaborn
    def fifthorder(xs ,a ,b ,c ,d ,e ,f):
        return a+ b * xs + c * xs * xs + d * xs ** 3 + e * xs ** 4 + f * xs ** 5

    from quickreduce_funcs import format_plot

    # In[202]:

    fnum = {'comp': 628, 'thar': 627}
    cam = 'r'

    best_fits = {}
    best_fits['comp'] = calib_coefs['comp'][fnum['comp']][cam]
    best_fits['thar'] = calib_coefs['thar'][fnum['thar']][cam]

    # In[393]:


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

