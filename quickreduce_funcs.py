import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Non-standard dependencies
# import PyCosmic


def pair_exposures(allhdus,cams_same=True,max_matches=1):
    exp_types = list(allhdus.keys())
    if 'science' not in exp_types:
        return None
    exp_types.remove('science')
    science_dict = allhdus['science']
    cameras = list(science_dict.keys())
    all_cams = cameras.copy()
    if cams_same and len(all_cams) > 1:
        cameras = [cameras[0]]

    exposure_pairs = {}
    for cam in cameras:
        sci_filenums = list(science_dict[cam].keys())
        exposure_pairs[cam] = {}
        for filnum in sci_filenums:
            exposure_pairs[cam][filnum] = {}
            for exptype in exp_types:
                these_filnums = np.asarray(list(allhdus[exptype][cam].keys()))
                closest_filnum_inds = np.argsort(np.abs(these_filnums-filnum))
                if len(closest_filnum_inds)>max_matches:
                    closest_filnum_inds = closest_filnum_inds[:max_matches]
                exposure_pairs[cam][filnum][exptype] = these_filnums[closest_filnum_inds]
    if cams_same and len(all_cams) > 1:
        for i in range(1,len(all_cams)):
            exposure_pairs[all_cams[i]] = exposure_pairs[all_cams[0]]
    return exposure_pairs


def get_all_filedata(filenum_dict, datadir, template, master_template, maskname,cameras=['r' ,'b'], opamps=[1 ,2 ,3 ,4], tags = '',
                     cut_bias_cols = False, convert_adu_to_e = False, fibersplit = False, master_types = [], **kwargs):
    all_hdus = {}
    skip_opamp_assignment = False
    if opamps is None:
        skip_opamp_assignment = True
        opamps = [1]
    for imtype,filenums in filenum_dict.items():
        all_cam_hdus = {}
        for camera in cameras:
            all_opamp_hdus = {}
            for opamp in opamps:
                curamp_hdus = {}
                for filnum in filenums:
                    filnm = template.format(cam=camera,imtype=imtype,filenum=filnum,opamp=opamp,maskname=maskname,tags=tags)
                    if fibersplit:
                        cur_dat = fits.open(os.path.join(datadir, filnm), mode='readonly')[1]
                    else:
                        cur_dat = fits.open(os.path.join(datadir ,filnm),mode='readonly')[0]
                    if cut_bias_cols or convert_adu_to_e:
                        cur_dat_header = cur_dat.header
                        if cut_bias_cols:
                            datasec = cur_dat.header['DATASEC'].strip('[]')
                            (x1, x2), (y1, y2) = [[int(x)-1 for x in va.split(':')] for va in datasec.split(',')]
                            cur_dat_data = cur_dat.data[y1:y2+1,x1:x2+1]
                            cur_dat_header = scrub_header(cur_dat_header, cur_dat_data.shape)
                        else:
                            cur_dat_data = cur_dat.data
                        if convert_adu_to_e:
                            cur_dat_data = (cur_dat_data * cur_dat_header['EGAIN']) #- cur_dat_header['ENOISE']
                            cur_dat_header['EGAIN'] = 1.0
                            #cur_dat_header['ENOISE'] = 0.0
                        outhdu = fits.PrimaryHDU(data=cur_dat_data, header=cur_dat_header)
                    else:
                        outhdu = cur_dat
                    curamp_hdus[filnum] = outhdu

                if skip_opamp_assignment:
                    all_opamp_hdus = curamp_hdus.copy()
                else:
                    all_opamp_hdus[opamp] = curamp_hdus.copy()

            all_cam_hdus[camera] = all_opamp_hdus.copy()

        all_hdus[imtype] = all_cam_hdus

    for imtype in master_types:
        all_cam_hdus = {}
        for camera in cameras:
            filnm = master_template.format(cam=camera, imtype=imtype, maskname=maskname,tags=tags)
            cur_dat = fits.open(os.path.join(datadir, filnm))[0]
            outhdu = fits.PrimaryHDU(data=cur_dat.data, header=cur_dat.header)
            all_cam_hdus[camera] = outhdu
            all_hdus[imtype] = all_cam_hdus.copy()

    return all_hdus


def get_all_fibersplitdata(filenums, datadir, template, maskname,cameras=['r' ,'b'], opamps=[1 ,2 ,3 ,4], tags = '',
                     imtype=''):
    all_cam_dat = {}
    all_cam_heads = {}
    casettes = np.arange(8)+1
    fibs = np.arange(16)+1
    for camera in cameras:
        all_dat = {}
        all_heads = {}
        for opamp in opamps:
            curamp_dats = []
            curamp_heads = []
            for filnum in filenums:
                for casette in casettes:
                    for fib in fibs:
                        fiber = 'FIBER{}{:02d}'.format(casette,fib)
                        filnm = template.format(cam=camera,imtype=imtype,filenum=filnum,fibname=fiber,maskname=maskname,tags=tags)
                        cur_dat = fits.open(os.path.join(datadir ,cameradir[camera] ,filnm))[0]
                        cur_dat_data = cur_dat.data
                        cur_dat_header = cur_dat.header
                        cur_dat_data = (cur_dat_data*cur_dat_header['EGAIN']) - cur_dat_header['ENOISE']
                        curamp_dats.append(cur_dat_data)
                        curamp_heads.append(cur_dat_header)
            # master_dat = np.zeros(shape=(cur_dat.shape[0],cur_dat.shape[1],len(all_dats)))
            curamp_dats = np.asarray(curamp_dats)
            all_dat[opamp] = curamp_dats.copy()
            all_heads[opamp] = curamp_heads.copy()

        all_cam_dat[camera] = all_dat.copy()
        all_cam_heads[camera] = all_heads.copy()
        print_data_neatly(all_cam_dat)
    return all_cam_dat, all_cam_heads


def print_data_neatly(hdudict):
    for key1,val1 in hdudict.items():
        print(key1)
        for key2,val2 in val1.items():
            print("\t{}".format(key2))
            if type(val2) is dict:
                for key3,val3 in val2.items():
                    print("\t\t{}".format(key3))
                    if type(val3) is dict:
                        for key4,val4 in val3.items():
                            print("\t\t\t{}".format(key4))
                            print("\t\t\t\tdatatype = {}".format(type(val4)))
                            if type(val4) is fits.PrimaryHDU:
                                print("\t\t\t\tdatashape = ({},{})".format(*val4.data.shape))
                    else:
                        print("\t\t\tdatatype = {}".format(type(val3)))
                        if type(val3) is fits.PrimaryHDU:
                            print("\t\t\tdatashape = ({},{})".format(*val3.data.shape))


def get_dict_temp(dict_of_hdus):
    blank_dictofdicts = {}
    for imtype,camdict in dict_of_hdus.items():
        blank_dictofdicts[imtype] = {}
        for camera,filenumdict in camdict.items():
            blank_dictofdicts[imtype][camera] = {}
    return blank_dictofdicts


def stitch_these_camera_data(hdudict,filenum,camera,imtype,mask_name,save_info,save_each=False,make_plot=False):
    xorients = {-1: 'l', 1: 'r'}
    yorients = {-1: 'b', 1: 'u'}

    img = {}
    for opamp,hdu in hdudict.keys():
        header = hdu.header
        xsign = np.sign(header['CHOFFX'])
        ysign = np.sign(header['CHOFFY'])
        location = yorients[ysign] + xorients[xsign]
        print("Imtype: {}  In filenum: {} Camera: {} Opamp: {} located at {}".format(imtype, filenum, camera, opamp,
                                                                                     location))
        img[location] = hdu.data

    trans = {}
    ## Transform opamps to the correct directions
    trans['bl'] = img['bl']
    trans['br'] = np.fliplr(img['br'])
    trans['ul'] = np.flipud(img['ul'])
    trans['ur'] = np.fliplr(np.flipud(img['ur']))
    del img

    y_bl, x_bl = trans['bl'].shape
    y_ul, x_ul = trans['ul'].shape
    y_br, x_br = trans['br'].shape
    y_ur, x_ur = trans['ur'].shape

    ## Make sure the shapes work
    if y_bl != y_br or y_ul != y_ur:
        print("yr and yl not the same")
    if x_bl != x_ul or x_br != x_ur:
        print("xb and xu not the same")

    ## Create the full-sized image array
    merged = np.ndarray(shape=(y_bl + y_ul, x_bl + x_br))

    ## Assign the opamps to the correct region of the array
    ## bl
    merged[:y_bl, :x_bl] = trans['bl']
    ## ul
    merged[y_bl:, :x_bl] = trans['ul']
    ## br
    merged[:y_bl, x_bl:] = trans['br']
    ## ur
    merged[y_bl:, x_bl:] = trans['ur']

    ## Update header
    opamp = 1
    header = hdu[opamp].header.copy()
    header['NAXIS1'] = int(y_bl + y_ul)
    header['NAXIS2'] = int(x_bl + x_br)
    header['DATASEC'] = '[1:{},1:{}]'.format(header['NAXIS1'],header['NAXIS2'])
    header['TRIMSEC'] = header['DATASEC']
    header['CHOFFX'] = 0
    header['CHOFFY'] = 0
    header['FILENAME'] = header['FILENAME'].split('c{}'.format(opamp))[0]

    for key in ['OPAMP']:#, 'CHOFFX', 'CHOFFY']:
        header.remove(key)

    header.add_history("Stitched 4 opamps by quickreduce on {}".format(save_info['date']))

    ## Save data and header to working memory
    outhdu = fits.PrimaryHDU(data=merged, header=header)

    if save_each:
        ## Save data and header to fits file
        save_hdu(outhdu, save_info, camera, imtype, mask_name, filenum)


    if make_plot:
        ## Plot the image
        plt.figure()
        plot_array = outhdu.data - np.min(outhdu.data) + 1e-4
        plt.imshow(np.log(plot_array), 'gray', origin='lowerleft')
        plt.savefig(filename.replace('.fits', '.png'), dpi=1200)
        plt.show()
    return outhdu


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


def scrub_header(header,array_shape):
    header.remove('NOVERSCN')
    header.remove('NBIASLNS')
    header.remove('BIASSEC')
    header.remove('TRIMSEC')
    header.remove('DATASEC')
    header['NAXIS1'] = int(array_shape[0])
    header['NAXIS2'] = int(array_shape[1])
    return header

def remove_cosmic_rays(filelist = [],readnoise='ENOISE',sigmadet=8,crgain='EGAIN',crverbose=True,crreturndata=False):
    import PyCosmic
    if crreturndata:
        outdats_crs = []
    for i,fil in enumerate(filelist):
        rootfile = fil.split('.fits')[0]
        savefile = rootfile+'c.fits'
        maskfile = rootfile+'.crmask.fits'

        if os.path.exists(savefile):
            try:    os.remove(savefile)
            except: pass
        if os.path.exists(maskfile):
            try:    os.remove(maskfile)
            except: pass
        if crreturndata:
            outdat,pycosmask,pyheader = PyCosmic.detCos(fil,maskfile,savefile,rdnoise=readnoise,sigma_det=sigmadet,
                                                       gain=crgain,verbose=crverbose,return_data=crreturndata)
            outdats_crs.append(outdat)
        else:
            PyCosmic.detCos(rootfile, maskfile, savefile, rdnoise=readnoise, sigma_det=sigmadet, gain=crgain,
                            verbose=crverbose, return_data=crreturndata)
    if crreturndata:
        return np.asarray(outdats_crs)
    else:
        print("Cosmic ray removal successful")


def save_hdu(outhdu, save_info, camera='r', imtype='comp', mask_name='A02', filenum=999):
    outname = save_info['template'].format(cam=camera, imtype=imtype, maskname=mask_name,
                                           filenum=filenum, tags=save_info['tags'])
    filename = os.path.join(save_info['datadir'], outname)
    outhdu.writeto(filename, overwrite=True)
