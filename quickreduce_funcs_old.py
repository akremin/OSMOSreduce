import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

# Non-standard dependencies
# import PyCosmic


#
# def get_all_filedata(filenum_dict, info=None,
#                      cut_bias_cols = False, convert_adu_to_e = False):
#     if info is None or 'tags' not in info.keys():
#         tags = ''
#     else:
#         tags = info['tags']
#     if info is None or 'datadir' not in info.keys():
#         datadir = './'
#     else:
#         tags = info['datadir']
#     if info is None or 'tags' not in info.keys():
#         tags = ''
#     else:
#         tags = info['tags']
#     if info is None or 'tags' not in info.keys():
#         tags = ''
#     else:
#         tags = info['tags']
#     if info is None or 'tags' not in info.keys():
#         tags = ''
#     else:
#         tags = info['tags']
#    template, maskname, cameras = ['r', 'b'], opamps = [1, 2, 3, 4],



def get_all_filedata(filenum_dict, datadir, template, master_template, maskname,cameras=['r' ,'b'], opamps=[1 ,2 ,3 ,4], tags = '',
                     cut_bias_cols = False, convert_adu_to_e = False, master_types = [], **kwargs):
    all_data, all_headers = {}, {}
    skip_opamp_assignment = False
    if opamps is None:
        skip_opamp_assignment = True
        opamps = [1]
    for imtype,filenums in filenum_dict.items():
        all_cam_dat = {}
        all_cam_heads = {}
        for camera in cameras:
            all_dat = {}
            all_heads = {}
            for opamp in opamps:
                curamp_dats = {}
                curamp_heads = {}
                for filnum in filenums:
                    filnm = template.format(cam=camera,imtype=imtype,filenum=filnum,opamp=opamp,maskname=maskname,tags=tags)
                    cur_dat = fits.open(os.path.join(datadir ,filnm))[0]
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
                    curamp_dats[filnum] = cur_dat_data
                    curamp_heads[filnum] = cur_dat_header

                if skip_opamp_assignment:
                    all_dat = curamp_dats.copy()
                    all_heads = curamp_heads.copy()
                else:
                    all_dat[opamp] = curamp_dats.copy()
                    all_heads[opamp] = curamp_heads.copy()

            all_cam_dat[camera] = all_dat.copy()
            all_cam_heads[camera] = all_heads.copy()

        all_data[imtype] = all_cam_dat.copy()
        all_headers[imtype] = all_cam_heads.copy()

    for imtype in master_types:
        all_cam_dat = {}
        all_cam_heads = {}
        for camera in cameras:
            filnm = master_template.format(cam=camera, imtype=imtype, maskname=maskname,tags=tags)
            cur_dat = fits.open(os.path.join(datadir, filnm))[0]
            all_cam_heads[camera] = cur_dat.header
            all_cam_dat[camera] = cur_dat.data
        all_data[imtype] = all_cam_dat.copy()
        all_headers[imtype] = all_cam_heads.copy()

    return all_data, all_headers




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

def print_data_neatly(data):
    for key1,val1 in data.items():
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
                            if type(val4) is np.ndarray:
                                print("\t\t\t\tdatashape = ({},{})".format(*val4.shape))
                    else:
                        print("\t\t\tdatatype = {}".format(type(val3)))
                        if type(val3) is np.ndarray:
                            print("\t\t\tdatashape = ({},{})".format(*val3.shape))

def stitch_these_camera_data(data_dict,header_dict,filenumbers,camera,imtype,common_func_vals,save_each=False):
    xorients = {-1: 'l', 1: 'r'}
    yorients = {-1: 'b', 1: 'u'}
    cam_data_dict = data_dict[camera]
    cam_header_dict = header_dict[camera]

    cam_stitched_data = {}
    cam_stitched_headers = {}
    for filenum in filenumbers[imtype]:
        img = {}
        for opamp in cam_data_dict.keys():
            header = cam_header_dict[opamp][filenum]
            xsign = np.sign(header['CHOFFX'])
            ysign = np.sign(header['CHOFFY'])
            location = yorients[ysign] + xorients[xsign]
            print("Imtype: {}  In filenum: {} Camera: {} Opamp: {} located at {}".format(imtype, filenum, camera, opamp,
                                                                                         location))
            img[location] = cam_data_dict[opamp][filenum]

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
        header = cam_header_dict[opamp][filenum].copy()
        header['NAXIS1'] = int(y_bl + y_ul)
        header['NAXIS2'] = int(x_bl + x_br)
        header['DATASEC'] = '[1:{},1:{}]'.format(header['NAXIS1'],header['NAXIS2'])
        header['TRIMSEC'] = header['DATASEC']
        header['CHOFFX'] = 0
        header['CHOFFY'] = 0
        header['FILENAME'] = header['FILENAME'].split('c{}'.format(opamp))[0]
        for key in ['OPAMP']:#, 'CHOFFX', 'CHOFFY']:
            header.remove(key)
        header.add_history("Stitched 4 opamps by quickreduce on {}".format(common_func_vals['date']))

        ## Save data and header to working memory
        cam_stitched_data[filenum] = merged
        cam_stitched_headers[filenum] = header

        if save_each:
            ## Save data and header to fits file
            outhdu = fits.PrimaryHDU(data=merged, header=header)
            filename = common_func_vals['template'].format(imtype=imtype, cam=camera, filenum=filenum,
                                                           maskname=common_func_vals['maskname'], tags=common_func_vals['tags'])
            filename = os.path.join(common_func_vals['datadir'], filename)
            outhdu.writeto(filename, overwrite=True)

            ## Plot the image
            plt.figure()
            plot_array = outhdu.data - np.min(outhdu.data) + 1e-4
            plt.imshow(np.log(plot_array), 'gray', origin='lowerleft')
            plt.savefig(filename.replace('.fits', '.png'), dpi=1200)
            plt.show()

    return cam_stitched_data,cam_stitched_headers


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

def scrub_header(header,array_shape):
    header.remove('NOVERSCN')
    header.remove('NBIASLNS')
    header.remove('BIASSEC')
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