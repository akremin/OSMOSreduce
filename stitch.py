import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt


def stitch_all_images(all_hdus,date):
    stitched_hdu_dict = {}
    hdu_opamp_dict = {}
    for (camera, filenum, imtype, opamp),hdu in all_hdus.items():
        if (camera, filenum, imtype) not in hdu_opamp_dict.keys():
            hdu_opamp_dict[(camera, filenum, imtype)] = {}
        hdu_opamp_dict[(camera, filenum, imtype)][opamp] = hdu

    for (camera, filenum, imtype),opampdict in hdu_opamp_dict.items():
        outhdu = stitch_these_camera_data(opampdict , date)
        stitched_hdu_dict[(camera, filenum, imtype, None)] = outhdu

    return stitched_hdu_dict


def stitch_these_camera_data(hdudict,date):
    xorients = {-1: 'l', 1: 'r'}
    yorients = {-1: 'b', 1: 'u'}

    img = {}
    for opamp,hdu in hdudict.items():
        header = hdu.header
        xsign = np.sign(header['CHOFFX'])
        ysign = np.sign(header['CHOFFY'])
        location = yorients[ysign] + xorients[xsign]
        #print("Imtype: {}  In filenum: {} Camera: {} Opamp: {} located at {}".format(imtype, filenum, camera, opamp,
        #                                                                            location))
        img[location] = hdu.data

    trans = {}
    ## Transform opamps to the correct directions
    trans['bl'] = img['bl']
    trans['br'] = np.fliplr(img['br'])
    trans['ul'] = np.flipud(img['ul'])
    trans['ur'] = np.fliplr(np.flipud(img['ur']))

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

    ## This returns the image with wavelength increasing to the left
    ## For simplicity, flip image so wavelength increases with pixel number
    merged = np.fliplr(merged)

    ## Update header
    opamp = 1
    header = hdudict[opamp].header.copy()
    header['NAXIS1'] = int(y_bl + y_ul)
    header['NAXIS2'] = int(x_bl + x_br)
    header['DATASEC'] = '[1:{},1:{}]'.format(header['NAXIS1'],header['NAXIS2'])
    header['TRIMSEC'] = header['DATASEC']
    header['CHOFFX'] = 0
    header['CHOFFY'] = 0
    header['FILENAME'] = header['FILENAME'].split('c{}'.format(opamp))[0]

    for key in ['OPAMP']:#, 'CHOFFX', 'CHOFFY']:
        header.remove(key)

    header.add_history("Stitched 4 opamps by quickreduce on {}".format(date))

    plt.figure(); plt.imshow(merged,origin='lower'); plt.show()
    ## Save data and header to working memory
    outhdu = fits.PrimaryHDU(data=merged, header=header)

    return outhdu