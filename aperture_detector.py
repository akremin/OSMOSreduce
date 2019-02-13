import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
import os
from scipy.signal import medfilt
from astropy.table import Table

def cutout_all_apperatures(all_hdus,cameras,deadfibers=[],summation_preference='simple'):
    apcut_hdus = {}
    cam_apperatures = {}
    for camera in cameras:
        bad_cam_fibers = []
        for fib in deadfibers:
            if fib[0].lower() == camera.lower():
                bad_cam_fibers.append(fib)
        #fib_hdu = all_hdus.pop((camera,'master','fibermap',None))
        fib_hdu = all_hdus[(camera, 'master', 'fibermap', None)]
        fiber_imag = fib_hdu.data
        fiber_imag = fiber_imag - np.min(fiber_imag) + 1e-4
        apperatures = find_apperatures(fiber_imag,bad_cam_fibers,cam=camera)
        cam_apperatures[camera] = apperatures

    for (camera, filenum, imtype, opamp),hdu in all_hdus.items():
        oneds = cutout_oned_apperatures(hdu.data ,cam_apperatures[camera],summation_preference)
        outhead = hdu.header.copy(strip=True)
        outhead.remove('datasec' ,ignore_missing=True)
        outhead.remove('trimsec' ,ignore_missing=True)
        outhead.remove('CHOFFX' ,ignore_missing=True)
        outhead.remove('CHOFFY' ,ignore_missing=True)
        outhead.remove('NOPAMPS' ,ignore_missing=True)

        outhdu = fits.BinTableHDU(data=Table(data=oneds) ,header=outhead, name='flux')
        apcut_hdus[(camera, filenum, imtype, opamp)] = outhdu
        print("Completed Apperature Cutting of: cam={}, fil={}, type={}".format(camera,filenum,imtype))

    return apcut_hdus

def cutout_oned_apperatures(image,apperatures,summation_preference):
    twod_cutouts = cutout_twod_apperatures(image,apperatures)
    oned_cutouts = {}
    if summation_preference != 'simple':
        print("Only simply equal weight summation of 2d to 1d implemented.\n")
        summation_preference = 'simple'

    if summation_preference == 'simple':
        for key, cutout in twod_cutouts.items():
            oned_cutout = cutout.sum(axis=0)
            oned_cutouts[key] = oned_cutout
    else:
        pass
        ## Not yet implemented

    return oned_cutouts


def cutout_twod_apperatures(image,apperatures):
    cutout_dict = {}
    for tet,tet_dict in apperatures.items():
        start,end = tet_dict['start'],tet_dict['end']
        bottoms = tet_dict['bottoms']
        tops = tet_dict['tops']
        cut_img = image[start:end,:]

        for fib in bottoms.keys():
            list_tops = tops[fib].astype(int)
            list_bottoms = bottoms[fib].astype(int)
            out_array = np.ndarray(shape=(list_tops[0]-list_bottoms[0]+1,len(list_tops)))
            for ii,(bot,top) in enumerate(zip(list_bottoms,list_tops)):
                out_array[:,ii] = cut_img[bot:top+1,ii]
            cutout_dict[fib] = out_array

    return cutout_dict


def find_apperatures(image,badfibs,cam='r',height=5e4,prominence=1e3):
    tetris = np.arange(1,9)
    if cam=='b':
        tetris=tetris[::-1]

    starts,ends = get_all_tetris_edges(image)

    apperatures = {}
    for tet,start,end in zip(tetris,starts,ends):
        apperatures[tet] = {'start':start,'end':end}
        cut_image = image[start:end,:]
        peak_array, left_array, right_array, deviations = find_peak_inds(cut_image, height, prominence)

        bad_nums = []
        for fib in badfibs:
            if int(fib[1]) == int(tet):
                bad_nums.append(int(fib[2:]))

        bad_nums = np.sort(bad_nums)[::-1]
        fiber_nums = list(np.arange(1, 17))

        for bad_num in bad_nums:
            fiber_nums.pop(bad_num - 1)

        peaks,devs,bots,tops = {},{},{},{}
        for row in np.arange(peak_array.shape[0]):
            fibnum = fiber_nums[row]
            fibername = '{}{}{:02d}'.format(cam,tet,fibnum)
            peaks[fibername] = peak_array[row, :]
            devs[fibername] = deviations[row]
            tops[fibername] =  peak_array[row, :] + deviations[row]
            bots[fibername] =  peak_array[row, :] - deviations[row]

        plot_single_tet_apperatures(cut_image, peaks,bots,tops, tet)
        plt.show()
        apperatures[tet]['peaks'] = peaks
        apperatures[tet]['bottoms'] = bots
        apperatures[tet]['tops'] = tops
        apperatures[tet]['deviations'] = devs
    return apperatures


def get_all_tetris_edges(image):
    sumd = image.sum(axis=1)
    normd_sum = sumd / np.max(sumd)
    fig,(ax1,ax2) = plt.subplots(1, 2, figsize=(16, 8))
    ax1.plot(np.arange(len(normd_sum)), normd_sum,'b-')

    medfiltd = medfilt(normd_sum, 11)
    binary = (medfiltd > 0.04).astype(int)
    ax2.plot(np.arange(len(binary)), binary,'g-')

    grad = np.gradient(binary)
    starts = np.where(grad > 0)[0]
    ends = np.where(grad < 0)[0]
    starts = starts[::2] - 4
    ends = ends[::2] + 4

    widths = ends - starts
    med_width = np.median(widths)
    width = min(np.max(widths), med_width + 12)

    changes = widths - width - 10  # 10 is for padding
    start_changes = np.floor(changes / 2).astype(int)
    end_changes = np.ceil(changes / 2).astype(int)

    starts += start_changes
    ends -= end_changes

    plt.subplots(1, 1, figsize=(16, 16))
    plt.imshow(image, origin='lower-left')
    plt.hlines(starts, xmin=0, xmax=image.shape[0], color='r')
    plt.hlines(ends, xmin=0, xmax=image.shape[0], color='r')
    plt.yticks([]), plt.xticks([])
    plt.show()
    return starts, ends




def find_peak_inds(image,height,prominence):
    binwidth = 2
    pixels = np.arange(binwidth,image.shape[1]-binwidth)
    row_lens = []
    inds = []
    lefts,rights = [],[]
    for pix in pixels:
        subset = image[:,pix-binwidth:pix+binwidth+1]
        oned = subset.sum(axis=1)
        peak_inds, outdict = find_peaks(oned,height=height,prominence=prominence,width=(1,30))
        lefts.append(np.floor(outdict['left_ips']).astype(int))
        rights.append(np.ceil(outdict['right_ips']).astype(int))
        inds.append(peak_inds)
        row_lens.append(len(peak_inds))

    nrows = int(np.median(row_lens))

    def get_peak_array(inds,nrows,pixels,row_lens,ncols,binwidth):
        peak_array = np.ndarray(shape=(nrows,ncols))
        for ii,(pixel, rowl) in enumerate(zip(pixels,row_lens)):
            if rowl != nrows:
                if pixel == 0:
                    first_good_ind = np.where(np.array(row_lens)==nrows)[0][0]
                    peak_array[:,pixel] = inds[first_good_ind]
                else:
                    peak_array[:,pixel] = peak_array[:,pixel - 1]
            else:
                peak_array[:, pixel] = inds[ii]

        for pix in np.arange(binwidth):
            peak_array[:,pix] = peak_array[:,binwidth]
            peak_array[:, -1*(pix+1)] = peak_array[:, -1*(binwidth+1)]
        return peak_array

    peak_array = get_peak_array(inds,nrows, pixels, row_lens, image.shape[1], binwidth)
    left_array = get_peak_array(lefts,nrows, pixels, row_lens, image.shape[1], binwidth)
    right_array = get_peak_array(rights, nrows, pixels, row_lens, image.shape[1], binwidth)
    from scipy.signal import medfilt

    deviations = np.zeros(shape=(nrows,))
    for row in np.arange(nrows):
        for itter in range(3):
            peak_array[row,:] = medfilt(peak_array[row,:],71)
            left_array[row, :] = medfilt(left_array[row, :], 71)
            right_array[row, :] = medfilt(right_array[row, :], 71)
        deviations[row] = np.median(np.ceil((right_array[row,:]-left_array[row,:])/2.))+1


    return peak_array,left_array,right_array,deviations


def plot_single_tet_apperatures(image,peaks,bots,tops,tet):
    plt.figure()
    pixels = np.arange(image.shape[1])
    plt.imshow(image, origin='lower-left',cmap='Greys',aspect='auto',interpolation ='nearest')

    for fib in peaks.keys():
        peak_set = peaks[fib]
        bot_set = bots[fib]
        top_set = tops[fib]
        plt.plot(pixels,peak_set ,'b-')
        plt.plot(pixels,top_set,'r-')
        plt.plot(pixels,bot_set,'r-')
        plt.text(200,peak_set[200],fib)
    plt.title(str(tet))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    #plt.show()




if __name__ == '__main__':
    cameras = ['b','r']
    bad_fibs = ['b804']
    filenums = np.arange(573, 578)
    mock_hdu_list = {}

    for camera in cameras:
        for imnum in filenums:
            if imnum == 573:
                image = fits.open(os.path.abspath(
                    os.path.join('..', '..', 'OneDrive - umich.edu', 'Research', 'M2FSReductions', 'A02', 'data_products',
                                 '{}_fibmap_{}_A02_stitched_bc.fits'.format(camera, imnum))))[0].data.astype(np.float64)
            else:
                image += fits.open(os.path.abspath(
                    os.path.join('..', '..', 'OneDrive - umich.edu', 'Research', 'M2FSReductions', 'A02', 'data_products',
                                 '{}_fibmap_{}_A02_stitched_bc.fits'.format(camera, imnum))))[0].data.astype(np.float64)

        # image = np.flipud(image)
        image = image - np.min(image) + 1e-4
        # image = np.log(image-image.min()+1.1)

        mock_hdu_list[(camera, 'master', 'fibermap', None)] = image
    cutout_all_apperatures(mock_hdu_list,cameras,deadfibers=bad_fibs,summation_preference='simple')
