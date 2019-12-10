import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.signal import find_peaks
from scipy.signal import medfilt


def cutout_all_apperatures(all_hdus,cameras,deadfibers=[],summation_preference='simple',\
                            show_plots=True,save_plots=False,save_template='{}{}{}{}{}.png'):
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
        if save_plots:
            defined_save_template = save_template(cam=camera,ap='',imtype='fibmap',step='apcut',comment='{comment}')
        else:
            defined_save_template = save_template
        apperatures = find_apperatures(fiber_imag,bad_cam_fibers,cam=camera,\
                                       show_plots=show_plots,save_plots=save_plots,\
                                       save_template=defined_save_template)
        cam_apperatures[camera] = apperatures

    for (camera, filenum, imtype, opamp),hdu in all_hdus.items():

        oneds = cutout_oned_apperatures(hdu.data,cam_apperatures[camera],summation_preference)
        outhead = hdu.header.copy(strip=True)
        outhead = update_header(outhead,cam_apperatures[camera])

        outhdu = fits.BinTableHDU(data=Table(data=oneds) ,header=outhead, name='flux')
        apcut_hdus[(camera, filenum, imtype, opamp)] = outhdu
        print("Completed Apperature Cutting of: cam={}, fil={}, type={}".format(camera,filenum,imtype))

    return apcut_hdus


def update_header(header,cam_app):
    header.remove('datasec', ignore_missing=True)
    header.remove('trimsec', ignore_missing=True)
    header.remove('CHOFFX', ignore_missing=True)
    header.remove('CHOFFY', ignore_missing=True)
    header.remove('NOPAMPS', ignore_missing=True)

    for tet, tet_dict in cam_app.items():
        start, end = tet_dict['start'], tet_dict['end']
        peaks = tet_dict['peaks']
        for fib in peaks.keys():
            med_peak = np.median(peaks[fib])
            header['yloc_{}'.format(fib[1:])] = start + med_peak
    return header



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


def find_apperatures(image,badfibs,cam='r',height=0.2,prominence=0.1,\
                     show_plots=True,save_plots=False,\
                     save_template='outfile_{}.png'):
    tetris = np.arange(1,9)
    if cam=='b':
        tetris=tetris[::-1]

    full_image_savename = save_template.format(comment="_tet_cuts")
    starts,ends = get_all_tetris_edges(image,show_plots, save_plots, full_image_savename)

    apperatures = {}
    for tet,start,end in zip(tetris,starts,ends):
        print("Detecting apertures for cam={}, tet={}, identified to start at pixel row {} and end at {}".format(cam,tet,start,end))
        apperatures[tet] = {'start':start,'end':end}

        bad_nums = []
        for fib in badfibs:
            if int(fib[1]) == int(tet):
                bad_nums.append(int(fib[2:]))

        bad_nums = np.sort(bad_nums)[::-1]
        fiber_nums = list(np.arange(1, 17))

        nfibs = 16-len(bad_nums)
        cut_image = image[start:end,:]
        peak_array, left_array, right_array, deviations,recvd_nrows = find_peak_inds(cut_image, height, prominence,nfibs)

        # if nfibs != recvd_nrows:
        #     print("Trying to identify the problematic fiber(s)")
        #     devs = []
        #     for column in range(peak_array.shape[1]):
        #         devs.append(peak_array[1:, column] - peak_array[:-1, column])
        #     devs = np.median(devs, axis=0)
        #     nbads = 16 - recvd_nrows - 1
        #     if nfibs < recvd_nrows:
        #         print("It appears that you claimed a dead fiber that was alive, trying to identify")
        #         print(
        #             "Warning, this is untested. It's best that you identify and correctly label the dead fibers and rerun")
        #         for bad_num in bad_nums:
        #             if devs[bad_num - 1 - nbads] > 1.6 * np.median(devs):
        #                 fiber_nums.pop(bad_num - 1)
        #             else:
        #                 print("It appears that {} is actually an okay fiber".format(fiber_nums[bad_num - 1]))
        #     else:
        #         print("It appears that there were more dead fibers, trying to identify")
        #         print(
        #             "Warning, this is untested. It's best that you identify and correctly label the dead fibers and rerun")
        #         ## this bit here is untested
        #         bad_locs = np.where(np.array(devs) > 1.6 * np.median(devs))[0]
        #         if len(bad_locs) == nbads + 1:
        #             alt_bad_nums = bad_locs.astype(int) + 1 + nbads
        #             alt_bad_nums = np.sort(alt_bad_nums)[::-1]
        #             print("Fibers {} were identified. Removing".format(fiber_nums[alt_bad_nums]))
        #             for bad_num in alt_bad_nums:
        #                 fiber_nums.pop(bad_num - 1)
        #
        # else:
        #     for bad_num in bad_nums:
        #         fiber_nums.pop(bad_num - 1)

        for bad_num in bad_nums:
            fiber_nums.pop(bad_num - 1)

        peaks,devs,bots,tops = {},{},{},{}
        if nfibs>peak_array.shape[0]:
            print("There were fewer identified peaks than expected for: cam={} tet={}  start={}, end={}".format(cam,tet,start,end))
        for index,fibnum in enumerate(fiber_nums):
            fibername = '{}{}{:02d}'.format(cam,tet,fibnum)
            peaks[fibername] = peak_array[index, :]
            devs[fibername] = deviations[index]
            tops[fibername] =  peak_array[index, :] + deviations[index]
            bots[fibername] =  peak_array[index, :] - deviations[index]

        if save_plots or show_plots:
            plot_single_tet_apperatures(cut_image, peaks,bots,tops, tet)
        if save_plots:
            plt.savefig(save_template.format(comment='_cutout_'+str(tet)),dpi=600)
        if show_plots:
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.show()
        plt.close()
        apperatures[tet]['peaks'] = peaks
        apperatures[tet]['bottoms'] = bots
        apperatures[tet]['tops'] = tops
        apperatures[tet]['deviations'] = devs
    return apperatures


def get_all_tetris_edges(image, show_plots, save_plots, save_template):
    sumd = image.sum(axis=1)
    gaus_kern = np.exp(-(np.arange(-5,6))*(np.arange(-5,6))/(2*2.5*2.5))#np.array([1, 3, 7, 13,21,31,21, 13, 7, 3, 1])
    sumd = np.convolve(sumd, gaus_kern/np.sum(gaus_kern), mode='same')
    normd_sum = sumd / np.max(sumd)

    medfiltd = medfilt(normd_sum, 11)
    ntetri = 10.
    widths = np.array([10])
    threshold = 0.4
    if threshold > medfiltd.max():
        threshold = medfiltd.max() - 0.01
    while ntetri > 8 or np.any(widths < 100):
        binary = (medfiltd > threshold).astype(int)
        grad = np.gradient(binary)
        starts = np.where(grad > 0)[0]
        ends = np.where(grad < 0)[0]
        starts = starts[::2] - 4
        ends = ends[::2] + 4
        ntetri = len(starts)
        widths = ends-starts
        threshold -= 0.01

    widths = ends - starts
    med_width = np.median(widths)
    width = min(np.max(widths), med_width + 12)

    changes = widths - width - 22  # 10 is for padding
    start_changes = np.floor(changes / 2).astype(int)
    end_changes = np.ceil(changes / 2).astype(int)

    starts += start_changes
    ends -= end_changes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    if save_plots or show_plots:
        ax1.plot(np.arange(len(normd_sum)), normd_sum, 'b-')
        ax2.plot(np.arange(len(binary)), binary, 'g-')
    if save_plots:
        plt.savefig(save_template.replace('tet_cuts','tet_slices'))
    if show_plots:
        plt.show()
    plt.close()

    plt.subplots(1, 1, figsize=(16, 16))
    if save_plots or show_plots:
        plt.imshow(image, origin='lower-left')
        plt.hlines(starts, xmin=0, xmax=image.shape[0], color='r')
        plt.hlines(ends, xmin=0, xmax=image.shape[0], color='r')
        plt.yticks([]), plt.xticks([])
    if save_plots:
        plt.savefig(save_template)
    if show_plots:
        plt.show()
    plt.close()
    return starts, ends




def find_peak_inds(image,height,prominence,nrows):
    binwidth = 2
    pixels = np.arange(binwidth,image.shape[1]-binwidth)
    row_lens = []
    inds = []
    lefts,rights = [],[]
    for pix in pixels:
        subset = image[:,pix-binwidth:pix+binwidth+1]
        oned = subset.sum(axis=1)
        oned = oned/oned.max()
        peak_inds, outdict = find_peaks(oned,height=height,prominence=prominence,width=(1,10))
        lefts.append(np.floor(outdict['left_ips']).astype(int))
        rights.append(np.ceil(outdict['right_ips']).astype(int))
        inds.append(peak_inds)
        row_lens.append(len(peak_inds))

    peak_array,nprows_rec = get_peak_array(inds,nrows, pixels, row_lens, image.shape[1], binwidth)
    left_array,nlrows_rec = get_peak_array(lefts,nrows, pixels, row_lens, image.shape[1], binwidth)
    right_array,nrrows_rec = get_peak_array(rights, nrows, pixels, row_lens, image.shape[1], binwidth)

    nrows = int(np.median([nprows_rec,nlrows_rec,nrrows_rec]))

    deviations = np.zeros(shape=(nrows,))
    for row in np.arange(nrows):
        for itter in range(3):
            peak_array[row,:] = medfilt(peak_array[row,:],71)
            left_array[row, :] = medfilt(left_array[row, :], 71)
            right_array[row, :] = medfilt(right_array[row, :], 71)
        deviations[row] = np.median(np.ceil((right_array[row,:]-left_array[row,:])/2.))+1

    return peak_array,left_array,right_array,deviations,nrows

def get_peak_array(inds,nrows,pixels,row_lens,ncols,binwidth):
    peak_array = np.zeros(shape=(nrows,ncols))
    #full_inds = np.where((np.array(row_lens) == nrows))[0]
    #valid_locations= []
    first_good_ind = None
    for ii in range(len(pixels)):
        if (row_lens[ii] == nrows) and np.all(np.diff(inds[ii])> 4.):
            # valid_locations.append(ii)
            first_good_ind = ii
            break
    if first_good_ind is None:
        print("There were no columns detected with length={}. Maxlen={}, Minlen={}, typical length={}.".format(nrows,np.max(row_lens),np.min(row_lens),np.median(row_lens)),
              "Please check to see if deadfibers need to be added or removed")
        print("Process will likely fail")
        # print("Looking to see if there is an additional fiber")
        # nrows=nrows+1
        # for ii in range(len(pixels)):
        #     if (row_lens[ii] == (nrows+1)) and np.all(np.diff(inds[ii]) > 4.):
        #         # valid_locations.append(ii)
        #         first_good_ind = ii
        #         break
        # if first_good_ind is None:
        #     print("Trying {} fibers didn't work either. Trying one less than original".format(nrows))
        #     nrows = nrows - 2
        #     for ii in range(len(pixels)):
        #         if (row_lens[ii] == nrows) and np.all(np.diff(inds[ii]) > 4.):
        #             # valid_locations.append(ii)
        #             first_good_ind = ii
        #             break
        #     if first_good_ind is None:
        #         print("Trying {} fibers didn't work either. An error is likely to occur".format(nrows))
        #     else:
        #         print("Success using {} fibers!".format(nrows))
        # else:
        #     print("Success using {} fibers!".format(nrows))

    #first_good_ind = int(np.min(valid_locations))
    for ii,(pixel, rowl) in enumerate(zip(pixels,row_lens)):
        correctly_assigned = False
        if rowl == nrows:
            peak_array[:, pixel] = inds[ii]
            correctly_assigned = True
        elif rowl == (nrows+1):
            diffs = np.diff(inds[ii])
            typical_diff = np.median(diffs)
            min_inds = np.argsort(diffs)[:2]
            ## Don't care which  of the two is smaller,
            ## just want in the correct index order:
            min_inds = np.sort(min_inds)
            minims = diffs[min_inds]
            if min_inds[1]-min_inds[0] == 1:
                diffs = np.delete(diffs,min_inds[1])
                diffs = np.delete(diffs,min_inds[0])
                if np.all(np.abs((diffs/np.sum(minims))-1.)<0.1):
                    outlist = inds[ii]
                    outlist = np.delete(outlist,min_inds[1])
                    peak_array[:, pixel] = np.array(outlist)
                    correctly_assigned = True
        elif rowl == (nrows-1):
            diffs = np.diff(inds[ii])
            typical_diff = np.median(diffs)
            max_ind = np.argmax(diffs)
            maxdiff = diffs[max_ind]
            diffs = diffs[diffs!=maxdiff]
            if np.all(np.abs((float(maxdiff)/diffs)-2.)<0.25):
                outlist = list(inds[ii])
                outlist.insert(max_ind+1,int(outlist[max_ind]+typical_diff))
                peak_array[:,pixel] = np.array(outlist)
                correctly_assigned = True

        if not correctly_assigned:
            if ii == 0:
                peak_array[:,pixel] = inds[first_good_ind]
            else:
                peak_array[:,pixel] = peak_array[:,pixel - 1]


    ## For the first and last few pixels, set them equal to the last valid pixel
    for pix in np.arange(binwidth):
        peak_array[:,pix] = peak_array[:,binwidth]
        peak_array[:, -1*(pix+1)] = peak_array[:, -1*(binwidth+1)]
    return peak_array, nrows


def plot_single_tet_apperatures(image,peaks,bots,tops,tet):
    plt.figure()
    pixels = np.arange(image.shape[1])
    plt.imshow(image, origin='lower-left',cmap='Greys',aspect='auto',interpolation ='nearest')

    for fib in peaks.keys():
        peak_set = peaks[fib]
        bot_set = bots[fib]
        top_set = tops[fib]
        plt.plot(pixels,peak_set ,'b-',lw=1)
        plt.plot(pixels,top_set,'r-',lw=1)
        plt.plot(pixels,bot_set,'r-',lw=1)
        plt.text(200,peak_set[200],fib)
    plt.title(str(tet))
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
