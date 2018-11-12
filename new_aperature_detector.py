import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import find_peaks
image = fits.open('apptesting.fits')[0].data
image = np.flipud(image)

camera = 'b'
bad_fibs = ['b113','r804']


def plot_this(image,peak_array,left_array,right_array,deviations,tet=8,badfibs=[],cam='b'):
    plt.figure()
    pixels = np.arange(image.shape[1])
    plt.imshow(image, origin='lower-left',cmap='Greys')
    bad_nums = []
    for fib in bad_fibs:
        if fib[0].lower() == camera.lower():
            if int(fib[1]) == int(tet):
                bad_nums.append(int(fib[2:]))

    bad_nums = np.sort(bad_nums)[::-1]
    fiber_nums = list(np.arange(1, 17))

    for bad_num in bad_nums:
        fiber_nums.pop(bad_num-1)

    for row in np.arange(peak_array.shape[0]):
        peaks = peak_array[row,:]
        devs = deviations[row]
        fibnum = fiber_nums[row]
        plt.plot(pixels,peaks ,'b-')
        plt.plot(pixels,peaks+devs,'r-')
        plt.plot(pixels,peaks-devs,'r-')
        plt.text(200,peaks[200],str(fibnum))
    plt.title(str(tet))
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.tight_layout()
    #plt.show()

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
        peak_array[row,:] = medfilt(peak_array[row,:],71)
        left_array[row, :] = medfilt(left_array[row, :], 71)
        right_array[row, :] = medfilt(right_array[row, :], 71)
        deviations[row] = 1.+np.median(np.ceil((right_array[row,:]-left_array[row,:])/2.))


    return peak_array,left_array,right_array,deviations

def do_all(image,height=5e4,prominence=1e3,tet=8,badfibs=[],cam='b'):
    peak_array, left_array, right_array,deviations = find_peak_inds(image,height,prominence)
    plot_this(image,peak_array,left_array,right_array,deviations,tet,badfibs=badfibs,cam=cam)


if __name__ == '__main__':
    tetris = np.arange(1,9)
    if camera=='b':
        tetris=tetris[::-1]

    rows,cols = image.shape
    pixels = np.arange(2,cols-2)
    from apperature_cut import get_all_tetris_edges
    starts,ends = get_all_tetris_edges(image)

    for tet,start,end in zip(tetris,starts,ends):
        cut_image = image[start:end,:]
        do_all(cut_image,height=5e4,prominence=2e4,tet=tet,badfibs=bad_fibs,cam=camera)
    plt.show()
