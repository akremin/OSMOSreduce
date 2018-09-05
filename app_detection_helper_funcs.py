import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import cv2
from scipy.signal import medfilt
from scipy.signal import argrelmin, argrelmax, argrelextrema

from scipy.optimize import curve_fit
from scipy.interpolate import CubicSpline
from scipy.interpolate import InterpolatedUnivariateSpline as IUS
from scipy.interpolate import UnivariateSpline as US

def imageset(img):
    cimg = img - np.min(img) + 1e-4
    log_img = np.log(cimg)

    pre_normd = cimg
    normalized = np.float64(pre_normd)/np.max(pre_normd)
    int_img = np.uint8(normalized*(2**8))
    outset = {}
    outset['raw'] = img
    outset['zerod'] = cimg
    outset['log'] = log_img
    outset['normd'] = normalized
    outset['int'] = int_img
    return outset
    
def show_image(thisimg):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image',thisimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gauss(xs,amp,mean,sigma):
    s2 = sigma*sigma
    x2 = (xs-mean)*(xs-mean)
    return amp*(1/np.sqrt(2*np.pi*s2))*np.exp(-x2/(2*s2))
    
def fortran_ind_range(n_nums):
    return range(1,n_nums+1)
    
def get_all_tetris_edges(image):
    sumd = image.sum(axis=1)
    normd_sum = sumd/np.max(sumd)
    plt.figure(), plt.plot(range(len(normd_sum)),normd_sum)

    medfiltd = medfilt(normd_sum,11)
    binary = (medfiltd > 0.04).astype(int)
    plt.figure(), plt.plot(range(len(binary)),binary)

    grad = np.gradient(binary)
    starts = np.where(grad>0)[0]
    ends = np.where(grad<0)[0]
    starts = starts[::2]-4
    ends = ends[::2]+4

    widths = ends-starts
    med_width = np.median(widths)
    width = min(np.max(widths),med_width+12)   

    changes = widths - width - 10 # 10 is for padding
    start_changes = np.floor(changes/2).astype(int)
    end_changes = np.ceil(changes/2).astype(int)

    starts += start_changes
    ends -= end_changes
    
    plt.figure(), plt.subplots(1,1,figsize=(16,16))
    plt.imshow(image,origin='lower-left')
    plt.hlines(starts,xmin=0,xmax=image.shape[0],color='r')
    plt.hlines(ends,xmin=0,xmax=image.shape[0],color='r')
    plt.yticks([]),plt.xticks([])
    plt.show()
    return starts,ends

def get_extrema(compact_slice,nfibers=16):
    ymins = argrelmin(compact_slice)[0].astype(int)
    ymaxs = argrelmax(compact_slice)[0].astype(int)
    
    while len(ymaxs)>nfibers:
        yvals = compact_slice[ymaxs]
        bad_ind = np.argmin(yvals)
        ymaxs = np.delete(ymaxs,bad_ind)
    
    
    #if len(ymins)<(len(ymaxs)+1):
    
    if ymins[1]<ymaxs[0]:
            ymins = np.delete(ymins,1)
    elif ymins[-2]> ymaxs[-1]:
        ymins = np.delete(ymins,-2)
 
    if ymins[0]>ymaxs[0]:
        ymins = np.insert(ymins,0,ymaxs[0]-(ymins[0]-ymaxs[0]))
    if ymins[-1]<ymaxs[-1]:
        ymins = np.append(ymins,ymaxs[-1]+(ymaxs[-1]-ymins[-1]))
        
    while len(ymins)>(nfibers+1):
        if ymins[1]<ymaxs[0]:
            ymins = np.delete(ymins,1)
        elif ymins[-2]> ymaxs[-1]:
            ymins = np.delete(ymins,-2)
        
    if len(ymins)>(nfibers+1):
        plt.plot(range(len(compact_slice)),compact_slice)
        plt.plot(ymins,np.zeros(len(ymins)),'r.')
        plt.plot(ymaxs,np.ones(len(ymaxs))*40,'b.')
        print("problem with mins",len(ymins))
        print(ymins)
    if len(ymaxs)>nfibers:
        plt.plot(range(len(compact_slice)),compact_slice)
        plt.plot(ymins,np.zeros(len(ymins)),'r.')
        plt.plot(ymaxs,np.ones(len(ymaxs))*40,'b.')
        print("problem with maxs",len(ymaxs))
        print(ymaxs)
            

    for iterations in range(3):
        seps = ymins[1:]-ymins[:-1]

        if np.abs(seps[0]-np.median(seps))>2*np.std(seps[1:]):
            questioned_ind = 0
            questioned_yloc = ymins[questioned_ind]
            questioned_yval = compact_slice[questioned_yloc]
            test_yloc = np.clip(np.int(ymins[questioned_ind+1]-np.ceil(np.median(seps))),0,len(compact_slice)-1)
            test_yval = compact_slice[test_yloc]
            if np.abs(questioned_yval-test_yval) < 4:
                #print("changing from {} to {}".format(questioned_yloc,test_yloc))
                ymins[questioned_ind] = test_yloc
            else:
                #print(ymins[1]-ymins[0],np.median(seps),np.mean(seps),np.std(seps))
                #print(questioned_yval,test_yval)
                pass

        seps = ymins[1:]-ymins[:-1]

        if np.abs(seps[-1]-np.median(seps))>2*np.std(seps[:-1]):
            questioned_ind = len(ymins)-1
            questioned_yloc = ymins[questioned_ind]
            questioned_yval = compact_slice[questioned_yloc]
            test_yloc = np.clip(np.int(ymins[questioned_ind-1]+np.ceil(np.median(seps))),0,len(compact_slice)-1)
            test_yval = compact_slice[test_yloc]
            if np.abs(questioned_yval-test_yval) < 4:
                #print("changing from {} to {}".format(questioned_yloc,test_yloc))
                ymins[questioned_ind] = test_yloc
            else:
                #print(ymins[-1]-ymins[-2],np.median(seps),np.mean(seps),np.std(seps))
                #print(questioned_yval,test_yval)
                pass

    if len(ymins)>(nfibers+1):
        plt.plot(range(len(compact_slice)),compact_slice)
        plt.plot(ymins,np.zeros(len(ymins)),'r.')
        plt.plot(ymaxs,np.ones(len(ymaxs))*40,'b.')
        print("problem with mins",len(ymins))
        print(ymins)
    if len(ymaxs)>nfibers:
        plt.plot(range(len(compact_slice)),compact_slice)
        plt.plot(ymins,np.zeros(len(ymins)),'r.')
        plt.plot(ymaxs,np.ones(len(ymaxs))*40,'b.')
        print("problem with maxs",len(ymaxs))
        print(ymaxs)
            
    return ymins,ymaxs

def recursive_poly(x,order,params):
    if order == 0:
        return params
    elif order == 1:
        return recursive_poly(x,order-1,params[0])*x+params[-1]
    else:
        return recursive_poly(x,order-1,params[:-1])*x+params[-1]

def recursive_poly_np(x,order,params):
    return np.asarray(recursive_poly(x,order,params)).astype(np.float64)

def quad(x,a,b,c):
    return recursive_poly_np(x,2,[a,b,c])
def fourthorder(x,a,b,c,d,e):
    return recursive_poly_np(x,4,[a,b,c,d,e])
def sixthorder(x,a,b,c,d,e,f,g):
    return recursive_poly_np(x,6,[a,b,c,d,e,f,g])
def eighthorder(x,a,b,c,d,e,f,g,h,i):
    return recursive_poly_np(x,8,[a,b,c,d,e,f,g,h,i])
def tenthorder(x,a,b,c,d,e,f,g,h,i,j,k):
    return recursive_poly_np(x,10,[a,b,c,d,e,f,g,h,i,j,k])

def fitter_switch():
    return {2:quad, 4:fourthorder, 6: sixthorder, 8:eighthorder, 10:tenthorder}

def find_aperatures(fibermap,camera='r',deadfibers=None,resol_factor=100.,nvertslices=2**6,
                      function_order=4):
    mapping_image = imageset(fibermap)
    
    fiber_iterator = np.arange(1,16+1)
    tetris_iterator_dict = {i:fiber_iterator.copy() for i in range(1,8+1)}
    if deadfibers is not None:
        for deadfiber in deadfibers:
            if deadfiber[0] == camera:
                tetnum = int(deadfiber[1])
                fibernum = int(deadfiber[2:])
                curtetfibs = tetris_iterator_dict[tetnum]
                tetris_iterator_dict[tetnum] = curtetfibs[curtetfibs != fibernum]
            
    if type(function_order) is str:
        dopoly = False
        dospline = True
        sval = 3*len(xcoords[0])
    else:
        dopoly = True
        dospline = False
        fitter_func_switch = fitter_switch()
        fitfunc = fitter_func_switch[function_order]
        p0 = [0.001]*(function_order+1)#,0.001,0.001,0.001,0.001,0.001,0.001)
        
    starts,ends = get_all_tetris_edges(mapping_image['zerod'])

    tetrinums = np.asarray(list(tetris_iterator_dict.keys()))
    tetrinums.sort()
    if camera.lower() == 'r':
        tetris_numbering = tetrinums
    elif camera.lower() == 'b':
        tetris_numbering = tetrinums[::-1]
    else:
        print("Didn't understand the camera name")
    del tetrinums
    
    ntetri = len(tetris_numbering)
    stepsize = int(mapping_image['zerod'].shape[1]/nvertslices)

    aperatures = {}
    aperatures['camera'] = camera
    aperatures['resolution_factor'] = resol_factor
    aperatures['tetri'] = {}
    for itti,start,end in zip(np.arange(len(starts)),starts,ends):
        tetnum = tetris_numbering[itti]
        tetris = {}
        tetris['tetris_start'] = starts[itti]
        tetris['tetris_end'] = ends[itti]
        tetris['fibers'] = {}
    
        tet = mapping_image['zerod'][start:end,:].copy()
        tetris_image = tet/tet.max()
        xpix = np.arange(tetris_image.shape[1])
        
        fiber_numbers = tetris_iterator_dict[tetnum]
        xcoords = np.zeros(shape=(nvertslices))
        ymin_aps = np.zeros(shape=(nvertslices,len(fiber_numbers)+1))
        ymax_locs =  np.zeros(shape=(nvertslices,len(fiber_numbers))) 

        for vertslice in range(nvertslices):
            curslice = tetris_image[:,vertslice*stepsize:(vertslice+1)*stepsize]
            compact_slice = curslice.sum(axis=1)
            ymins, ymaxs = get_extrema(compact_slice,nfibers=len(fiber_numbers))
            mean_x = (vertslice+0.5)*stepsize
            xcoords[vertslice] = mean_x
            ymin_aps[vertslice,:] = ymins
            ymax_locs[vertslice,:] = ymaxs
            
        fit_mins = {}
        if dopoly:
            p0[-1] = 5
            fit_min_0 = curve_fit(fitfunc, xcoords, ymin_aps[:,0],  p0=p0 )[0]
        else: #dospline
            fit_min_0 = US(xcoords,ymin_aps[:,0].astype(np.float64), s=sval )                     
        fit_mins[0] = fit_min_0
        
        for ittj,fibnum in enumerate(fiber_numbers):
            fit_min1 = fit_mins[ittj]
            if dopoly:    
                p0[-1] = 10*fibnum+5
                fit_min2 = curve_fit(fitfunc, xcoords, ymin_aps[:,ittj+1], p0=p0)[0]
                p0[-1] = 10*fibnum
                fit_max  = curve_fit(fitfunc, xcoords, ymax_locs[:,ittj],  p0=p0)[0]
                fllows = fitfunc(xpix.astype(float),*fit_min1)
                flhighs = fitfunc(xpix.astype(float),*fit_min2)        
            elif dospline:
                fit_min2 = US(xcoords, ymin_aps[:,ittj+1].astype(np.float64), s=sval )
                fit_max =  US(xcoords, ymax_locs[:, ittj].astype(np.float64), s=sval )
                fllows =  fit_min1(xpix.astype(float))
                flhighs = fit_min2(xpix.astype(float))

            lows,highs = resol_factor*np.asarray(fllows),resol_factor*np.asarray(flhighs)
            hr_lows = np.floor(lows).astype(int)
            hr_highs = np.ceil(highs).astype(int)
            maxwidth = np.max(np.abs(hr_highs-hr_lows))+1
            cut_mid = maxwidth//2
            lbs,ubs = [],[]
            xpix = np.arange(tetris_image.shape[1])
            for low,high,x in zip(hr_lows,hr_highs,xpix):
                npix = np.abs(low-high) + 1
                halfpix = npix//2
                lbs.append(cut_mid-halfpix)
                ubs.append(cut_mid+(npix-halfpix))

            fit_mins[ittj+1] = fit_min2
            
            fiber  = {}
            fiber['hr_lowbounds'] = hr_lows
            fiber['hr_highbounds'] = hr_highs
            fiber['lowbounds'] = lbs
            fiber['highbounds'] = ubs
            fiber['lower_apps'] = fllows
            fiber['higher_apps'] = flhighs
            tetris['fibers'][fibnum] = fiber
                              
        aperatures['tetri'][tetnum] = tetris

    return aperatures


def cutout_2d_aperatures(image,aperatures):
    camera = aperatures['camera']
    resol_factor = aperatures['resolution_factor']

    all_cutouts = {}

    for i,tetris in aperatures['tetri'].items():
        start,end = tetris['tetris_start'],tetris['tetris_end']
        tetris_cutout = image[start:end,:]
        
        hyperres = np.ndarray(shape=(tetris_cutout.shape[0]*resol_factor,tetris_cutout.shape[1]))
        for row in range(tetris_cutout.shape[0]):
            hyperres[resol_factor*row:resol_factor*(row+1),:] = tetris_cutout[row,:]/resol_factor

        xpix = np.arange(tetris_cutout.shape[1])
        cutouts = {}
        for j,fiber in tetris['fibers'].items():
            maxwidth = np.max(np.abs(fiber['hr_highbounds']-fiber['hr_lowbounds']))+1
            cutout_array = np.zeros(shape=(maxwidth,tetris_cutout.shape[1]))
            zippd_iterator = zip(fiber['lowbounds'],fiber['highbounds'],
                                 fiber['hr_lowbounds'],fiber['hr_highbounds'],xpix)
            for lowerbound,upperbound,lowhr,highhr,x in zippd_iterator:
                npix = np.abs(lowhr-highhr) + 1
                if npix == maxwidth:
                    cutout_array[:,x] = hyperres[lowhr:highhr+1,x]
                else:
                    cutout_array[lowerbound:upperbound,x] = hyperres[lowhr:highhr+1,x]
            cutouts[j] = cutout_array
        all_cutouts[i] = cutouts
        
    flattened_cutouts = {}
    for tetrisnumber,tetrisdict in all_cutouts.items():
        for fibernumber,fiber_cutout in tetrisdict.items():
            flattened_name = '{}{}{:02d}'.format(camera,tetrisnumber,int(fibernumber))
            flattened_cutouts[flattened_name] = fiber_cutout
    return flattened_cutouts
        
def cutout_1d_aperatures(image,aperatures):
    twod_cutouts = cutout_2d_aperatures(image,aperatures)
    oned_cutouts = {}
    for key,cutout in twod_cutouts.items():
        oned_cutout = cutout.sum(axis=0)
        oned_cutouts[key] = oned_cutout[::-1]
        
    return oned_cutouts