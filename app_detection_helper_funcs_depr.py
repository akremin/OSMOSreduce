import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import cv2
from scipy.signal import medfilt
from scipy.signal import argrelmin, argrelmax, argrelextrema

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

def get_extrema(compact_slice):
    ymins = argrelmin(compact_slice)[0].astype(int)
    ymaxs = argrelmax(compact_slice)[0].astype(int)
    
    while len(ymaxs)>16:
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
        
    while len(ymins)>17:
        if ymins[1]<ymaxs[0]:
            ymins = np.delete(ymins,1)
        elif ymins[-2]> ymaxs[-1]:
            ymins = np.delete(ymins,-2)
        
    if len(ymins)>17:
        plt.plot(range(len(compact_slice)),compact_slice)
        plt.plot(ymins,np.zeros(len(ymins)),'r.')
        plt.plot(ymaxs,np.ones(len(ymaxs))*40,'b.')
        print("problem with mins",len(ymins))
        print(ymins)
    if len(ymaxs)>16:
        plt.plot(range(len(compact_slice)),compact_slice)
        plt.plot(ymins,np.zeros(len(ymins)),'r.')
        plt.plot(ymaxs,np.ones(len(ymaxs))*40,'b.')
        print("problem with maxs",len(ymaxs))
        print(ymaxs)
            

    for iterations in range(3):
        seps = ymins[1:]-ymins[:-1]

        if np.abs(seps[0]-np.median(seps))>2*np.std(seps[1:]):
            questioned_ind = 0
            #print("first pt",seps[0],np.median(seps),np.std(seps),questioned_ind,ymins[questioned_ind])
            questioned_yloc = ymins[questioned_ind]
            questioned_yval = compact_slice[questioned_yloc]
            test_yloc = np.clip(np.int(ymins[questioned_ind+1]-np.ceil(np.median(seps))),0,len(compact_slice)-1)
            test_yval = compact_slice[test_yloc]
            if np.abs(questioned_yval-test_yval) < 4:
                print("changing from {} to {}".format(questioned_yloc,test_yloc))
                ymins[questioned_ind] = test_yloc
            else:
                print(ymins[1]-ymins[0],np.median(seps),np.mean(seps),np.std(seps))
                print(questioned_yval,test_yval)

        seps = ymins[1:]-ymins[:-1]

        if np.abs(seps[-1]-np.median(seps))>2*np.std(seps[:-1]):
            questioned_ind = len(ymins)-1
            #print("last pt",seps[-1],np.median(seps),np.std(seps),questioned_ind,ymins[questioned_ind])
            questioned_yloc = ymins[questioned_ind]
            questioned_yval = compact_slice[questioned_yloc]
            test_yloc = np.clip(np.int(ymins[questioned_ind-1]+np.ceil(np.median(seps))),0,len(compact_slice)-1)
            test_yval = compact_slice[test_yloc]
            if np.abs(questioned_yval-test_yval) < 4:
                print("changing from {} to {}".format(questioned_yloc,test_yloc))
                ymins[questioned_ind] = test_yloc
            else:
                print(ymins[-1]-ymins[-2],np.median(seps),np.mean(seps),np.std(seps))
                print(questioned_yval,test_yval)

    if len(ymins)>17:
        plt.plot(range(len(compact_slice)),compact_slice)
        plt.plot(ymins,np.zeros(len(ymins)),'r.')
        plt.plot(ymaxs,np.ones(len(ymaxs))*40,'b.')
        print("problem with mins",len(ymins))
        print(ymins)
    if len(ymaxs)>16:
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

def find_aperatures(mapping_image,camera='r',deadfibers=None,resol_factor=100,nsteps=2**6,
                      function_order=4):
    fibermap = imageset(mapping_image)
    
    fiber_iterator = np.arange(1,16+1)
    tetris_iterator_dict = {i:fiber_iterator.copy() for i in range(1,8+1)}
    if deadfibers is not None:
        for deadfiber in deadfibers:
            if deadfiber[0] == camera:
                tetnum = int(deadfiber[1])
                fibernum = int(deadfiber[2:])
                curtetfibs = tetris_iterator_dict[tetnum]
                tetris_iterator_dict[tetnum] = curtetfibs[curtetfibs != fibernum]
            
    #if 'camera' in relevant_info.keys():
    #    camera = relevant_info['camera']
    #else:
    #    camera = 'r'
    #if 'deadfibers' in relevant_info.keys():
    #    dead_fibers = relevant_info['deadfibers']
    #else:
    #    dead_fibers = None
    #if 'resolution_factor' in relevant_info.keys():
    #    resol_factor = relevant_info['resolution_factor']
    #else:
    #    resol_factor = 100
    #if 'nsteps' in relevant_info.keys():
    #    nsteps = relevant_info['nsteps']
    #else:
    #    nsteps = 2**6
    #if 'function_order' in relevant_info.keys():
    #    function_order = relevant_info['function_order']
    #else:
    #    function_order = 4
        
    if type(function_order) is str:
        dopoly = False
        dospline = True
        sval = 3*len(xcoords[0])
    else:
        dopoly = True
        dospline = False
        fitfunc = fitter_func_switch[function_order]
        p0 = [0.001]*(function_order+1)#,0.001,0.001,0.001,0.001,0.001,0.001)
    starts,ends = get_all_tetris_edges(fibermap['zerod'])

    if camera.lower() == 'r':
        tetris_number = 1
    elif camera.lower() == 'b':
        tetris_number = 8
    else:
        print("Didn't understand the camera name")

    fibm_tetri = {}
    for start,end in zip(starts,ends):
        fibtet = fibermap['zerod'][start:end,:].copy()
        fibm_tetri[tetris_number] = fibtet/fibtet.max()
        if camera.lower() == 'r':
            tetris_number += 1
        elif camera.lower() == 'b':
            tetris_number -= 1

    ntetri = len(fibm_tetri.items())
    nappers = np.zeros(ntetri)
    stepsize = int(fibm_tetri[1].shape[1]/nsteps)
    xcoords = np.zeros(shape=(ntetri,nsteps))
    ymin_aps = np.zeros(shape=(ntetri,nsteps,17))
    ymax_locs =  np.zeros(shape=(ntetri,nsteps,16))    
    all_lower_apps = {}
    all_upper_apps = {}
    all_lower_bounds = {}
    all_upper_bounds = {}
    all_lower_hrbounds = {}
    all_upper_hrbounds = {}

    for i in range(1,ntetri+1):
        tetras_image = fibm_tetri[i]
        for j in range(nsteps):
            curslice = tetras_image[:,j*stepsize:(j+1)*stepsize]
            compact_slice = curslice.sum(axis=1)
            ymins, ymaxs = get_extrema(compact_slice)
            mean_x = (j+0.5)*stepsize
            xcoords[i-1,j] = mean_x
            ymin_aps[i-1,j,:] = ymins
            ymax_locs[i-1,j,:] = ymaxs
            
        fit_mins,fit_maxs = {},{}
        lower_apps = {}
        upper_apps = {}
        lower_bounds = {}
        upper_bounds = {}
        lower_hrbounds = {}
        upper_hrbounds = {}
        tetris_image = fibm_tetri[i]
        xpix = np.arange(tetris_image.shape[1])
        itter_xcoords = xcoords[i-1].astype(np.float64)
        p0[-1] = 5
        if dopoly:
            fit_min_0 = curve_fit(fitfunc, xcoords[i-1], ymin_aps[i-1,:,0],  p0=p0 )[0]
        else: #dospline
            fit_min_0 = US(itter_xcoords,ymin_aps[i-1,:,0].astype(np.float64), s=sval )
        fit_mins[0] = [0,fit_min_0]
        for j in range(1,16+1):
            fit_min1 = fit_mins[j-1][1]
            if dopoly:    
                p0[-1] = 10*j+5
                fit_min2 = curve_fit(fitfunc, xcoords[i-1], ymin_aps[i-1,:,j],    p0=p0)[0]
                p0[-1] = 10*j
                fit_max  = curve_fit(fitfunc, xcoords[i-1], ymax_locs[i-1,:,j-1], p0=p0)[0]
                fllows = fitfunc(xpix.astype(float),*fit_min1)
                flhighs = fitfunc(xpix,*fit_min2)        
            elif dospline:
                fit_min2 = US(itter_xcoords, ymin_aps[i-1,:,j].astype(np.float64),s=sval )
                fit_max = US(itter_xcoords, ymax_locs[i-1,:,j-1].astype(np.float64),s=sval )
                fllows = fit_min1(xpix)
                flhighs = fit_min2(xpix)

            lows,highs = resol_factor*np.asarray(fllows),resol_factor*np.asarray(flhighs)
            hr_lows = np.floor(lows).astype(int)
            hr_highs = np.ceil(highs).astype(int)
            maxwidth = np.max(np.abs(hr_highs-hr_lows))+1
            cut_mid = maxwidth//2
            lbs = []
            ubs = []  
            xpix = np.arange(tetris_image.shape[1])
            for low,high,x in zip(hr_lows,hr_highs,xpix):
                npix = np.abs(low-high) + 1
                halfpix = npix//2
                lowerbound = cut_mid-halfpix
                upperbound = cut_mid+(npix-halfpix)
                lbs.append(lowerbound)
                ubs.append(upperbound)

            lower_apps[j] = fllows
            upper_apps[j] = flhighs
            lower_bounds[j] = lbs
            upper_bounds[j] = ubs
            lower_hrbounds[j] = hr_lows
            upper_hrbounds[j] = hr_highs
            fit_mins[j] = [fit_min1,fit_min2]
            fit_maxs[j] = fit_max

        all_lower_apps[i] = lower_apps
        all_upper_apps[i] = upper_apps
        all_lower_hrbounds[i] = lower_hrbounds
        all_upper_hrbounds[i] = upper_hrbounds  
        all_lower_bounds[i] = lower_bounds
        all_upper_bounds[i] = upper_bounds  

    aperatures = {}
    aperatures['camera'] = camera
    aperatures['resolution_factor'] = resol_factor
    aperatures['tetri'] = {}
    for i in range(1,8+1):
        tetris = {}
        tetris['tetris_start'] = starts[i]
        tetris['tetris_end'] = ends[i]
        tetris['fibers'] = {}
        for j in range(1,16+1):
            fiber  = {}
            fiber['hr_lowbounds'] = all_lower_hrbounds[i][j]
            fiber['hr_highbounds'] = all_upper_hrbounds[i][j]
            fiber['lowbounds'] = all_lower_bounds[i][j]
            fiber['highbounds'] = all_upper_bounds[i][j]
            fiber['lower_apps'] = all_lower_apps[i][j]
            fiber['higher_apps'] = all_upper_apps[i][j]
            tetris['fibers'][j] = fiber
        aperatures['tetri'][i] = tetris
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
        oned_cutouts[key] = oned_cutout
        
    return oned_cutouts