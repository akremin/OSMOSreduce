# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 11:35:05 2017

@author: kremin
"""

import numpy as np
from astropy.io import fits as pyfits
import matplotlib
import os
if os.environ['HOSTNAME'] == 'umdes7.physics.lsa.umich.edu':
    matplotlib.use('Qt4Agg')
    from ds9 import ds9 as DS9
else:
    matplotlib.use('Qt5Agg')
    from pyds9 import DS9
import matplotlib.pyplot as plt
import scipy.signal as signal
from scipy import fftpack
from scipy.optimize import curve_fit
import pickle
import pdb
import fnmatch
import cv2
from slit_find import normalized_Canny, get_template, match_template
from scipy.signal import argrelextrema
import lacosmic
import PyCosmic

def comparemethods(orig,lacos,pycos,dancos,imgtag=''):
    plt.figure()
    plt.subplot(221)
    plt.imshow(orig,cmap='gray'),plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(223)
    plt.imshow(lacos,cmap='gray'),plt.title('LACosmic '+imgtag), plt.xticks([]), plt.yticks([])
    plt.subplot(222)
    plt.imshow(dancos,cmap='gray'),plt.title('Dans Code '+imgtag), plt.xticks([]), plt.yticks([])
    plt.subplot(224)
    plt.imshow(pycos,cmap='gray'),plt.title('PyCosmic '+imgtag), plt.xticks([]), plt.yticks([])
    plt.show()

def filter_image(img):
    img_sm = signal.medfilt(img,5)
    sigma = 2.0 
    bad = np.abs(img-img_sm) / sigma > 8.0
    img_cr = img.copy()
    img_cr[bad] = img_sm[bad]
    return img_cr,bad
    
def getrawdata(datadir,clus_id,masknumber):
    #import, clean, and add science fits files
    sciencefiles = np.array([])
    hdulists_science = np.array([])
    for curfile in os.listdir(datadir+clus_id+'/mask'+masknumber+'/data_products/science/'): #search and import all science filenames
        if fnmatch.fnmatch(curfile, '*b.fits'):
            sciencefiles = np.append(sciencefiles,curfile)
            scifits = pyfits.open(datadir+clus_id+'/mask'+masknumber+'/data_products/science/'+curfile)
            hdulists_science = np.append(hdulists_science,scifits)
    science_file = sciencefiles[0]
    hdulist_science = pyfits.open(datadir+clus_id+'/mask'+masknumber+'/data_products/science/'+science_file)
    naxis1 = hdulist_science[0].header['NAXIS2']
    naxis2 = hdulist_science[0].header['NAXIS1']
    
    #import flat data
    flatfiles = np.array([])
    hdulists_flat = np.array([])
    for curfile in os.listdir(datadir+clus_id+'/mask'+masknumber+'/data_products/flat/'): #search and import all science filenames
        if fnmatch.fnmatch(curfile, '*b.fits'):
            flatfiles = np.append(flatfiles,curfile)
            flatfits = pyfits.open(datadir+clus_id+'/mask'+masknumber+'/data_products/flat/'+curfile)
            hdulists_flat = np.append(hdulists_flat,flatfits)
    if len(hdulists_flat) < 1:
        raise Exception('proc4k.py did not detect any flat files')
    
    #import arc data
    arcfiles = np.array([])
    hdulists_arc = np.array([])
    for curfile in os.listdir(datadir+clus_id+'/mask'+masknumber+'/data_products/comp/'): #search and import all science curfilenames
        if fnmatch.fnmatch(curfile, '*b.fits'):
            arcfiles = np.append(arcfiles,curfile)
            arcfits = pyfits.open(datadir+clus_id+'/mask'+masknumber+'/data_products/comp/'+curfile)
            hdulists_arc = np.append(hdulists_arc,arcfits)
    if len(hdulists_arc) < 1:
        raise Exception('proc4k.py did not detect any arc files')
    return hdulists_science,hdulists_flat,hdulists_arc,naxis1,naxis2

def edges(fitdat):
    color = 'gray'
    fitdat[fitdat > 32768] = 32768 #2**15
    #fitdat[fitdat < 0.0000001] = 0.0000001
    #fitdat=np.log(fitdat)
    imgdata_0 = fitdat-np.min(fitdat)
    imgo = (255.*(imgdata_0/np.max(imgdata_0))).astype(np.uint8)
    img = imgo#cv2.medianBlur(imgo,5) 
    canny = cv2.Canny(img,10,240)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    plt.figure()
    #plt.subplot(2,2,1)
    plt.imshow(canny,cmap = color)
    plt.colorbar()
    plt.title('Canny'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,2)
    plt.figure()
    plt.imshow(laplacian,cmap = color)
    plt.colorbar()
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,3)
    plt.figure()
    plt.imshow(sobelx,cmap = color)
    plt.colorbar()
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    #plt.subplot(2,2,4),
    plt.figure()
    plt.imshow(sobely,cmap = color)
    plt.colorbar()
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    return canny,sobelx,sobely,laplacian
    
def pc():
    plt.close('all')
   
if os.environ['HOSTNAME'] == 'umdes7.physics.lsa.umich.edu':
    datadir = '/u/home/kremin/value_storage/goodman_jan17/'
else:
    datadir = '/home/kremin/SOAR_data/'   
    
clus_id = 'Kremin10'
masknumber = '1'
#cans = []
#sobxs = []
#sobys = []
#laps = [] 


# Take in all the data
#hdulists_science,hdulists_flat,hdulists_arc,naxis1,naxis2 = getrawdata(datadir,clus_id,masknumber)

# Find edges in all the data
#for fit in hdulists_science:
#    canny,sobelx,sobely,laplacian = edges(fit.data)
#    cans.append(canny)
#    sobxs.append(sobelx)
#    sobys.append(sobely)
#    laps.append(laplacian)

    
#original_science = hdulists_science[0].data
#dfilt,danmask = filter_image(hdulists_science[0].data)
#dan = dfilt + np.abs(np.nanmin(dfilt))
#lacos,lacosmask = lacosmic.lacosmic(data=hdulists_science[0].data,contrast=2.5,cr_threshold=10,neighbor_threshold=6,effective_gain=0.56,readnoise=3.69)
#hdulists_science[0].writeto('./science00.fits')
pycos, pycosmask = PyCosmic.detCos('science00.fits','science00_cleanedmask.fits','science00_cleaned.fits',rdnoise='RDNOISE',sigma_det=8,gain=1.0,verbose=True,return_data=True)#gain = 'GAIN'
#dpy = original_science-pycos
#dlac = original_science-lacos
#ddan = original_science-dan
comparemethods(original_science,lacosmask,pycosmask,danmask,'CR Mask')
comparemethods(original_science,lacos,pycos,dan,'CR Removed')
comparemethods(original_science,dlac,dpy,ddan,'(Orig-CR Removed)')
plt.figure(); plt.imshow(lacosmask.astype(np.int8)-pycosmask.astype(np.int8),cmap='gray'); plt.xticks([]), plt.yticks([]); plt.show()