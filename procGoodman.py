#!/bin/python
''' 
    Anthony Kremin


 Perform the bias subtraction and remove the relative gain
 differences for a single M2FS spectrograph image or a list of images.
 ie this works on sets of blue or red independently, but taking into account
 the 4 amplifiers in a specific spectrograph.
'''
#-----------------------------------------------------------------------------

import os
import numpy as np
from astropy.io import fits as pyfits

from matplotlib import pyplot
import pdb
# Version and Date

versNum = "1.0"
versDate = "2017-01-13"

############################
#### Define various routines
############################
directory_path = '/u/home/kremin/value_storage/Github/M2FSreduce/example_goodman/'
overwrite = False
#biasl = 597; biash = 626
#filsl = 578; filsh = 637
#nccds = 4
#spectags = ['r','b']


############################
#### Script starts here ####
############################
biaspath = os.path.join(directory_path,'bias')
biasfiles = os.listdir(biaspath)
if not overwrite and ('masterbias.fits' in biasfiles):
    os.path.rename(os.path.join(biaspath,'masterbias.fits'),os.path.join(biaspath,'masterbias.'+str(time.time())+'.fits'))

os.chdir(biaspath)

for bfile in biasfiles:
    if bfile[-4:]!='.fits':
        continue
    testfile = pyfits.open(bfile)
    testnaxis1 = testfile[0].header['NAXIS1']
    testnaxis2 = testfile[0].header['NAXIS2']
    bias = np.zeros((testnaxis2,testnaxis1))
    break
        
# Find bias for this ccd
nbiases = 0
for bfile in biasfiles:
    if bfile[:10] == 'masterbias' or (bfile[-4:]!='.fits'):
         continue
    else:
        nbiases += 1
    biasfile = pyfits.open(bfile)
    bias += np.array(biasfile[0].data)

master_bias = bias/float(len(bias_range))


        # Do the bias calculation
        for filnum in tosubtract_range:
            tsubfilename = '%s%04dc%d' % (spectrograph,filnum,ccd)
            if os.path.isfile(tsubfilename+'.fits'):
                fitsfile = pyfits.open(tsubfilename+'.fits')

                naxis1 = fitsfile[0].header['NAXIS1']
                naxis2 = fitsfile[0].header['NAXIS2']
                assert naxis1 == testnaxis1, "Make sure that the lengths of arrays are consistent"
                assert naxis2 == testnaxis2, "Make sure that the lengths of arrays are consistent"
                overscan_ranges = fitsfile[0].header['BIASSEC'].strip('[').strip(']').split(',')
                overscanx,overscany = [rng.split(':') for rng in overscan_ranges]
                ccdxbin,ccdybin = fitsfile[0].header['BINNING'].split('x')
                detector = fitsfile[0].header['INSTRUME']
                telescope = fitsfile[0].header['TELESCOP']
                current_data = np.array(fitsfile[0].data)
    
                try:
                    bias_subd = current_data - master_bias
                    fitsfile[0].data = bias_subd[:int(overscanx[0]),:int(overscany[0])]
                except:
                    pdb.set_trace()
                
                #fitsfile[0].data = bias_subd[:overscanx[0],:overscany[0]]
                fitsfile[0].header['STATE'] = 'BIAS SUBD'
                if os.path.isfile(tsubfilename+'_b.fits'):
                    if overwrite:
                        print "Overwriting file: "+tsubfilename+'_b.fits'
                        os.remove(tsubfilename+'_b.fits')
                        fitsfile.writeto(tsubfilename+'_b.fits')
                else:
                    print "Making new file: "+tsubfilename+'_b.fits'
                    fitsfile.writeto(tsubfilename+'_b.fits')
