#! /usr/bin/env python
''' 
    Anthony Kremin

    Lightly Modified from  Paul Martini's proc4k.py (for OSMOS)
    for use on M2FS data.

 Perform the overscan subtraction and remove the relative gain
 differences for a single M2FS spectrograph image or a list of images.
 ie this works on sets of blue or red independently, but taking into account
 the 4 amplifiers in a specific spectrograph.

 Steps:
  1. determine if input is a file or list of files
  2. identify binning, size of overscan
  3. remove overscan and trim
  4. remove relative gain variations (TBD)

   8 Sep 2011: initial version for just bias subtraction
  12 Sep 2011: tested, adapted to run on MDM computers, work on MDM4K data
  16 Sep 2011: added glob module
   8 Feb 2012: fixed error in even/odd column definitions, added more
 		tracking and debugging information

'''
#-----------------------------------------------------------------------------
#from __future__ import division
import string
import os
import sys
import numpy as np
from astropy.io import fits as pyfits
import glob
from matplotlib import pyplot
from procM2FSfuncs import filt,ftlgd,usage

# Version and Date

versNum = "1.0"
versDate = "2016-08-04"

############################
#### Define various routines
############################

scriptname = sys.argv[0][string.rfind(sys.argv[0], "/") + 1::]

############################
#### Script starts here ####
############################

Debug = False
BiasSingle = 0
BiasRow = 1
BiasFit = 2
#BiasType = BiasRow
BiasType = BiasSingle
#BiasType = BiasFit
Gain = False	# keep as False until gain values are known
R4K = True

# Gain values for each amplifier [to be computed]
r4k_gain_q1e = 1.0
r4k_gain_q1o = 1.0
r4k_gain_q2e = 1.0
r4k_gain_q2o = 1.0
r4k_gain_q3e = 1.0
r4k_gain_q3o = 1.0
r4k_gain_q4e = 1.0
r4k_gain_q4o = 1.0
mdm4k_gain_q1 = 1.0
mdm4k_gain_q2 = 1.0
mdm4k_gain_q3 = 1.0
mdm4k_gain_q4 = 1.0

# switch to more primitive (slower) code at MDM
AT_MDM = False
user = os.getlogin()
if string.find(user, 'obs24m') >= 0 or string.find(user, 'obs13m') >= 0:
    AT_MDM = True

files = []
for inputs in sys.argv[1:]:
    files.append(glob.glob(inputs))

if len(files) == 0:
    usage(scriptname, versNum, versDate)
    sys.exit(1)

for fil in files:
    if os.path.isfile(fil):
        fitsfile = pyfits.open(fil)
        naxis1 = fitsfile[0].header['NAXIS1']
        naxis2 = fitsfile[0].header['NAXIS2']
        overscan_ranges = fitsfile[0].header['BIASSEC'].strip('[').strip(']').split(',')
        overscans = [rng.split(':') for rng in overscan_ranges]
        overscanx = overscans[0]
        overscany = overscans[1]
        binning = fitsfile[0].header['BINNING'].split('x')
        ccdxbin = binning[0]
        ccdybin = binning[1]
        detector = fitsfile[0].header['INSTRUME']
        telescope = fitsfile[0].header['TELESCOP']
        overscanx /= ccdxbin
        overscany /= ccdybin


        #print file, naxis1, naxis2, overscanx, overscany, detector
        print "Processing %s[%d:%d] OVERSCANX=%d:%d OVERSCANY=%d:%d from %s \
        obtained at the %s" \
        % (fil, naxis1, naxis2, overscanx[0],overscanx[1], overscany[0],overscany[1], detector, telescope)


        if 'R4K' in detector:
            # if not R4K, assume MDM4K
            R4K = False
        #   IRAF units: 1:32, 33:556, 557:1080, 1081:1112
        # Python units: 0:31, 32:555, 556:1079, 1080:1111
        c1 = overscanx        # 32   first image column counting from *zero*
        c2 = int(0.5 * naxis1) - 1    # 555  last image column on first half
        c3 = c2 + 1        # 556  first image column on second half
        c4 = naxis1 - overscanx - 1    # 1079 last image column
        r1 = overscany        # 0    first image row
        r2 = int(0.5 * naxis2) - 1    # 523  last image row on first half
        r3 = r2 + 1        # 524  first image row on second half
        r4 = naxis2 - overscany - 1    # 1047 last image row
        outnaxis1 = c4 - c1 + 1        # 1048 columns in output, trimmed image
        outnaxis2 = r4 - r1 + 1        # 1048 rows in output, trimmed image
        collen = int(0.5 * outnaxis1)    # number of rows in an image quadrant
        rowlen = int(0.5 * outnaxis2)    # number of rows in an image quadrant

    #
    # Assumed layout: (ds9 perspective)
      #      #    #
    #    q2    q4
      #      #    #
    #    q1    q3
      #      #    #
    # each R4K quadrant has an even 'e' and an odd 'o' amplifier
    #
      #      #    #
    #    q1    q2         2->1    4->2   3-> 3     1->4
      #      #    #
    #    q4    q3
      #      #    #
        if Debug:
            print "Quadrants in IRAF pixels: "
            print " q1: [%d : %d, %d : %d] " % (c1 + 1, c2 + 1, r1 + 1, r2 + 1)
            print " q2: [%d : %d, %d : %d] " % (c1 + 1, c2 + 1, r3 + 1, r4 + 1)
            print " q3: [%d : %d, %d : %d] " % (c3 + 1, c4 + 1, r1 + 1, r2 + 1)
            print " q4: [%d : %d, %d : %d] " % (c3 + 1, c4 + 1, r3 + 1, r4 + 1)
        ## Calculate the bias level for each amplifier
        data = fitsfile[0].data
        # identify the columns to use to calculate the bias level
        # skip the first and last columns of the overscan
        # changed to 'list' for hiltner due to primitive python version
        starti = 4 / ccdxbin
        if AT_MDM:
            if R4K:
                cols_over_q1e = list(np.arange(starti, overscanx - 2, 2))
                cols_over_q1o = list(np.arange(starti + 1, overscanx - 2, 2))
                cols_over_q2e = cols_over_q1e
                cols_over_q2o = cols_over_q1o
                cols_over_q3e = list(np.arange(naxis1 - overscanx + starti, naxis1 - 2, 2))
                cols_over_q3o = list(np.arange(naxis1 - overscanx + starti + 1, naxis1 - 2, 2))
                cols_over_q4e = cols_over_q3e
                cols_over_q4o = cols_over_q3o
                cols_q1e = list(np.arange(c1, c2, 2))
                cols_q1o = list(np.arange(c1 + 1, c2 + 2, 2))
                cols_q2e = cols_q1e
                cols_q2o = cols_q1o
                cols_q3e = list(np.arange(c3, c4, 2))
                cols_q3o = list(np.arange(c3 + 1, c4 + 2, 2))
                cols_q4e = cols_q3e
                cols_q4o = cols_q3o
            else:
                cols_over_q1 = list(np.arange(starti, overscanx - 2, 1))
                cols_over_q2 = cols_over_q1
                cols_over_q3 = list(np.arange(naxis1 - overscanx + starti, naxis1 - 2, 1))
                cols_over_q4 = cols_over_q3
                cols_q1 = list(np.arange(c1, c2 + 1, 1))
                cols_q2 = cols_q1
                cols_q3 = list(np.arange(c3, c4 + 1, 1))
                cols_q4 = cols_q3
        else:
            if R4K:
                # identify the even and odd columns in the overscan
                cols_over_q1e = np.arange(starti, overscanx - starti, 2)
                cols_over_q1o = np.arange(starti + 1, overscanx - starti, 2)
                cols_over_q2e = cols_over_q1e
                cols_over_q2o = cols_over_q1o
                cols_over_q3e = np.arange(naxis1 - overscanx + starti, naxis1 - starti, 2)
                cols_over_q3o = np.arange(naxis1 - overscanx + starti + 1, naxis1 - starti, 2)
                cols_over_q4e = cols_over_q3e
                cols_over_q4o = cols_over_q3o
                # identify the even and odd columns in each quadrant
                cols_q1e = np.arange(c1, c2, 2)
                cols_q2e = cols_q1e
                cols_q1o = np.arange(c1 + 1, c2 + 2, 2)
                cols_q2o = cols_q1o
                cols_q3e = np.arange(c3, c4, 2)
                cols_q4e = cols_q3e
                cols_q3o = np.arange(c3 + 1, c4 + 2, 2)
                cols_q4o = cols_q3o
            else:
                cols_over_q1 = np.arange(starti, overscanx-2, 1)
                cols_over_q2 = cols_over_q1
                cols_over_q3 = np.arange(naxis1-overscanx+starti, naxis1-2, 1)
                cols_over_q4 = cols_over_q3
                cols_q1 = np.arange(c1,c2+1,1)
                cols_q2 = cols_q1
                cols_q3 = np.arange(c3,c4+1,1)
                cols_q4 = cols_q3
        if Debug:
            print "Overscan columns: "
            print "Q1/Q2 overscan even first and last columns:", cols_over_q1e[0], cols_over_q1e[-1], len(cols_over_q1e)
            print "Q1/Q2 overscan odd first and last columns:", cols_over_q1o[0], cols_over_q1o[-1], len(cols_over_q1o)
            print "Q3/Q4 overscan even first and last columns:", cols_over_q3e[0], cols_over_q3e[-1], len(cols_over_q3e)
            print "Q3/Q4 overscan odd first and last columns:", cols_over_q3o[0], cols_over_q3o[-1], len(cols_over_q3o)
        if Debug:
            print "Image columns: "
            print "Q1/Q2 even first and last columns:", cols_q1e[0], cols_q1e[-1], len(cols_q1e), r1, r2, len(cols_q1e)
            print "Q1/Q2 odd first and last columns:", cols_q1o[0], cols_q1o[-1], len(cols_q1o), r1+rowlen, r2+rowlen, len(cols_q1o)
            print "Q3/Q4 even first and last columns:", cols_q3e[0], cols_q3e[-1], len(cols_q3e), r1, r2, len(cols_q3e)
            print "Q3/Q4 odd first and last columns:", cols_q3o[0], cols_q3o[-1], len(cols_q3o), r1+rowlen, r2+rowlen, len(cols_q3o)
        # create arrays with the median overscan vs. row for each amplifier
        if R4K:
            bias_q1e = np.zeros(rowlen, dtype=float)
            bias_q1o = np.zeros(rowlen, dtype=float)
            bias_q2e = np.zeros(rowlen, dtype=float)
            bias_q2o = np.zeros(rowlen, dtype=float)
            bias_q3e = np.zeros(rowlen, dtype=float)
            bias_q3o = np.zeros(rowlen, dtype=float)
            bias_q4e = np.zeros(rowlen, dtype=float)
            bias_q4o = np.zeros(rowlen, dtype=float)
        else:
            bias_q1 = np.zeros(rowlen, dtype=float)
            bias_q2 = np.zeros(rowlen, dtype=float)
            bias_q3 = np.zeros(rowlen, dtype=float)
            bias_q4 = np.zeros(rowlen, dtype=float)
        # calculate 1-D bias arrays for each amplifier
        for i in range(r1, r2+1, 1):
            if R4K:
                bias_q1e[i] = np.median(data[i,cols_over_q1e]) 	# data[rows, columns]
                bias_q1o[i] = np.median(data[i,cols_over_q1o])
                bias_q2e[i] = np.median(data[i+rowlen,cols_over_q2e])
                bias_q2o[i] = np.median(data[i+rowlen,cols_over_q2o])
                bias_q3e[i] = np.median(data[i,cols_over_q3e])
                bias_q3o[i] = np.median(data[i,cols_over_q3o])
                bias_q4e[i] = np.median(data[i+rowlen,cols_over_q4e])
                bias_q4o[i] = np.median(data[i+rowlen,cols_over_q4o])
            else: #MDM4K
                bias_q1[i] = np.median(data[i,cols_over_q1]) 	# data[rows, columns]
                bias_q2[i] = np.median(data[i+rowlen,cols_over_q2])
                bias_q3[i] = np.median(data[i,cols_over_q3])
                bias_q4[i] = np.median(data[i+rowlen,cols_over_q4])

    ##########################################################################
    # Subtract the bias from the output
    ##########################################################################

        if BiasType == BiasSingle:
            OverscanKeyValue = 'BiasSingle'
            suffix = 'b'
      # subtract a single bias value for each amplifier
            if R4K:
                bq1e = np.median(bias_q1e)
                bq1o = np.median(bias_q1o)
                bq2e = np.median(bias_q2e)
                bq2o = np.median(bias_q2o)
                bq3e = np.median(bias_q3e)
                bq3o = np.median(bias_q3o)
                bq4e = np.median(bias_q4e)
                bq4o = np.median(bias_q4o)
                if AT_MDM:
                    for r in range(r1,r2+1):
                        for c in cols_q1e:
                            data[r,c] -= bq1e
                        for c in cols_q1o:
                            data[r,c] -= bq1o
                        for c in cols_q2e:
                            data[r+rowlen,c] -= bq2e
                        for c in cols_q2o:
                            data[r+rowlen,c] -= bq2o
                        for c in cols_q3e:
                            data[r,c] -= bq3e
                        for c in cols_q3o:
                            data[r,c] -= bq3o
                        for c in cols_q4e:
                            data[r+rowlen,c] -= bq4e
                        for c in cols_q4o:
                            data[r+rowlen,c] -= bq4o
                else:
                    data[r1:r2+1,cols_q1e] -= bq1e
                    data[r1:r2+1,cols_q1o] -= bq1o
                    data[r3:r4+1,cols_q2e] -= bq2e
                    data[r3:r4+1,cols_q2o] -= bq2o
                    data[r1:r2+1,cols_q3e] -= bq3e
                    data[r1:r2+1,cols_q3o] -= bq3o
                    data[r3:r4+1,cols_q4e] -= bq4e
                    data[r3:r4+1,cols_q4o] -= bq4o
            else:
                bq1 = np.median(bias_q1)
                bq2 = np.median(bias_q2)
                bq3 = np.median(bias_q3)
                bq4 = np.median(bias_q4)
                if AT_MDM:
                    for r in range(r1,r2+1):
                        for c in cols_q1:
                            data[r,c] -= bq1
                        for c in cols_q2:
                            data[r+rowlen,c] -= bq2
                        for c in cols_q3:
                            data[r,c] -= bq3
                        for c in cols_q4:
                            data[r+rowlen,c] -= bq4
                else:
                    data[r1:r2+1,cols_q1] -= bq1
                    data[r3:r4+1,cols_q2] -= bq2
                    data[r1:r2+1,cols_q3] -= bq3
                    data[r3:r4+1,cols_q4] -= bq4
        elif BiasType == BiasRow:
            # not implemented on Hiltner, for MDM4K, etc.
            print "Warning: This mode has not been fully tested"
            OverscanKeyValue = 'BiasRow'
      # subtract a bias value for each row of each amplifier
      #print r1, r2, len(bias_q1e)
            suffix = 'br'
            for i in range(r1, r2 + 1, 1):
                data[i,cols_q1e] -= bias_q1e[i]
                data[i,cols_q1o] -= bias_q1o[i]
                data[i+rowlen,cols_q2e] -= bias_q2e[i]
                data[i+rowlen,cols_q2o] -= bias_q2o[i]
                data[i,cols_q3e] -= bias_q3e[i]
                data[i,cols_q3o] -= bias_q3o[i]
                data[i+rowlen,cols_q4e] -= bias_q4e[i]
                data[i+rowlen,cols_q4o] -= bias_q4o[i]
        elif BiasType == BiasFit:
            OverscanKeyValue = 'BiasFit'
   #         print "Error: Have not implemented a fit to the bias yet. Please use BiasSingle"
            suffix = 'bf'

            xl = range(r1, r2 + 1, 1)
            d = 4

            f_q1e = filt(xl, bias_q1e)
            f_q1o = filt(xl, bias_q1o)
            f_q2e = filt(xl, bias_q2e)
            f_q2o = filt(xl, bias_q2o)
            f_q3e = filt(xl, bias_q3e)
            f_q3o = filt(xl, bias_q3o)
            f_q4e = filt(xl, bias_q4e)
            f_q4o = filt(xl, bias_q4o)
            for i in xl:
                data[i,cols_q1e] -= ftlgd(xl, f_q1e, i, d)
                data[i,cols_q1o] -= ftlgd(xl, f_q1o, i, d)
                data[i+rowlen,cols_q2e] -= ftlgd(xl, f_q2e, i, d)
                data[i+rowlen,cols_q2o] -= ftlgd(xl, f_q2o, i, d)
                data[i,cols_q3e] -= ftlgd(xl, f_q3e, i, d)
                data[i,cols_q3o] -= ftlgd(xl, f_q3o, i, d)
                data[i+rowlen,cols_q4e] -= ftlgd(xl, f_q4e, i, d)
                data[i+rowlen,cols_q4o] -= ftlgd(xl, f_q4o, i, d)
           # sys.exit(1)
#            pyplot.plot(xl, [a for a in xl], color='blue')
#            pyplot.plot(xl, [ftlgd(xl, xl, a, d) for a in xl], color='red')
#            print bias_q1e
#            print xl

            pyplot.plot(xl, f_q1e, color='blue')
            pyplot.plot(xl, [ftlgd(xl, f_q1e, a, d) for a in xl], color='red')
    #pyplot.step(bedge[:-1], [a + 1e-20 for a in  histbg], color='black')
    #pyplot.step(bedge[:-1], [a + 1e-20 for a in histreal], color='red')
  ##  pyplot.bar(bedge[:-1], fakehistn_l[0], edgecolor='green', width=0.4, log=True, fill=False)
    #pyplot.yscale('log')
    #pyplot.ylim(ymin=1e-1)
            pyplot.show()
        else:
            print "Error: Bias subtraction type not parsed correctly"
            sys.exit(1)

    ##########################################################################
    # Apply the gain correction  [not yet implemented]
    ##########################################################################

        if Gain:
            if R4K:
                if AT_MDM:
                    for r in range(r1,r2+1):
                        for c in cols_q1e:
                            data[r,c] -= r4k_gain_q1e
                        for c in cols_q1o:
                            data[r,c] -= r4k_gain_q1o
                        for c in cols_q2e:
                            data[r+rowlen,c] -= r4k_gain_q2e
                        for c in cols_q2o:
                            data[r+rowlen,c] -= r4k_gain_q2o
                        for c in cols_q2o:
                            data[r,c] -= r4k_gain_q3e
                        for c in cols_q2o:
                            data[r,c] -= r4k_gain_q3o
                        for c in cols_q2o:
                            data[r+rowlen,c] -= r4k_gain_q4e
                        for c in cols_q2o:
                            data[r+rowlen,c] -= r4k_gain_q4o
                else:
                    data[r1:r2,cols_q1e] /= r4k_gain_q1e
                    data[r1:r2,cols_q1o] /= r4k_gain_q1o
                    data[r3:r4,cols_q2e] /= r4k_gain_q2e
                    data[r3:r4,cols_q2o] /= r4k_gain_q2o
                    data[r1:r2,cols_q3e] /= r4k_gain_q3e
                    data[r1:r2,cols_q3o] /= r4k_gain_q3o
                    data[r3:r4,cols_q4e] /= r4k_gain_q4e
                    data[r3:r4,cols_q4o] /= r4k_gain_q4o
            else:
                if AT_MDM:
                    for r in range(r1,r2+1):
                        for c in cols_q1:
                            data[r,c] /= mdm4k_gain_q1
                        for c in cols_q2:
                            data[r+rowlen,c] /= mdm4k_gain_q2
                        for c in cols_q2:
                            data[r,c] /= mdm4k_gain_q3
                    for c in cols_q2:
                            data[r+rowlen,c] /= mdm4k_gain_q4
                else:
                    data[r1:r2,cols_q1] /= mdm4k_gain_q1
                    data[r3:r4,cols_q2] /= mdm4k_gain_q2
                    data[r1:r2,cols_q3] /= mdm4k_gain_q3
                    data[r3:r4,cols_q4] /= mdm4k_gain_q4


    ##########################################################################
    # Write the output file
    ##########################################################################

        fitsfile[0].data = data[r1:r4+1,c1:c4+1]
        OverscanKeyComment = 'Overscan by proc4k.py v%s (%s)' % (versNum, versDate)
        GainKeyValue = 'Relative'
        GainKeyComment = 'Gain removed by proc4k.py'
    #BiasKeyValue = '%s' % (versNum)
    #BiasKeyComment = 'Gain removed by proc4k.py'



        fitsfile[0].header.update('BIASPROC', OverscanKeyValue, OverscanKeyComment)       
        fitsfile[0].header['SECPIX'] = 0.273*ccdxbin
        outfile = fil[:string.find(fil, '.fits')]+suffix+'.fits'
        if os.path.isfile(outfile):
            print "  Warning: Overwriting pre-existing file %s" % (outfile)
            os.remove(outfile)
        fitsfile.writeto(outfile)
        fitsfile.close()

# print "%s Done" % (sys.argv[0])
print "%s Done" % (scriptname)

