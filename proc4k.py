#! /usr/bin/env python
#
# Paul Martini (OSU)
#
#  proc4k.py files
#
# Perform the overscan subtraction and remove the relative gain
# differences for a single R4K image or a list of R4K images.
# Also works for MDM4K.
#
# Steps:
#  1. determine if input is a file or list of files
#  2. identify binning, size of overscan
#  3. remove overscan and trim
#  4. remove relative gain variations (TBD)
#
#   8 Sep 2011: initial version for just bias subtraction
#  12 Sep 2011: tested, adapted to run on MDM computers, work on MDM4K data
#  16 Sep 2011: added glob module
#   8 Feb 2012: fixed error in even/odd column definitions, added more
# 		tracking and debugging information
#
#-----------------------------------------------------------------------------
#from __future__ import division
import string as str
import os
from sys import argv, exit
import numpy as np
from astropy.io import fits as pyfits
import glob
from matplotlib import pyplot

# Version and Date

versNum = "1.1.0"
versDate = "2012-02-08"

############################
#### Define various routines
############################

scriptname = argv[0][str.rfind(argv[0], "/") + 1::]


def usage():
    print("\nUsage for %s v%s (%s):" % (scriptname, versNum, versDate))
    print("	%s file.fits [or file*.fits or file1.fits file2.fits\
    ]" % (scriptname))
    print("\nWhere: file.fits, file*.fits, etc. are fits files\n")


def parseinput():
    flags = []
    files = []
    # check for any command-line arguments and input files
    for i in range(1, len(argv)):
        if str.find(argv[i], "-") == 0:
            flags.append(argv[i].strip("-"))
        else:
            files.append(argv[i])
    # check that the input files exist
    for i in range(1, len(files)):
        if os.path.isfile(files[i]) == 0:
            print("\n** ERROR: " + files[i] + " does not exist.")
            exit(1)
    return files, flags


def filt(x, l):
    y = [0] * len(x)
    c = 0.6745

    for a in range(0, len(x)):
        y[a] = l[x[a]]
    m = np.median(y)
    print(m)
    dv = [elm - m for elm in y]
    mad = np.median(np.fabs(dv) / c)  # Median-Asbolute-Deviation
#    print m + mad / 2
    for b in range(0, len(y)):
        if y[b] > m + 20 * mad / 2 or y[b] < m - 20 * mad / 2:
            print("reject: %d " % b)
            y[b] = m
    return y


############################
#### Script starts here ####
############################


#BiasType = BiasRow
BiasType = BiasSingle


for file in files:
    if os.path.isfile(file):
        fitsfile = pyfits.open(file)
        naxis1 = fitsfile[0].header['NAXIS1']
        naxis2 = fitsfile[0].header['NAXIS2']
        overscanx = fitsfile[0].header['OVERSCNX']
        overscany = fitsfile[0].header['OVERSCNY']    # should be 0
        ccdxbin = fitsfile[0].header['CCDXBIN']
        ccdybin = fitsfile[0].header['CCDYBIN']
        detector = fitsfile[0].header['DETECTOR']
        telescope = fitsfile[0].header['TELESCOP']
        overscanx /= ccdxbin
        overscany /= ccdybin

        #print file, naxis1, naxis2, overscanx, overscany, detector
        print("Processing %s[%d:%d] OVERSCANX=%d OVERSCANY=%d from %s \
        obtained at the %s" \
        % (file, naxis1, naxis2, overscanx, overscany, detector, telescope))
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

    if Debug:
        print("Quadrants in IRAF pixels: ")
        print(" q1: [%d : %d, %d : %d] " % (c1 + 1, c2 + 1, r1 + 1, r2 + 1))
        print(" q2: [%d : %d, %d : %d] " % (c1 + 1, c2 + 1, r3 + 1, r4 + 1))
        print(" q3: [%d : %d, %d : %d] " % (c3 + 1, c4 + 1, r1 + 1, r2 + 1))
        print(" q4: [%d : %d, %d : %d] " % (c3 + 1, c4 + 1, r3 + 1, r4 + 1))

    ## Calculate the bias level for each amplifier
    data = fitsfile[0].data
    # identify the columns to use to calculate the bias level
    # skip the first and last columns of the overscan
    # changed to 'list' for hiltner due to primitive python version
    starti = 4 / ccdxbin

    cols_over_q1 = np.arange(starti, overscanx-2, 1)
    cols_over_q2 = cols_over_q1
    cols_over_q3 = np.arange(naxis1-overscanx+starti, naxis1-2, 1)
    cols_over_q4 = cols_over_q3
    cols_q1 = np.arange(c1,c2+1,1)
    cols_q2 = cols_q1
    cols_q3 = np.arange(c3,c4+1,1)
    cols_q4 = cols_q3

    bias_q1 = np.zeros(rowlen, dtype=float)
    bias_q2 = np.zeros(rowlen, dtype=float)
    bias_q3 = np.zeros(rowlen, dtype=float)
    bias_q4 = np.zeros(rowlen, dtype=float)
    # calculate 1-D bias arrays for each amplifier
    for i in range(r1, r2+1, 1):
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

        bq1 = np.median(bias_q1)
        bq2 = np.median(bias_q2)
        bq3 = np.median(bias_q3)
        bq4 = np.median(bias_q4)
        data[r1:r2+1,cols_q1] -= bq1
        data[r3:r4+1,cols_q2] -= bq2
        data[r1:r2+1,cols_q3] -= bq3
        data[r3:r4+1,cols_q4] -= bq4


##########################################################################
# Apply the gain correction  [not yet implemented]
##########################################################################

    if Gain:
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
    outfile = file[:str.find(file, '.fits')]+suffix+'.fits'
    if os.path.isfile(outfile):
        print("  Warning: Overwriting pre-existing file %s" % (outfile))
        os.remove(outfile)
    fitsfile.writeto(outfile)
    fitsfile.close()

# print "%s Done" % (argv[0])
print("%s Done" % (scriptname))

