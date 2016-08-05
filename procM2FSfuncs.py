# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 13:30:08 2016

@author: kremin
"""
import os
import sys
import numpy as np



def usage(scriptname, versNum, versDate):
    print "\nUsage for %s v%s (%s):" % (scriptname, versNum, versDate)
    print "	%s file.fits [or file*.fits or file1.fits file2.fits]" % (scriptname)
    print "\nWhere: file.fits, file*.fits, etc. are fits files\n"


def parseinput():
    flags = []
    files = []
    # check for any command-line arguments and input files
    for i in range(1, len(sys.argv)):
        if "-" in sys.argv[i]:
            flags.append(sys.argv[i].strip("-"))
        else:
            files.append(sys.argv[i])
    # check that the input files exist
    for i in range(1, len(files)):
        if os.path.isfile(files[i]) == 0:
            print "\n** ERROR: " + files[i] + " does not exist."
            sys.exit(1)
    return files, flags


def filt(x, l):
    y = [0] * len(x)
    c = 0.6745

    for a in range(0, len(x)):
        y[a] = l[x[a]]
    m = np.median(y)
    print m
    dv = [elm - m for elm in y]
    mad = np.median(np.fabs(dv) / c)  # Median-Asbolute-Deviation
#    print m + mad / 2
    for b in range(0, len(y)):
        if y[b] > m + 20 * mad / 2 or y[b] < m - 20 * mad / 2:
            print "reject: %d " % b
            y[b] = m
    return y


def ftlgd(x, l, i, d):
    coe = np.polynomial.legendre.legfit(x, l, d)
    return np.polynomial.legendre.legval(i, coe)