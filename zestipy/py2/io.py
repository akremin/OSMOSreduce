# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:09:26 2015

@author: kremin
"""
import numpy as np
import os
from astropy.table import Table

def check_directories(dir,msknm=None):
    '''Ensure that the mask file and the other expected files exist
    at the specified path.'''
    if msknm:
        maskdir = os.path.join(dir,msknm)
    else:
        maskdir = dir
    #redestdir = os.path.join(maskdir,'red_ests')
    if not os.path.exists(maskdir):
        print 'Directory   %s  does not exist' % maskdir
        #raise(IOError)
        print "Making the directory"
        os.makedirs(maskdir)        
        
    #if not os.path.exists(redestdir):
    #    print 'You do not have a ./ %s / %s directory in cwd' % (msknm,'red_ests')
    #    print cwd
    #    print "Creating red_est folder for saving files"
    #    os.makedirs(os.path.join(redestdir))



def get_file_specs(pathtomask,skipprevgood=False,skipprevbad=False,usetests=False):
    '''Return a list of all spectra found in a particular path,
    with options of skipping certain spectra for which a redshift
    has already been determined.'''
    dir_list = os.listdir(pathtomask)
    sfiles = []
    gfiles = []
    gspecs = []
    sspecs = []
    for dfil in dir_list:
        check1 = os.path.splitext(dfil)[1]=='.txt'
        absfil = os.path.join(pathtomask,dfil)
        check2 = os.path.getsize(absfil) > 0
        check3 = dfil.find('final')<0
        if check1 and check2 and check3:
            if dfil.find('speczs')>=0:
                gfiles.append(dfil)
            elif dfil.find('skipped')>=0:
                sfiles.append(dfil)
    if skipprevgood and len(gfiles)>0:
        for gfil in gfiles:
            gpath = os.path.join(pathtomask,gfil)
            gcol = []
            this_was_a_test = False
            with open(gpath,'r') as gopen:
                for line in gopen:
                    current_frame = line.split("\t")[0].strip(' ')
                    if current_frame.isdigit():
                        gcol.append(current_frame)
                    elif len(current_frame)>3:
                        if current_frame[0:4] == 'This':
                            this_was_a_test = True
                        else:
                            pass
                    else:
                        pass

            if not this_was_a_test:
                for objnum in gcol:
                    gspecs.append(str(objnum))
            elif this_was_a_test and usetests:
                for objnum in gcol:
                    gspecs.append(str(objnum))
            else:
                pass
    if skipprevbad and len(sfiles)>0:
        for sfil in sfiles:
            spath = os.path.join(pathtomask,sfil)
            sopen = open(spath,'r')
            scol = []
            for line in sopen:
                scol.append(line.split("\t")[0])
            if len(scol)>0:
                scol = scol[1:]
                if scol[-1][0:4]=='This': 
                    if usetests:
                        scol = scol[:-1]
                    else:
                        continue
                for objnum in scol:
                    sspecs.append(str(objnum))
    return np.unique((gspecs+sspecs))


def get_objects_list(delz,quality_flag):
    '''Analyze outputed fits file'''
    quality_flag_table = Table.read("qualityflag_cut_table.fits",format='fits')
    #    qual_flag_inds = np.where( ( (quality_flag_table['Quality_Flag']<=quality_flag) & 
    #                                (quality_flag_table['Quality_Flag']>1) ) )
    #    qual_checked_data = quality_flag_table[qual_flag_inds]
    qual_checked_data = quality_flag_table[quality_flag_table['Quality_Flag']==quality_flag]
    object_locs = np.where( ( (qual_checked_data['Photo-z']<(qual_checked_data['Spec-z']+delz)) & 
                          (qual_checked_data['Photo-z']>(qual_checked_data['Spec-z']-delz)) ) )[0]
    full_obs_of_interest = qual_checked_data[object_locs]
    full_mask_ids = full_obs_of_interest['Mask_ID']
    mask_ids = np.unique(full_mask_ids)
    return mask_ids
