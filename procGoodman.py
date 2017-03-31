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
#import fitsio as pyfits
from astropy.io import fits as pyfits
import time
import pdb
from astropy.table import Table
# Version and Date

versNum = "1.2"
versDate = "2017-03-28"



######################################
#### Helper functions and classes ####
######################################
def get_median_bias(biasfilenames):
    if type(biasfilenames)==str:
        print('the median of 1 is itself')
        hdu = pyfits.open(biasfilenames)[0]
        return hdu.data[0],hdu.header
    elif len(biasfilenames)==1:
        print('the median of 1 is itself')
        hdu = pyfits.open(biasfilenames[0])[0]
        return hdu.data[0],hdu.header
    hdu = pyfits.open(biasfilenames[0])[0]
    curdata, exampleheader = hdu.data[0],hdu.header
    data = np.zeros((len(biasfilenames),curdata.shape[0],curdata.shape[1]))
    data[0,:,:] = curdata
    for i,name in enumerate(biasfilenames[1:]):
        curdata = pyfits.getdata(name)[0]
        data[i,:,:] = curdata
    return np.median(data,axis=0),exampleheader
    
    
class GoodmanFile:
    def __init__(self,filename='0000.filebase.fits'):
        pieces = filename.split('.')
        npieces = len(pieces)
        self.type = pieces[-1]
        self.imnum =  pieces[0]
        if npieces > 3:
            print("This doesn't follow the standard format. returning null for some info")
            self.mask = None
            self.target = None
            self.type = ''
        else:
            self.type, targetmask = pieces[1].split('_')
            self.target, self.mask = targetmask.split('-')
            
def getCleanFileList(datapath):
    lis = os.listdir(datapath)
    fileinfotable = Table(names=('filename','type','expnum','target','masknum'),dtype=('S128','S7','S4','S12','S3'))
    filename_translator = {'f': 'flat','b':'bias','s':'science','c':'comp','a':'comp'}
    for fil in lis:
        if fil.split('.')[-1]!='fits':
            continue
        passed = False
        rootname = fil[:-5]
        try:
            expnum, name = rootname.split('.')
        except ValueError:
            print(fil+" is a fits file that doesn't match format of expnum.filename.fits")
            continue
        try:
            possibletype, targetmask = name.split('_')
        except ValueError:
            possibletype = name
            targetmask = 'None-None'
        try:            
            target, mask = targetmask.split('-')
        except ValueError:
            target = 'None'
            mask = 'None'

        #pdb.set_trace()
        for filetype in ['science','bias','biases','flat','flats','comp','comparcs','arcs','comparc','arc','comps']:
            if possibletype.lower().find(filetype) > -1:
                standardizedtype = filename_translator[filetype[0]]
                passed = True
                break
        if not passed:
            print(fil+" is a fits file but didn't match any of the types comp,flat,bias,science\n")
        else:
            fileinfotable.add_row((os.path.join(datapath,fil),standardizedtype,expnum,target,mask))
    fileinfotable.convert_bytestring_to_unicode()
    return fileinfotable

    

def procGoodman(path_to_raw_data = './', basepath_to_save_data = './',overwrite = True): 
    if not os.path.exists(path_to_raw_data):
        print("That raw data directory doesn't exist\n\n")
        exit
    elif len(os.listdir(path_to_raw_data))==0:
        print("That raw data directory is empty\n\n")
        exit
                
    if os.path.exists(basepath_to_save_data):
        if overwrite:
            os.system('rm -rf '+basepath_to_save_data)
        else:
            os.rename(basepath_to_save_data,basepath_to_save_data+'_'+str(int(time.time()))+'.old')
        
    
    print("Making data products directory\n")
    os.makedirs(basepath_to_save_data)
        
    for direct in ['flat','science','comp']:
        print("Making "+direct+' directory\n')
        os.makedirs(os.path.join(basepath_to_save_data,direct))
    
            
    fileinfotable = getCleanFileList(path_to_raw_data)        
    #pdb.set_trace()
    
    biasfiletable = fileinfotable[fileinfotable['type']=='bias']
    nonbiasfileinfotable = fileinfotable[fileinfotable['type']!='bias']
    #if not overwrite and ('masterbias.fits' in biasfiles):
    #    os.path.rename(os.path.join(biaspath,'masterbias.fits'),os.path.join(biaspath,'masterbias.'+str(time.time())+'.fits.old'))
    
    
    #pdb.set_trace()
    masterbias,biasheader = get_median_bias(biasfiletable['filename'])
    biaslist = ''
    for name in biasfiletable['expnum']:
        biaslist += name + ','
    biasheader.add_history("Median 'master' bias for bias files: "+biaslist[:-1])
    pyfits.writeto(os.path.join(basepath_to_save_data,'masterbias.fits'),masterbias,header=biasheader)
    
    blank_stringcolumn = np.ones(len(nonbiasfileinfotable)).astype(str)
    c1 = Table.Column(data=blank_stringcolumn,name='BiasSubdLoc',dtype='S100')
    c2 = Table.Column(data=blank_stringcolumn,name='date',dtype='S10')
    c3 = Table.Column(data=blank_stringcolumn,name='time',dtype='S28')
    c4 = Table.Column(data=blank_stringcolumn,name='exptime',dtype='S4')
    c5 = Table.Column(data=blank_stringcolumn,name='binning',dtype='S3')
    c6 = Table.Column(data=blank_stringcolumn,name='filter',dtype='S10')
    c7 = Table.Column(data=blank_stringcolumn,name='grating',dtype='S10')
    c8 = Table.Column(data=blank_stringcolumn,name='target_RA',dtype='S12')
    c9 = Table.Column(data=blank_stringcolumn,name='target_Dec',dtype='S13')
    c10 = Table.Column(data=blank_stringcolumn,name='fullname',dtype='S20')
    c11 = Table.Column(data=blank_stringcolumn,name='maskname',dtype='S20')
    nonbiasfileinfotable.add_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11])
    
    
    nonbiasfileinfotable.convert_bytestring_to_unicode()
    # Do the bias subtraction
    for row in nonbiasfileinfotable:
        hdulist = pyfits.open(row['filename'],mode='readonly')
        currentdata = hdulist[0].data[0]
        currentheader = hdulist[0].header
        if row['type'] != currentheader['OBSTYPE'].lower():
            if row['type']=='science' and currentheader['OBSTYPE']=='OBJECT':
                pass
            else:
                print("File: "+row['filename']+"\nWas thought to be: "+row['type']+"\nBut the header claims it is: "+currentheader['OBSTYPE'].lower())
        assert masterbias.shape == currentdata.shape, "Make sure that the shapes of arrays are consistent"
        try:
            bias_subd = currentdata - masterbias
        except:
            pdb.set_trace()
        justname = os.path.basename(row['filename'])
        justname = justname.replace('.fits','_b.fits')
        currentheader.add_history("Bias Subtracted on "+time.ctime()+"    by procGoodman.py v"+versNum)
        print("Making new file: "+justname+'\n')
        pyfits.writeto(os.path.join(basepath_to_save_data,row['type'],justname),bias_subd,header=currentheader)
        row['date'],row['time'] = currentheader['DATE'],currentheader['TIME']
        row['exptime'] = currentheader['EXPTIME']
        row['binning'] = str(currentheader['PARAM18'])+'x'+str(currentheader['PARAM22'])
        row['filter'],row['grating'] = currentheader['FILTER2'],currentheader['GRATING']
        row['target_RA'],row['target_Dec'] = currentheader['OBSRA'],currentheader['OBSDEC']
        row['fullname'] = currentheader['OBJECT']
        row['maskname'] = currentheader['SLIT']
        row['BiasSubdLoc'] = os.path.join(basepath_to_save_data,row['type'],justname)
        hdulist.close()

    nonbiasfileinfotable.write(os.path.join(basepath_to_save_data,'fileinfo.csv'),format='ascii.csv')
    nonbiasfileinfotable.write(os.path.join(basepath_to_save_data,'fileinfo.fits'),format='fits')

if __name__ == '__main__':
    path_to_raw_data = '/u/home/kremin/value_storage/Data/goodman_jan17/Kremin10/data'
    #'/u/home/kremin/value_storage/Github/M2FSreduce/example_goodman/'
    basepath_to_save_data = '/u/home/kremin/value_storage/Data/goodman_jan17/Kremin10/data_products'
    overwrite = True
    
    
