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
  
          
def getCleanGoodmanFileList(datapath):
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


def getCleanM2FSFileList(datapath):
    lis = os.listdir(datapath)
    fileinfotable = Table(names=('filename','type','expnum','target','masknum'),dtype=('S128','S7','S4','S12','S3'))
    filetypes = ['bias','object','comp','flat'] #'dark'
    for fil in lis:
        if fil[-5:]!='fits':
            continue
        passed = False
        rootname = fil[:-5]
        head = pyfits.getheader(fil)
        if head['BINNING'] != '2x2':
            continue
        filetype = head['EXPTYPE'].lower()
        if filetype == 'dark':
            continue
        elif filetype not in filetypes:
            print(fil+" is a fits file but didn't match any of the types object, flat, bias, or dark\n")
            continue
        plate, platestp, obj = head['PLATE'], head['PLATESTP'], head['OBJECT'] 
        if 'kremin' not in plate.lower() and 'kremin' not in platestp.lower() and 'kremin' not in obj.lower():
            continue
        else:
            fileinfotable.add_row((os.path.join(datapath,fil),filetype,expnum,target,mask))
    fileinfotable.convert_bytestring_to_unicode()
    return fileinfotable
    
    

def assign_values_from_header(row,header,detector = 'goodman'):
    if detector.lower() == 'goodman':
        row['date'],row['time'] = header['DATE'],header['TIME']
        row['exptime'] = header['EXPTIME']
        row['binning'] = "{}x{}".format(header['PARAM18'],header['PARAM22'])
        row['filter'],row['grating'] = header['FILTER2'],header['GRATING']
        row['target_RA'],row['target_Dec'] = header['OBSRA'],header['OBSDEC']
        row['fullname'] = header['OBJECT']
        row['maskname'] = header['SLIT']
    elif detector.lower() == 'M2FS':
        row['date'],row['time'] = header['UT-DATE'],header['UT-TIME']
        row['exptime'] = header['EXPTIME']
        row['binning'] = str(header['BINNING'])
        row['filter'],row['grating'] = header['FILTER'],"{}_{}".format(header['SHOE']+header['SLIDE'])
        row['target_RA'],row['target_Dec'] = header['RA-D'],header['DEC-D']
        row['fullname'] = "{}_{}_{}".format(header['OBJECT']+header['PLATE']+header['CONFIGFL'])
        row['maskname'] = header['PLATESTP']
    return row
    
    
def procGoodman(path_to_raw_data = './', basepath_to_save_data = './',overwrite = True, detector = 'goodman'): 
    # Make sure everything exists
    if not os.path.exists(path_to_raw_data):
        print("That raw data directory doesn't exist\n\n")
        exit
    elif len(os.listdir(path_to_raw_data))==0:
        print("That raw data directory is empty\n\n")
        exit
                
    if os.path.exists(basepath_to_save_data):
        if overwrite:
            remove_com = 'rm -rf {}'.format(basepath_to_save_data)
            ans = str(input("Are you sure you want to execute {}".format(remove_com)))
            if ans.lower() == 'y':
                os.system(remove_com)
        else:
            os.rename(basepath_to_save_data,basepath_to_save_data+'_'+str(int(time.time()))+'.old')
        
    print("Making data products directory\n")
    os.makedirs(basepath_to_save_data)
        
    for directory in ['flat','science','comp']:
        print("Making {} directory\n".format(directory))
        os.makedirs(os.path.join(basepath_to_save_data,directory))
    
    if detector == 'goodman':
        fileinfotable = getCleanGoodmanFileList(path_to_raw_data)        
    elif detector == 'm2fs':
        fileinfotable = getCleanM2FSFileList(path_to_raw_data) 
    
    biasfiletable = fileinfotable[fileinfotable['type']=='bias']
    nonbiasfileinfotable = fileinfotable[fileinfotable['type']!='bias']

    # Determine the master bias
    masterbias,biasheader = get_median_bias(biasfiletable['filename'])
    
    # Create an output string that containts all of the bias filenames
    biaslist = ''
    for name in biasfiletable['expnum']:
        biaslist += name + ','
    biasheader.add_history("Median 'master' bias for bias files: "+biaslist[:-1])
    
    # Save the masterbias
    pyfits.writeto(os.path.join(basepath_to_save_data,'masterbias.fits'),masterbias,header=biasheader)
    
    # Include additional columns
    blnk_col = np.ones(len(nonbiasfileinfotable)).astype(str)
    c1 =  Table.Column(data=blnk_col,        name='BiasSubdLoc', dtype='S100')
    c2 =  Table.Column(data=blnk_col.copy(), name='date',        dtype='S10')
    c3 =  Table.Column(data=blnk_col.copy(), name='time',        dtype='S28')
    c4 =  Table.Column(data=blnk_col.copy(), name='exptime',     dtype='S4')
    c5 =  Table.Column(data=blnk_col.copy(), name='binning',     dtype='S3')
    c6 =  Table.Column(data=blnk_col.copy(), name='filter',      dtype='S10')
    c7 =  Table.Column(data=blnk_col.copy(), name='grating',     dtype='S10')
    c8 =  Table.Column(data=blnk_col.copy(), name='target_RA',   dtype='S12')
    c9 =  Table.Column(data=blnk_col.copy(), name='target_Dec',  dtype='S13')
    c10 = Table.Column(data=blnk_col.copy(), name='fullname',    dtype='S20')
    c11 = Table.Column(data=blnk_col.copy(), name='maskname',    dtype='S20')
    nonbiasfileinfotable.add_columns([c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11])
    
    # Magic to make sure strings look correct
    nonbiasfileinfotable.convert_bytestring_to_unicode()
    
    # Do the bias subtraction
    for row in nonbiasfileinfotable:
        # Get the Data and the Header
        hdulist = pyfits.open(row['filename'],mode='readonly')
        currentdata = hdulist[0].data[0]
        currentheader = hdulist[0].header
        hdulist.close()
        # If it isn't what we expected, make sure it's a type of interest
        if row['type'] != currentheader['OBSTYPE'].lower():
            if row['type']=='science' and currentheader['OBSTYPE']=='OBJECT':
                pass
            else:
                print("File: "+row['filename']+"\nWas thought to be: "+row['type']+"\nBut the header claims it is: "+currentheader['OBSTYPE'].lower())

        # Make sure that the shapes of arrays are consistent
        assert masterbias.shape == currentdata.shape, "Masterbias and the current data aren't the same shape {}".format(row['filename'])

        # Perform the bias subtraction
        try:
            bias_subd = currentdata - masterbias
        except:
            pdb.set_trace()
            
        # Name the file with a  *_b*
        justname = os.path.basename(row['filename'])
        justname = justname.replace('.fits','_b.fits')
        
        # Save to header for prosperity
        currentheader.add_history("Bias Subtracted on "+time.ctime()+"    by proc_ccd.py v"+versNum)
        print("Making new file: "+justname+'\n')
        
        # Write file
        pyfits.writeto(os.path.join(basepath_to_save_data,row['type'],justname),bias_subd,header=currentheader)

        # Put some other parameters from the header into the documentation file
        row = assign_values_from_header(row,currentheader,detector)
        row['BiasSubdLoc'] = os.path.join(basepath_to_save_data,row['type'],justname)
        
    # Write the summary table of file information to both csv and fits formats
    nonbiasfileinfotable.write(os.path.join(basepath_to_save_data,'fileinfo.csv'),format='ascii.csv')
    nonbiasfileinfotable.write(os.path.join(basepath_to_save_data,'fileinfo.fits'),format='fits')




if __name__ == '__main__':
    path_to_raw_data = '/u/home/kremin/value_storage/Data/goodman_jan17/Kremin10/data'
    #'/u/home/kremin/value_storage/Github/M2FSreduce/example_goodman/'
    basepath_to_save_data = '/u/home/kremin/value_storage/Data/goodman_jan17/Kremin10/data_products'
    overwrite = True
    
    
