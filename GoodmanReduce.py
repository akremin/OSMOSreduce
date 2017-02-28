import numpy as np
from astropy.io import fits as pyfits
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.signal as signal
from scipy import fftpack
from scipy.optimize import curve_fit
import sys
import subprocess
import pickle
import pdb
from pyds9 import DS9
import copy
import os
import time

import re
import pandas as pd
import fnmatch

from get_photoz import query_galaxies
from slit_find import slit_find
from zestipy.data_structures import waveform, redshift_data, smooth_waveform
from testopt import interactive_plot, LineBrowser, polyfour
from sncalc import sncalc
from calibrations import load_calibration_lines
from scipy.signal import argrelextrema
from zestipy.z_est import z_est
from zestipy.plotting_tools import summary_plot

#from gal_trace import gal_trace
#import zestipy.io as zpio
#import zestipy.workflow_funcs as zwf


'''
IMPORTANT NOTES:
In the .oms file, the first and last RA/DEC represent a reference slit at the 
bottom of the mask and the center of the mask respectively.

Please list the calibration lamp(s) used during your observations here
'''
cal_lamp = ['HgNe','Argon','Neon']  #['Xenon','Argon'] #'Xenon','Argon','HgNe','Neon'

pixscale = 0.15 #arcsec/pixel  #pixel scale at for Goodman
wavelength_dir = 1   # Goodman has wavelengths increasing as pixel val increases, OSMOS is reversed #-1
#xbin = 1
#ybin = 1
xbin = 2
ybin = 2
xshift = 0#0.0/xbin    # with division this is in binned pixels
yshift = 6#10#740.0/ybin  # with division this is in binned pixels
binnedx = 2070#2071   # this is in binned pixels
binnedy = 1256#1257    # this is in binned pixels
binxpix_mid = int(binnedx/2)
binypix_mid = int(binnedy/2)
n_emptypixs = 5 # should be odd
instrument = "GOODMAN"


#datadir = '/u/home/kremin/value_storage/goodman_jan17/'
datadir = '/home/kremin/SOAR_data/'

# From goodman file header:
#PARAM16 =                    0 / Serial Origin,Pixels                           
#PARAM17 =                 2071 / Serial Length,Binned Pixels                    
#PARAM18 =                    2 / Serial Binning,Pixels                          
#PARAM19 =                    0 / Serial Post Scan,Pixels                        
#PARAM20 =                  740 / Parallel Origin,Pixels                         
#PARAM21 =                 1257 / Parallel Length,Binned Pixels                  
#PARAM22 =                    2 / Parallel Binning,Pixels                        
#PARAM23 =                    0 / Parallel Post Scan,Pixels         
#RDNOISE =             3.690000 / CCD readnoise [e-]                             
#GAIN    =             0.560000 / CCD gain [e-/ADU]    




##############################################################
#########    Begin the Start of the Code    ##################
##############################################################

def getch():
    import tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old_settings)
    return ch

def filter_image(img):
    img_sm = signal.medfilt(img,5)
    sigma = 2.0 
    bad = np.abs(img-img_sm) / sigma > 8.0
    img_cr = img.copy()
    img_cr[bad] = img_sm[bad]
    return img_cr



wm, fm, cal_states = load_calibration_lines(cal_lamp)


###################
#Define Cluster ID#
###################
try:
    id_import = str(sys.argv[1])
    clus_id = id_import
except:
    print("Cluster Name Error: You must enter a cluster name to perform reduction")
    print(' ')
    idnew = str((raw_input("Cluster ID: ")))
    clus_id = idnew

print(('Reducing cluster: ',clus_id))
###############################################################
#hack
with open(datadir+clus_id+'/GOODMAN_selected_comparison_lines-Kremin10.pkl','rb') as linestouse:
    linedict = pickle.load(linestouse)
    wm = np.asarray(linedict['lines'])
    fm = np.asarray(linedict['line_mags'])
    del linedict
#ask if you want to only reduce sdss galaxies with spectra
try:
    sdss_check = str(sys.argv[2])
    if sdss_check == 'sdss':
        sdss_check = True
    else:
        raise Exception(sdss_check+' is not an accepted input. \'sdss\' is the only accepted input here.')
except IndexError:
    sdss_check = False
sdss_check = False #hack
############################
#Import Cluster .fits files#
############################
for curfile in os.listdir(datadir+clus_id+'/maskfiles/'): #search and import all mosaics
    if fnmatch.fnmatch(curfile, 'mosaic_*'):
        image_file = curfile



#import, clean, and add science fits files
sciencefiles = np.array([])
hdulists_science = np.array([])
for curfile in os.listdir(datadir+clus_id+'/data_products/science/'): #search and import all science filenames
    if fnmatch.fnmatch(curfile, '*b.fits'):
        sciencefiles = np.append(sciencefiles,curfile)
        scifits = pyfits.open(datadir+clus_id+'/data_products/science/'+curfile)
        hdulists_science = np.append(hdulists_science,scifits)
science_file = sciencefiles[0]
hdulist_science = pyfits.open(datadir+clus_id+'/data_products/science/'+science_file)
naxis1 = hdulist_science[0].header['NAXIS2']
naxis2 = hdulist_science[0].header['NAXIS1']

#import flat data
flatfiles = np.array([])
hdulists_flat = np.array([])
for curfile in os.listdir(datadir+clus_id+'/data_products/flat/'): #search and import all science filenames
    if fnmatch.fnmatch(curfile, '*b.fits'):
        flatfiles = np.append(flatfiles,curfile)
        flatfits = pyfits.open(datadir+clus_id+'/data_products/flat/'+curfile)
        hdulists_flat = np.append(hdulists_flat,flatfits)
if len(hdulists_flat) < 1:
    raise Exception('proc4k.py did not detect any flat files')

#import arc data
arcfiles = np.array([])
hdulists_arc = np.array([])
for curfile in os.listdir(datadir+clus_id+'/data_products/comp/'): #search and import all science curfilenames
    if fnmatch.fnmatch(curfile, '*b.fits'):
        arcfiles = np.append(arcfiles,curfile)
        arcfits = pyfits.open(datadir+clus_id+'/data_products/comp/'+curfile)
        hdulists_arc = np.append(hdulists_arc,arcfits)
if len(hdulists_arc) < 1:
    raise Exception('proc4k.py did not detect any arc files')
###############################################################

#########################################################
#Need to parse .txt file for slit information#
###########################################append
RA = []; DEC = []; TYPE = []; slit_NUM = []
slit_X = []; slit_Y = []; slit_WIDTH = []; slit_LENGTH = []    
if instrument.upper() == 'OSMOS':      
    for curfile in os.listdir(datadir+clus_id+'/maskfiles/'):
        if fnmatch.fnmatch(curfile, '*.oms'):
            omsfile = curfile
    inputfile = open(clus_id+'/'+omsfile)
    alltext = inputfile.readlines()
    for line in alltext:
        RAmatch = re.search('TARG(.*)\.ALPHA\s*(..)(..)(.*)',line)
        DECmatch = re.search('DELTA\s*(...)(..)(.*)',line)
        WIDmatch = re.search('WID\s\s*(.*)',line)
        LENmatch = re.search('LEN\s\s*(.*)',line)
        Xmatch = re.search('XMM\s\s*(.*)',line)
        Ymatch = re.search('YMM\s\s*(.*)',line)
        if RAmatch:
            slit_NUM.append(RAmatch.group(1))
            RA.append(RAmatch.group(2)+':'+RAmatch.group(3)+':'+RAmatch.group(4))
        if DECmatch:
            DEC.append(DECmatch.group(1)+':'+DECmatch.group(2)+':'+DECmatch.group(3))
        if WIDmatch:
            slit_WIDTH.append(WIDmatch.group(1))
        if LENmatch:
            slit_LENGTH.append(LENmatch.group(1))
        if Xmatch:
            slit_X.append(0.5*naxis1+np.float(Xmatch.group(1))*(11.528)/(pixscale))
        if Ymatch:
            slit_Y.append(0.5*naxis2+np.float(Ymatch.group(1))*(11.528)/(pixscale)+yshift)
    TYPE = ['']*len(RA)
    SLIT_X = np.asarray(slit_X)
    SLIT_Y = np.asarray(slit_Y)
    SLIT_WIDTH = np.asarray(slit_WIDTH)
    SLIT_LENGTH = np.asarray(slit_LENGTH)
elif instrument.upper() == 'GOODMAN':
    with open(datadir+clus_id+'/maskfiles/'+clus_id+'_Mask1.txt','r') as fil:
        throw_y = []
        throw_x = []
        for line in fil:
            if line[:2] == 'M7':
                if len(throw_y)>0:
                    slit_WIDTH.append(np.max(throw_y)-np.min(throw_y))
                    slit_LENGTH.append(np.max(throw_x)-np.min(throw_x))
                    throw_y = []
                    throw_x = []
                else:
                    slit_WIDTH.append(None)
                    slit_LENGTH.append(None)
            if line[:3] == 'G0 ':
                vals = line.split(' ')
                slit_X.append(float(vals[1][1:].strip('\n\r')))
                slit_Y.append(float(vals[2][1:].strip('\n\r')))
            if line[:3]=='G1 ':
                vals = line.split(' ')
                throw_x.append(float(vals[2][1:].strip('\n\r')))
                throw_y.append(float(vals[3][1:].strip('\n\r')))
    #########################################################
    #Need to parse .msk file for ra,dec information#
    #########################################################
    with open(datadir+clus_id+'/maskfiles/'+clus_id+'_Mask1.msk','r') as fil:
        for line in fil:
            if line[:3] == 'MM_':
                mm_per_asec = float(line.split('=')[1].strip('\n\r'))
            if line[:4] == '[Obj':
                slit_NUM.append(int(line[7:].strip('\n\r[]')))
            if line[:3]=='DEC':
                DEC.append(line.split('=')[1].strip('\n\r'))
            if line[:3]=='RA=':
                RA.append(line.split('=')[1].strip('\n\r'))
            if line[:3]=='IS_':
                tf = line.split('=')[1].strip('\n\r')
                if tf == 'False':
                    TYPE.append('Target')
                else:
                    TYPE.append('Alignment')
    mm_per_asec = 0.32 #hack
    
SLIT_NUM = np.asarray(slit_NUM)
RA = np.asarray(RA)
DEC = np.asarray(DEC)
TYPE = np.asarray(TYPE)

#hack
if instrument.upper() == 'GOODMAN':
    correct_order_idx = np.argsort(SLIT_NUM)
    SLIT_NUM = SLIT_NUM[correct_order_idx]
    RA = RA[correct_order_idx]
    DEC = DEC[correct_order_idx]
    TYPE = TYPE[correct_order_idx]
    ## major hack 
    reorder = np.asarray([0,1,2,3,4,5,6,9,10,7,8,17,12,13,14,15,16,11])  
    RA = RA[reorder]
    DEC = DEC[reorder]
    TYPE = TYPE[reorder]
  
    # All widths and locs are currently in mm's  -> want pixels
    # X,Y in mm for mask is Y,-X for pixels on ccd  
    #ie axes are flipped and one is inverted
    SLIT_X = binxpix_mid - np.array(slit_Y[1:-1])*(1/(xbin*pixscale*mm_per_asec))
    SLIT_Y = binnedy + np.array(slit_X[1:-1])*(1/(ybin*pixscale*mm_per_asec)) - yshift
    SLIT_WIDTH = np.array(slit_WIDTH[1:])*(1/(xbin*pixscale*mm_per_asec))
    SLIT_LENGTH = np.array(slit_LENGTH[1:])*(1/(ybin*pixscale*mm_per_asec))


#remove throw away rows and dump into Gal_dat dataframe
Gal_dat = pd.DataFrame({'RA':RA,'DEC':DEC,'SLIT_WIDTH':SLIT_WIDTH,'SLIT_LENGTH':SLIT_LENGTH,'SLIT_X':SLIT_X,'SLIT_Y':SLIT_Y,'TYPE':TYPE,'NAME':SLIT_NUM})

###############################################################

############################
#Query SDSS for galaxy data#
############################
if os.path.isfile(datadir+clus_id+'/'+clus_id+'_sdssinfo.csv'):
    redshift_dat = pd.read_csv(datadir+clus_id+'/'+clus_id+'_sdssinfo.csv')
else:
    #returns a Pandas dataframe with columns
    #objID','SpecObjID','ra','dec','umag','gmag','rmag','imag','zmag','redshift','photo_z','extra'
    redshift_dat = query_galaxies(Gal_dat.RA,Gal_dat.DEC)
    redshift_dat.to_csv(datadir+clus_id+'/data_products/'+clus_id+'_sdssinfo.csv',index=False)


#merge into Gal_dat
Gal_dat = Gal_dat.join(redshift_dat)

gal_z = Gal_dat['spec_z']
gal_gmag = Gal_dat['gmag']
gal_rmag = Gal_dat['rmag']
gal_imag = Gal_dat['imag']

####################
#Open images in ds9#
####################
p = subprocess.Popen('ds9 '+datadir+clus_id+'/maskfiles/'+image_file+' -geometry 1200x900 -scale sqrt -scale mode zscale -fits '+datadir+clus_id+'/data_products/comp/'+arcfiles[0],shell=True)
#p = subprocess.Popen('ds9 '+datadir+clus_id+'/data_products/'+'/'+image_file+' -geometry 1200x900 -scale sqrt -scale mode zscale -fits '+clus_id+'/data_products/comp/'+arcfiles[0],shell=True)
time.sleep(2)
print("Have the images loaded? (y/n)")
while True: #check to see if images have loaded correctly
    char = getch()
    if char.lower() in ("y", "n"):
        if char.lower() == "y":
            print('Image has been loaded')
            break
        else:
            sys.exit('Check to make sure file '+image_file+' exists in '+datadir+clus_id+'/maskfiles/')

d = DS9() #start pyds9 and set parameters
d.set('frame 1')
d.set('single')
d.set('zscale contrast 9.04')
d.set('zscale bias 0.055')
d.set('zoom 2')
d.set('cmap Heat')
d.set('regions sky fk5')
#################################################################


####################################################################################
#Loop through mosaic image and decide if objects are galaxies, stars, sky, or other#
####################################################################################
#hack
skip_assign = 'y'
keys = np.arange(0,Gal_dat.SLIT_WIDTH.size,1).astype('string')
#if os.path.isfile(datadir+clus_id+'/'+clus_id+'_slittypes.pkl'):
#    skip_assign = (raw_input('Detected slit types file in path. Do you wish to use this (y) or remove and re-assign slit types (n)? '))
#skip_assign = 'y'
if skip_assign == 'n':
    slit_type = {}
    print('Is this a galaxy (g), a reference star (r), or empty sky (s)?')
    for i in range(len(Gal_dat)):
        d.set('pan to '+Gal_dat.RA[i]+' '+Gal_dat.DEC[i]+' wcs fk5')
        if Gal_dat.SLIT_WIDTH[i] == '1.0':
            d.set('regions command {box('+Gal_dat.RA[i]+' '+Gal_dat.DEC[i]+' 3 24) #color=green}')
        else:
            d.set('regions command {box('+Gal_dat.RA[i]+' '+Gal_dat.DEC[i]+' 12 12) #color=green}')
        while True:
            char = getch()
            if char.lower() in ("g", "r", "s"):
                break
        if char.lower() == 'g' and Gal_dat['TYPE'][i] == 'Target':
            print('Mask file confirms that was a targeted galaxy\n')
        elif char.lower() == 'r' and Gal_dat['TYPE'][i] == 'Alignment':
            print('Mask file confirms that was an alignment object\n')
        else:
            print('Mask file disagrees\n')
        slit_type[keys[i]] = char.lower()
    pickle.dump(slit_type,open(datadir+clus_id+'/'+clus_id+'_slittypes.pkl','wb'))
else:
    slit_type = pickle.load(open(datadir+clus_id+'/'+clus_id+'_slittypes.pkl','rb'))

stypes = pd.DataFrame(list(slit_type.values()),index=np.array(list(slit_type.keys())).astype('int'),columns=['slit_type'])
Gal_dat = Gal_dat.join(stypes)
##################################################################


d.set('frame 2')
d.set('zscale contrast 0.25')
d.set('zoom 0.40')



#######################################
#Reduction steps to prep science image#
#######################################
#hack
skip_cr_remov = 'y'
#if os.path.isfile(datadir+clus_id+'/data_products/science/'+clus_id+'_science.cr.fits'):
#    skip_cr_remov = (raw_input('Detected cosmic ray filtered file exists. Do you wish to use this (y) or remove and re-calculate (n)? '))
if skip_cr_remov == 'n':
    try:
        os.remove(datadir+clus_id+'/data_products/science/'+clus_id+'_science.cr.fits')
    except: pass
    scifits_c = copy.copy(hdulists_science[0]) #copy I will use to hold the smoothed and added results
    scifits_c.data *= 0.0
    print('SCIENCE REDUCTION')
    for scifits in hdulists_science:
        filt = filter_image(scifits.data)
        scifits_c.data += filt + np.abs(np.nanmin(filt))
    scifits_c.writeto(datadir+clus_id+'/data_products/science/'+clus_id+'_science.cr.fits')
else: 
    scifits_c = pyfits.open(datadir+clus_id+'/data_products/science/'+clus_id+'_science.cr.fits')[0]
    print('loading pre-prepared cosmic ray filtered files...')

print('FLAT REDUCTION')
if skip_cr_remov == 'n':
    try:
        os.remove(datadir+clus_id+'/data_products/flat/'+clus_id+'_flat.cr.fits')
    except: pass
    flatfits_c = copy.copy(hdulists_flat[0]) #copy I will use to hold the smoothed and added results
    flat_data = np.zeros((hdulists_flat.size,naxis1,naxis2))
    i = 0
    for flatfits in hdulists_flat:
        filt = filter_image(flatfits.data)
        flat_data[i] = (filt+np.abs(np.nanmin(filt)))/np.max(filt+np.abs(np.nanmin(filt)))
        i += 1
    flatfits_c.data = np.median(flat_data,axis=0)
    flatfits_c.writeto(datadir+clus_id+'/data_products/flat/'+clus_id+'_flat.cr.fits')
else: flatfits_c = pyfits.open(datadir+clus_id+'/data_products/flat/'+clus_id+'_flat.cr.fits')[0]

print('ARC REDUCTION')
if skip_cr_remov == 'n':
    try:
        os.remove(datadir+clus_id+'/data_products/comp/'+clus_id+'_arc.cr.fits')
    except: pass
    arcfits_c = copy.copy(hdulists_arc[0]) #copy I will use to hold the smoothed and added results
    arcfits_c.data *= 0.0
    for arcfits in hdulists_arc:
        filt = arcfits.data#filter_image(arcfits.data)
        arcfits_c.data += filt + np.abs(np.nanmin(filt))
    arcfits_c.writeto(datadir+clus_id+'/data_products/comp/'+clus_id+'_arc.cr.fits')
else: arcfits_c = pyfits.open(datadir+clus_id+'/data_products/comp/'+clus_id+'_arc.cr.fits')[0]




##################################################################
#Loop through regions and shift regions for maximum effectiveness#
##################################################################
#hack
skip_slitpositioning = 'n'
#if os.path.isfile(datadir+clus_id+'/'+clus_id+'_slit_pos_qual.tab'):
#    skip_slitpositioning = (raw_input('Detected slit position and quality file in path. Do you wish to use this (y) or remove and re-adjust (n)? '))
if skip_slitpositioning == 'n':
    good_spectra = np.array(['n']*len(Gal_dat))
    FINAL_SLIT_X = np.zeros(len(Gal_dat))
    FINAL_SLIT_Y = np.zeros(len(Gal_dat))
    BOX_WIDTH = np.zeros(len(Gal_dat))
    spectra = {}
    print('If needed, move region box to desired location. To increase the size, drag on corners')
    for i in range(BOX_WIDTH.size):
        lower_lim = 0
        upper_lim = 2000
        print(('SLIT ',i,'   OBJECT ',Gal_dat.NAME[i],'    which is a ',Gal_dat.TYPE[i]))
        #d.set('pan to 1150.0 '+str(Gal_dat.SLIT_Y[i])+' physical')
        d.set('pan to 1150.0 '+str(Gal_dat.SLIT_Y[i])+' physical')
        print(('Galaxy at ',Gal_dat.RA[i],Gal_dat.DEC[i]))
        #d.set('regions command {box(2000 '+str(Gal_dat.SLIT_Y[i])+' 4500 85) #color=green highlite=1}')
        # box(x,y,width,height)
        d.set('regions command {box('+str(Gal_dat.SLIT_X[i])+' '+str(Gal_dat.SLIT_Y[i])+' '+str(binnedx)+' '+str(Gal_dat.SLIT_LENGTH[i]+4*n_emptypixs)+') #color=green highlite=1}')
        #raw_input('Once done: hit ENTER')
        if Gal_dat.slit_type[i] == 'g':
            if sdss_check:
                if Gal_dat.spec_z[i] != 0.0: skipgal = False
                else: skipgal = True
            else: skipgal = False
            if not skipgal:
                good = False
                loops = 1
                while not good and loops <=3:
                    good = True
                    print('Move/stretch region box. Hit (y) when ready')
                    while True:
                        char = getch()
                        if char.lower() in ("y"):
                            break
                    newpos_str = d.get('regions').split('\n')
                    for n_string in newpos_str:
                        if n_string[:3] == 'box':
                            newpos = re.search('box\(.*,(.*),.*,(.*),.*\)',n_string)
                            FINAL_SLIT_X[i] = Gal_dat.SLIT_X[i]
                            FINAL_SLIT_Y[i] = int(float(newpos.group(1)))
                            BOX_WIDTH[i] = int(float(newpos.group(2)))

                            ##
                            #Sky subtract code
                            ##
                            try:
                                lowerbound = int(FINAL_SLIT_Y[i]-(BOX_WIDTH[i]/2.0))
                                upperbound = int(FINAL_SLIT_Y[i]+(BOX_WIDTH[i]/2.0))
                                cutflatdat = flatfits_c.data[lowerbound:upperbound,:]
                                cutscidat = scifits_c.data[lowerbound:upperbound,:]
                                cutarcdat = arcfits_c.data[lowerbound:upperbound,:]
                                pdb.set_trace()
                                science_spec,arc_spec,gal_spec,gal_cuts,BOX_WIDTH[i] = slit_find(cutflatdat,cutscidat,cutarcdat,lower_lim,upper_lim,int(Gal_dat.SLIT_LENGTH[i]),n_emptypixs,int(Gal_dat.SLIT_Y[i]))
                                spectra[keys[i]] = {'science_spec':science_spec,'arc_spec':arc_spec,'gal_spec':gal_spec,'gal_cuts':gal_cuts}
                                print('Is this spectra good (y) or bad (n)?')
                                while True:
                                    char = getch()
                                    if char.lower() in ("y","n"):
                                        break
                                plt.close()
                                good_spectra[i] = char.lower()


                                break
                            except:
                                raise
                                print('Fit did not fall within the chosen box. Please re-define the area of interest.')
                                good = False
                    loops += 1
                if loops == 4:
                    good_spectra[i] = 'n'
                    FINAL_SLIT_X[i] = Gal_dat.SLIT_X[i]
                    FINAL_SLIT_Y[i] = Gal_dat.SLIT_Y[i]
                    BOX_WIDTH[i] = Gal_dat.SLIT_LENGTH[i]
            else:
                good_spectra[i] = 'n'
                FINAL_SLIT_X[i] = Gal_dat.SLIT_X[i]
                FINAL_SLIT_Y[i] = Gal_dat.SLIT_Y[i]
                BOX_WIDTH[i] = Gal_dat.SLIT_LENGTH[i]
        else:
            good_spectra[i] = 'n'
            FINAL_SLIT_X[i] = Gal_dat.SLIT_X[i]
            FINAL_SLIT_Y[i] = Gal_dat.SLIT_Y[i]
            BOX_WIDTH[i] = Gal_dat.SLIT_LENGTH[i]
        print((FINAL_SLIT_X[i],FINAL_SLIT_Y[i],BOX_WIDTH[i]))
        d.set('regions delete all')
    print(FINAL_SLIT_X)
    np.savetxt(datadir+clus_id+'/'+clus_id+'_slit_pos_qual.tab',np.array(list(zip(FINAL_SLIT_X,FINAL_SLIT_Y,BOX_WIDTH,good_spectra)),dtype=[('float',float),('float2',float),('int',int),('str','|S1')]),delimiter='\t',fmt='%10.2f %10.2f %3d %s')
    pickle.dump(spectra,open(datadir+clus_id+'/'+clus_id+'_reduced_spectra.pkl','wb'))
else:
    FINAL_SLIT_X,FINAL_SLIT_Y,BOX_WIDTH = np.loadtxt(datadir+clus_id+'/'+clus_id+'_slit_pos_qual.tab',dtype='float',usecols=(0,1,2),unpack=True)
    good_spectra = np.loadtxt(datadir+clus_id+'/'+clus_id+'_slit_pos_qual.tab',dtype='string',usecols=(3,),unpack=True)
    spectra = pickle.load(open(datadir+clus_id+'/'+clus_id+'_reduced_spectra.pkl','rb'))

Gal_dat['FINAL_SLIT_X'],Gal_dat['FINAL_SLIT_Y'],Gal_dat['SLIT_WIDTH'],Gal_dat['good_spectra'] = FINAL_SLIT_X,FINAL_SLIT_Y,BOX_WIDTH,good_spectra

#Need to flip FINAL_SLIT_X coords to account for reverse wavelength spectra
Gal_dat['FINAL_SLIT_X_FLIP'] = binnedx - Gal_dat.FINAL_SLIT_X# 
#Gal_dat['FINAL_SLIT_X_FLIP'] = 4064 - Gal_dat.FINAL_SLIT_X
####################################################################


########################
#Wavelength Calibration#
########################
#hack
skip_wavelengthcalib = 'y'
#wave = np.zeros((len(Gal_dat),4064))
#if os.path.isfile(datadir+clus_id+'/'+clus_id+'_stretchshift.tab'):
#    skip_wavelengthcalib = (raw_input('Detected file with stretch and shift parameters for each spectra. Do you wish to use this (y) or remove and re-adjust (n)? '))
if skip_wavelengthcalib == 'n':
    #create write file
    f = open(datadir+clus_id+'/'+clus_id+'_stretchshift.tab','w')
    f.write('#X_SLIT_FLIP     Y_SLIT     SHIFT     STRETCH     QUAD     CUBE     FOURTH    FIFTH    WIDTH \n')
    
    #initialize polynomial arrays
    fifth,fourth,cube,quad,stretch,shift =  np.zeros((6,len(Gal_dat)))
    shift_est = 5066.0+Gal_dat['FINAL_SLIT_X'][0]-Gal_dat['FINAL_SLIT_X']#4.71e-6*(Gal_dat['FINAL_SLIT_X'] - (binnedx/2)-200)**2 + 4.30e-6*(Gal_dat['FINAL_SLIT_Y'] - (binnedy/2))**2 + binnedx
    stretch_est = 1.997*np.ones(len(Gal_dat))#-9.75e-9*(Gal_dat['FINAL_SLIT_X'] - (binnedx/2)+100)**2 - 2.84e-9*(Gal_dat['FINAL_SLIT_Y'] - (binnedy/2))**2 + 0.7139
    quad_est = -0.000024*np.ones(len(Gal_dat))#8.43e-9*(Gal_dat['FINAL_SLIT_X'] - (binnedx/2)+100) + 1.55e-10*(Gal_dat['FINAL_SLIT_Y'] - (binnedy/2)) + 1.3403e-5
    cube_est = (7.76e-13*(Gal_dat['FINAL_SLIT_X'][0] - (binnedx/2)+100) + 4.23e-15*(Gal_dat['FINAL_SLIT_Y'][0] - (binnedy/2)) - 5.96e-9)*np.ones(len(Gal_dat))
    fifth_est,fourth_est = np.zeros((2,len(Gal_dat)))
    calib_data = arcfits_c.data
    p_x = np.arange(0,binnedx,1)
    ii = 0
    #4763.0*np.ones(len(Gal_dat))# 1.96*np.ones(len(Gal_dat))#
    #do reduction for initial galaxy
    while ii <= stretch.size:
        if good_spectra[ii]=='y':
            #pdb.set_trace()
            f_x = np.sum(spectra[keys[ii]]['arc_spec'],axis=0)[:binnedx]
            f_x = f_x[::wavelength_dir]
            if len(f_x) != len(p_x):
                pdb.set_trace()
            d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[ii])+' physical') 
            d.set('regions command {box('+str(Gal_dat.SLIT_X[ii])+' '+str(Gal_dat.FINAL_SLIT_Y[ii])+' '+str(binnedx)+' '+str(Gal_dat.SLIT_WIDTH[ii])+') #color=green highlite=1}')
            #pdb.set_trace()
            #pdb.set_trace()
            #initial stretch and shift
            stretch_est[ii],shift_est[ii],quad_est[ii] = interactive_plot(p_x,f_x,stretch_est[ii],shift_est[ii],quad_est[ii],cube_est[ii],fourth_est[ii],fifth_est[ii],Gal_dat.FINAL_SLIT_X_FLIP[ii],wm,fm,cal_states)
            if len(f_x) != len(p_x):
                pdb.set_trace()
            #run peak identifier and match lines to peaks
            line_matches = {'lines':[],'line_mags':[],'peaks_p':[],'peaks_w':[],'peaks_h':[]}
            est_features = [fifth_est[ii],fourth_est[ii],cube_est[ii],quad_est[ii],stretch_est[ii],shift_est[ii]]
            xspectra = fifth_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**5 + fourth_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**4 + cube_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**3 + quad_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**2 + stretch_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii]) + shift_est[ii]
            fydat = f_x - signal.medfilt(f_x,171) #used to find noise
            fyreal = (f_x-f_x.min())/10.0
            peaks = argrelextrema(fydat,np.greater) #find peaks
            fxpeak = xspectra[peaks] #peaks in wavelength
            fxrpeak = p_x[peaks] #peaks in pixels
            fypeak = fydat[peaks] #peaks heights (for noise)
            fyrpeak = fyreal[peaks] #peak heights
            noise = np.std(np.sort(fydat)[:np.round(fydat.size*0.5)]) #noise level
            peaks = peaks[0][fypeak>noise]
            fxpeak = fxpeak[fypeak>noise] #significant peaks in wavelength
            fxrpeak = fxrpeak[fypeak>noise] #significant peaks in pixels
            fypeak = fyrpeak[fypeak>noise] #significant peaks height
            for j in range(wm.size):
                line_matches['lines'].append(wm[j]) #line positions
                line_matches['line_mags'].append(fm[j])
                nearest_line = np.argsort(np.abs(wm[j]-fxpeak))
                line_matches['peaks_p'].append(fxrpeak[nearest_line][0]) #closest peak (in pixels)
                line_matches['peaks_w'].append(fxpeak[nearest_line][0]) #closest peak (in wavelength)
                line_matches['peaks_h'].append(fypeak[nearest_line][0]) #closest peak (height)
            
            #Pick lines for initial parameter fit
            #cal_states = {'Xe':True,'Ar':False,'HgNe':False,'Ne':False}
            fig,ax = plt.subplots(1)
            
            #maximize window
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.subplots_adjust(right=0.8,left=0.05,bottom=0.20)

            vlines = []
            for j in range(wm.size):
                vlines.append(ax.axvline(wm[j],color='r',alpha=0.5))
            line, = ax.plot(wm,np.zeros(wm.size),'ro')
            yspectra = (f_x-f_x.min())/10.0
            fline, = plt.plot(xspectra,yspectra,'b',lw=1.5,picker=5)
            if len(f_x) != len(p_x):
                pdb.set_trace()            
            browser = LineBrowser(fig,ax,est_features,wm,fm,p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii],Gal_dat.FINAL_SLIT_X_FLIP[ii],vlines,fline,xspectra,yspectra,peaks,fxpeak,fxrpeak,fypeak,line_matches,cal_states)
            fig.canvas.mpl_connect('button_press_event', browser.onclick)
            fig.canvas.mpl_connect('key_press_event',browser.onpress)
            finishax = plt.axes([0.83,0.85,0.15,0.1])
            finishbutton = Button(finishax,'Finish',hovercolor='0.975')
            finishbutton.on_clicked(browser.finish)
            closeax = plt.axes([0.83, 0.65, 0.15, 0.1])
            button = Button(closeax, 'Replace (m)', hovercolor='0.975')
            button.on_clicked(browser.replace_b)
            nextax = plt.axes([0.83, 0.45, 0.15, 0.1])
            nextbutton = Button(nextax, 'Accept (n)', hovercolor='0.975')
            nextbutton.on_clicked(browser.next_go)
            deleteax = plt.axes([0.83,0.25,0.15,0.1])
            delete_button = Button(deleteax,'Delete (j)',hovercolor='0.975')
            delete_button.on_clicked(browser.delete_b)
            #stateax = plt.axes([0.83,0.05,0.15,0.1])
            #states = CheckButtons(stateax,cal_states.keys(), cal_states.values())
            #states.on_clicked(browser.set_calib_lines)
            fig.canvas.draw()
            plt.show()
            if len(f_x) != len(p_x):
                pdb.set_trace()            
            #fit 5th order polynomial to peak/line selections
            params,pcov = curve_fit(polyfour,(np.sort(browser.line_matches['peaks_p'])-Gal_dat.FINAL_SLIT_X_FLIP[ii]),np.sort(browser.line_matches['lines']),p0=[shift_est[ii],stretch_est[ii],quad_est[ii],cube_est[ii],1e-12,1e-12])
            cube_est = cube_est + params[3]
            fourth_est = fourth_est + params[4]
            fifth_est = fifth_est + params[5]
            print(params)
            #make calibration and clip on lower anchor point. Apply to Flux as well
            wave_model =  params[0]+params[1]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])+params[2]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**2+params[3]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**3.0+params[4]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**4.0+params[5]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**5.0
            spectra[keys[ii]]['wave'] = wave_model
            spectra[keys[ii]]['wave2'] = wave_model[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]
            spectra[keys[ii]]['gal_spec2'] = ((np.array(spectra[keys[ii]]['gal_spec']).T[:binnedx:wavelength_dir])[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]).T
            
            flu = f_x - np.min(f_x)
            flu = flu[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]
            Flux = flu/signal.medfilt(flu,201)
            fifth[ii],fourth[ii],cube[ii],quad[ii],stretch[ii],shift[ii] = params[5],params[4],params[3],params[2],params[1],params[0]
            with open('selected_comparison_lines-'+str(ii)+'-'+str(clus_id)+'.pkl','wb') as pklfile:
                pickle.dump(browser.line_matches,pklfile)
            plt.plot(spectra[keys[ii]]['wave2'],Flux/np.max(Flux[np.isfinite(Flux)]))
            plt.plot(wm,fm/np.max(fm),'ro')
            for j in range(browser.wm.size):
                plt.axvline(browser.wm[j],color='r')
            plt.xlim(3800,7500)
            try:
                plt.savefig(datadir+clus_id+'/figs/'+str(ii)+'.wave.png')
            except:
                os.mkdir(datadir+clus_id+'/figs')
                plt.savefig(datadir+clus_id+'/figs/'+str(ii)+'.wave.png')
            plt.show()
            f.write(str(Gal_dat.FINAL_SLIT_X_FLIP[ii])+'\t')
            f.write(str(Gal_dat.FINAL_SLIT_Y[ii])+'\t')
            f.write(str(shift[ii])+'\t')
            f.write(str(stretch[ii])+'\t')
            f.write(str(quad[ii])+'\t')
            f.write(str(cube[ii])+'\t')
            f.write(str(fourth[ii])+'\t')
            f.write(str(fifth[ii])+'\t')
            f.write(str(Gal_dat.SLIT_WIDTH[ii])+'\t')
            f.write('\n')
            print(('Wave calib',ii))
            ii += 1
            break
            
        f.write(str(Gal_dat.FINAL_SLIT_X_FLIP[ii])+'\t')
        f.write(str(Gal_dat.FINAL_SLIT_Y[ii])+'\t')
        f.write(str(shift[ii])+'\t')
        f.write(str(stretch[ii])+'\t')
        f.write(str(quad[ii])+'\t')
        f.write(str(cube[ii])+'\t')
        f.write(str(fourth[ii])+'\t')
        f.write(str(fifth[ii])+'\t')
        f.write(str(Gal_dat.SLIT_WIDTH[ii])+'\t')
        f.write('\n')
        ii+=1

    #usereducedlist = 'n'

    #estimate stretch,shift,quad terms with sliders for 2nd - all galaxies
    for i in range(ii,len(Gal_dat)):
        print(('Calibrating',i,'of',stretch.size))
        if Gal_dat.good_spectra[i] == 'y':
            #usereducedlist = str((raw_input('Should we use the reduced line list for the remainder of the galaxies? (y) or (n)? '))).lower()
            #if usereducedlist:
            #    wm_full = wm
            #    fm_full = fm
            #    wm = np.asarray(line_matches['lines']) #line positions
            #    fm = np.asarray(line_matches['line_mags']) #line heights
            #    pdb.set_trace()
            if sdss_check:
                if Gal_dat.spec_z[i] != 0.0: skipgal = False
                else: skipgal = True
            else: skipgal = False
            skipgal = False
            if not skipgal:
                p_x = np.arange(0,binnedx,1)
                f_x = np.sum(spectra[keys[i]]['arc_spec'],axis=0)[:binnedx]
                f_x = f_x[::wavelength_dir]
                d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[i])+' physical')
                d.set('regions command {box('+str(Gal_dat.SLIT_X[i])+' '+str(Gal_dat.FINAL_SLIT_Y[i])+' '+str(binnedx)+' '+str(Gal_dat.SLIT_WIDTH[i])+') #color=green highlite=1}')
                #stretch_est[i],shift_est[i],quad_est[i] = interactive_plot(p_x,f_x,stretch_est[i-1],shift_est[i-1]-(Gal_dat.FINAL_SLIT_X_FLIP[i]*stretch_est[0]-Gal_dat.FINAL_SLIT_X_FLIP[i-1]*stretch_est[i-1]),quad[i-1],cube[i-1],fourth[i-1],fifth[i-1],Gal_dat.FINAL_SLIT_X_FLIP[i])
                reduced_slits = np.where(stretch != 0.0)
                if len(f_x) != len(p_x):
                    pdb.set_trace()
                
                stretch_est[i],shift_est[i],quad_est[i] = interactive_plot(p_x,f_x,stretch_est[i],shift_est[i],quad_est[i],cube_est[i],fourth_est[i],fifth_est[i],Gal_dat.FINAL_SLIT_X_FLIP[i],wm,fm,cal_states)
                est_features = [fifth_est[i],fourth_est[i],cube_est[i],quad_est[i],stretch_est[i],shift_est[i]]

                #run peak identifier and match lines to peaks
                line_matches = {'lines':[],'line_mags':[],'peaks_p':[],'peaks_w':[],'peaks_h':[]}
                xspectra =  fifth_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**5 + fourth_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**4 + cube_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**3 + quad_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**2 + stretch_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i]) + shift_est[i]
                fydat = f_x - signal.medfilt(f_x,171) #used to find noise
                fyreal = (f_x-f_x.min())/10.0
                peaks = argrelextrema(fydat,np.greater) #find peaks
                fxpeak = xspectra[peaks] #peaks in wavelength
                fxrpeak = p_x[peaks] #peaks in pixels
                fypeak = fydat[peaks] #peaks heights (for noise)
                fyrpeak = fyreal[peaks] #peak heights
                noise = np.std(np.sort(fydat)[:np.round(fydat.size*0.5)]) #noise level
                peaks = peaks[0][fypeak>noise]
                fxpeak = fxpeak[fypeak>noise] #significant peaks in wavelength
                fxrpeak = fxrpeak[fypeak>noise] #significant peaks in pixels
                fypeak = fyrpeak[fypeak>noise] #significant peaks height
                for j in range(wm.size):
                    line_matches['lines'].append(wm[j]) #line positions
                    line_matches['line_mags'].append(fm[j])
                    nearest_line = np.argsort(np.abs(wm[j]-fxpeak))
                    line_matches['peaks_p'].append(fxrpeak[nearest_line][0]) #closest peak (in pixels)
                    line_matches['peaks_w'].append(fxpeak[nearest_line][0]) #closest peak (in wavelength)
                    line_matches['peaks_h'].append(fypeak[nearest_line][0]) #closest peak (height)
                
                #Pick lines for initial parameter fit
                #cal_states = {'Xe':True,'Ar':False,'HgNe':False,'Ne':False}
                fig,ax = plt.subplots(1)
                
                #maximize window
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                plt.subplots_adjust(right=0.8,left=0.05,bottom=0.20)

                vlines = []
                for j in range(wm.size):
                    vlines.append(ax.axvline(wm[j],color='r'))
                line, = ax.plot(wm,fm/2.0,'ro',picker=5)# 5 points tolerance
                yspectra = (f_x-f_x.min())/10.0
                fline, = plt.plot(xspectra,yspectra,'b',lw=1.5,picker=5)
                estx = quad_est[i]*(line_matches['peaks_p']-Gal_dat.FINAL_SLIT_X_FLIP[i])**2 + stretch_est[i]*(line_matches['peaks_p']-Gal_dat.FINAL_SLIT_X_FLIP[i]) + shift_est[i]

                browser = LineBrowser(fig,ax,est_features,wm,fm,p_x-Gal_dat.FINAL_SLIT_X_FLIP[i],Gal_dat.FINAL_SLIT_X_FLIP[i],vlines,fline,xspectra,yspectra,peaks,fxpeak,fxrpeak,fypeak,line_matches,cal_states)
                fig.canvas.mpl_connect('button_press_event', browser.onclick)
                fig.canvas.mpl_connect('key_press_event',browser.onpress)
                finishax = plt.axes([0.83,0.85,0.15,0.1])
                finishbutton = Button(finishax,'Finish',hovercolor='0.975')
                finishbutton.on_clicked(browser.finish)
                closeax = plt.axes([0.83, 0.65, 0.15, 0.1])
                button = Button(closeax, 'Replace (m)', hovercolor='0.975')
                button.on_clicked(browser.replace_b)
                nextax = plt.axes([0.83, 0.45, 0.15, 0.1])
                nextbutton = Button(nextax, 'Next (n)', hovercolor='0.975')
                nextbutton.on_clicked(browser.next_go)
                deleteax = plt.axes([0.83,0.25,0.15,0.1])
                delete_button = Button(deleteax,'Delete (j)',hovercolor='0.975')
                delete_button.on_clicked(browser.delete_b)
                #stateax = plt.axes([0.83,0.05,0.15,0.1])
                #states = CheckButtons(stateax,cal_states.keys(), cal_states.values())
                #states.on_clicked(browser.set_calib_lines)
                fig.canvas.draw()
                plt.show()
                
                #fit 5th order polynomial to peak/line selections
                try:
                    params,pcov = curve_fit(polyfour,(np.sort(browser.line_matches['peaks_p'])-Gal_dat.FINAL_SLIT_X_FLIP[i]),np.sort(browser.line_matches['lines']),p0=[shift_est[i],stretch_est[i],quad_est[i],1e-8,1e-12,1e-12])
                    cube_est[i] = params[3]
                    fourth_est[i] = params[4]
                    fifth_est[i] = params[5]
                except TypeError:
                    params = [shift_est[i],stretch_est[i],quad_est[i],cube_est[i-1],fourth_est[i-1],fifth_est[i-1]]

                print(params)
                #make calibration and clip on lower anchor point. Apply to Flux as well
                wave_model = params[0]+params[1]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])+params[2]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**2+params[3]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**3.0+params[4]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**4.0+params[5]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**5.0
                spectra[keys[i]]['wave'] = wave_model
                spectra[keys[i]]['wave2'] = wave_model[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]
                spectra[keys[i]]['gal_spec2'] = ((np.array(spectra[keys[i]]['gal_spec']).T[:binnedx:wavelength_dir])[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]).T

                flu = f_x[p_x >= np.sort(browser.line_matches['peaks_p'])[0]] - np.min(f_x[p_x >= np.sort(browser.line_matches['peaks_p'])[0]])
                #flu = flu[::-1]
                Flux = flu/signal.medfilt(flu,201)
                fifth[i],fourth[i],cube[i],quad[i],stretch[i],shift[i] = params[5],params[4],params[3],params[2],params[1],params[0]
                plt.plot(spectra[keys[i]]['wave2'],Flux/np.max(Flux))
                plt.plot(wm,fm/np.max(fm),'ro')
                for j in range(browser.wm.size):
                    plt.axvline(browser.wm[j],color='r')
                plt.xlim(3800,7400)
                try:
                    plt.savefig(datadir+clus_id+'/figs/'+str(i)+'.wave.png')
                except:
                    os.mkdir(datadir+clus_id+'/figs')
                    plt.savefig(datadir+clus_id+'/figs/'+str(i)+'.wave.png')
                plt.close()

        f.write(str(Gal_dat.FINAL_SLIT_X_FLIP[i])+'\t')
        f.write(str(Gal_dat.FINAL_SLIT_Y[i])+'\t')
        f.write(str(shift[i])+'\t')
        f.write(str(stretch[i])+'\t')
        f.write(str(quad[i])+'\t')
        f.write(str(cube[i])+'\t')
        f.write(str(fourth[i])+'\t')
        f.write(str(fifth[i])+'\t')
        f.write(str(Gal_dat.SLIT_WIDTH[i])+'\t')
        f.write('\n')
    f.close()
    pickle.dump(spectra,open(datadir+clus_id+'/'+clus_id+'_reduced_spectra_wavecal.pkl','wb'))
else:
    xslit,yslit,shift,stretch,quad,cube,fourth,fifth,wd = np.loadtxt(datadir+clus_id+'/'+clus_id+'_stretchshift.tab',dtype='float',usecols=(0,1,2,3,4,5,6,7,8),unpack=True)
    spectra = pickle.load(open(datadir+clus_id+'/'+clus_id+'_reduced_spectra_wavecal.pkl','rb'))

#summed science slits + filtering to see spectra
#Flux_science_old = np.array([np.sum(scifits_c2.data[Gal_dat.FINAL_SLIT_Y[i]-Gal_dat.SLIT_WIDTH[i]/2.0:Gal_dat.FINAL_SLIT_Y[i]+Gal_dat.SLIT_WIDTH[i]/2.0,:],axis=0)[::-1] for i in range(len(Gal_dat))])
#Flux_science = np.array([np.sum(spectra[keys[i]]['gal_spec'],axis=0)[::-1] for i in range(len(Gal_dat))])
Flux_science = []
for i in range(len(Gal_dat)):
    try:
        Flux_science.append(np.sum(spectra[keys[i]]['gal_spec2'],axis=0))
    except KeyError:
        if i != 0:
            Flux_science.append(np.zeros(len(Flux_science[i-1])))
        else:
            Flux_science.append(np.zeros(binnedx))
Flux_science = np.array(Flux_science)

#Add parameters to Dataframe
Gal_dat['shift'],Gal_dat['stretch'],Gal_dat['quad'],Gal_dat['cube'],Gal_dat['fourth'],Gal_dat['fifth'] = shift,stretch,quad,cube,fourth,fifth


####################
#Redshift Calibrate#
####################

#initialize data arrays
redshift_est = np.zeros(len(Gal_dat))
cor = np.zeros(len(Gal_dat))
HSN = np.zeros(len(Gal_dat))
KSN = np.zeros(len(Gal_dat))
GSN = np.zeros(len(Gal_dat))
SNavg = np.zeros(len(Gal_dat))
SNHKmin = np.zeros(len(Gal_dat))
template = np.zeros(len(Gal_dat))

sdss_elem = np.where(Gal_dat.spec_z > 0.0)[0]
sdss_red = Gal_dat[Gal_dat.spec_z > 0.0].spec_z
qualityval = {'Clear':np.zeros(len(Gal_dat))}

median_sdss_redshift = np.median(Gal_dat.spec_z[Gal_dat.spec_z > 0.0])
print(('Median SDSS redshift',median_sdss_redshift))


run_auto = False
 
# **You can typically leave this be unless you have a custom run ** #
# Create an instantiation of the z_est class for correlating spectra
# To use default search of HK lines with no priors applied select True
R = z_est(lower_w=3500.0,upper_w=7500.0,lower_z=0.05,upper_z=0.6,\
          z_res=3.0e-5,prior_width=0.02,use_zprior=False,\
          skip_initial_priors=True,\
          auto_pilot=run_auto)
     
     
template_names = ['spDR2-023.fit','spDR2-025.fit'] #['spDR2-0'+str(x)+'.fit' for x in np.arange(23,31)]
template_dir='sdss_templates'   #hack

path_to_temps = os.path.abspath(os.path.join(os.curdir,template_dir))  #hack
#Import template spectrum (SDSS early type) and continuum subtract the flux
R.add_sdsstemplates_fromfile(path_to_temps,template_names)
                
          
for k in range(len(Gal_dat)):
    if Gal_dat.slit_type[k] == 'g' and Gal_dat.good_spectra[k] == 'y':
        if sdss_check:
            if Gal_dat.spec_z[k] != 0.0: skipgal = False
            else: skipgal = True
        else: skipgal = False
        if not skipgal:
            test_waveform = waveform(spectra[keys[k]]['wave2'],Flux_science[k],keys[k])
            smoothed_waveform = test_waveform#smooth_waveform(test_waveform)
            #F1 = fftpack.rfft(Flux_science[k])
            #ut = F1.copy()
            #W = fftpack.rfftfreq(spectra[keys[k]]['wave2'].size,d=spectra[keys[k]]['wave2'][1001]-spectra[keys[k]]['wave2'][1000])
            #cut[np.where(W>0.15)] = 0
            #Flux_science2 = fftpack.irfft(cut)
            #Flux_sc = Flux_science2 - signal.medfilt(Flux_science2,171)
            d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[k])+' physical')
            d.set('regions command {box('+str(Gal_dat.SLIT_X[k])+' '+str(Gal_dat.FINAL_SLIT_Y[k])+' '+str(binnedx)+' '+str(Gal_dat.SLIT_WIDTH[k])+') #color=green highlite=1}')
            redshift_outputs = R.redshift_estimate(smoothed_waveform)
            redshift_est[k] = redshift_outputs.best_zest
            cor[k] = redshift_outputs.max_cor
            ztest = redshift_outputs.ztest_vals
            corr_val = redshift_outputs.corr_vals
            template[k] = redshift_outputs.template.name
            if not run_auto:
                qualityval['Clear'][k] = redshift_outputs.qualityval
            try:
                HSN[k],KSN[k],GSN[k] = sncalc(redshift_est[k],smoothed_waveform.wave,smoothed_waveform.continuum_subtracted_flux)
            except ValueError:
                HSN[k],KSN[k],GSN[k] = 0.0,0.0,0.0
            SNavg[k] = np.average(np.array([HSN[k],KSN[k],GSN[k]]))
            SNHKmin[k] = np.min(np.array([HSN[k],KSN[k]]))
       	# Create a summary plot of the best z-fit
        if datadir:
            savestr = 'redEst_%s_Tmplt%s.png' % (smoothed_waveform.name,redshift_outputs.template.name)  
            plt_name = os.path.join(datadir,clus_id,'red_ests',savestr)
            summary_plot(smoothed_waveform.wave, smoothed_waveform.flux, redshift_outputs.template.wave, \
                        redshift_outputs.template.flux, redshift_outputs.best_zest, redshift_outputs.ztest_vals, \
                        redshift_outputs.corr_vals, plt_name, smoothed_waveform.name, None) 

    else:
        redshift_est[k] = 0.0
        cor[k] = 0.0

    if k in sdss_elem.astype('int') and redshift_est[k] > 0:
        print(('Estimate: %.5f'%(redshift_est[k]), 'SDSS: %.5f'%(sdss_red.values[np.where(sdss_elem==k)][0])))
    print(('z found for galaxy '+str(k+1)+' of '+str(len(Gal_dat))))
    print('')

#Add redshift estimates, SN, Corr, and qualityflag to the Dataframe
Gal_dat['est_z'],Gal_dat['cor'],Gal_dat['HSN'],Gal_dat['KSN'],Gal_dat['GSN'],Gal_dat['quality_flag'] = redshift_est,cor,HSN,KSN,GSN,qualityval['Clear']

if np.any(Gal_dat['spec_z'] != 0):
    plt.figure()
    plt.plot(Gal_dat['spec_z'],Gal_dat['est_z'],'ro')
    plt.plot([0,1],[0,1],'k-.')
    plt.title('Redshift Comparison',size=18)
    plt.xlabel('SDSS Spectroscopic Redshift',size=16)
    plt.ylabel('This Spec Estimate',size=16)
    plt.savefig(datadir+clus_id+'/redshift_compare.png')
    plt.show()

f = open(datadir+clus_id+'/estimated_redshifts.tab','w')
f.write('RA\tDEC\tZ_est\tZ_sdss\tcorrelation\tH_S/N\tK_S/N\tG_S/N\ttemplate\tgal_gmag\tgal_rmag\tgal_imag\n')
for k in range(redshift_est.size):
    f.write(Gal_dat.RA[k]+'\t')
    f.write(Gal_dat.DEC[k]+'\t')
    f.write(str(Gal_dat.est_z[k])+'\t')
    f.write(str(Gal_dat.spec_z[k])+'\t')\
    #hack
    #if k in sdss_elem.astype('int'):
    #    f.write(str(sdss_red[sdss_elem==k].values[0])+'\t')
    #else:
    #    f.write(str(0.000)+'\t')
    f.write(str(cor[k])+'\t')
    f.write(str(HSN[k])+'\t')
    f.write(str(KSN[k])+'\t')
    f.write(str(GSN[k])+'\t')
    f.write(str(template[k])+'\t')
    f.write(str(gal_gmag[k])+'\t')
    f.write(str(gal_rmag[k])+'\t')
    f.write(str(gal_imag[k])+'\t')
    f.write('\n')
f.close()


plt.figure()
plt.hist(Gal_dat['est_z'],bins = 50);
plt.title('Redshift Histogram for Cluster SPT-CL 0549-6205  z~0.376\nMask 0, all qualities',size=18); 
plt.ylabel('Number of Galaxies',size=16); 
plt.xlabel('Estimated Redshift',size=16); 
plt.savefig(datadir+clus_id+'/mask0_redshifthistogram.png')

#Output dataframe
Gal_dat.to_csv(datadir+clus_id+'/results.csv')
