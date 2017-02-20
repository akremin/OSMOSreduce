'''
IMPORTANT NOTES:
In the .oms file, the first and last RA/DEC represent a reference slit at the 
bottom of the mask and the center of the mask respectively.

Please list the calibration lamp(s) used during your observations here
'''
cal_lamp = ['HgNe','Argon']  #['Xenon','Argon'] #'Xenon','Argon','HgNe','Neon'
print(('Using calibration lamps: ', cal_lamp))

import numpy as np
from astropy.io import fits as pyfits
import matplotlib
matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.signal as signal
import sys
import re
import subprocess
import pandas as pd
import copy
import os
import fnmatch
import time
import pickle
import pdb
from scipy import fftpack
from get_photoz import query_galaxies
from slit_find import slit_find
from ds9 import *
from testopt import *
from zestipy import *
from sncalc import *
from gal_trace import *

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

pixscale = 0.15 #arcsec/pixel  #pixel scale at for Goodman
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

datadir = '/u/home/kremin/value_storage/goodman_jan17/'

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

wm = []
fm = []
if 'Xenon' in cal_lamp:
    wm_Xe,fm_Xe = np.loadtxt('osmos_Xenon.dat',usecols=(0,2),unpack=True)
    wm_Xe = air_to_vacuum(wm_Xe)
    wm.extend(wm_Xe)
    fm.extend(fm_Xe)
if 'Argon' in cal_lamp:
    wm_Ar,fm_Ar = np.loadtxt('osmos_Argon.dat',usecols=(0,2),unpack=True)
    wm_Ar = air_to_vacuum(wm_Ar)
    wm.extend(wm_Ar)
    fm.extend(fm_Ar)
if 'HgNe' in cal_lamp:
    wm_HgNe,fm_HgNe = np.loadtxt('osmos_HgNe.dat',usecols=(0,2),unpack=True)
    wm_HgNe = air_to_vacuum(wm_HgNe)
    wm.extend(wm_HgNe)
    fm.extend(fm_HgNe)
if 'Neon' in cal_lamp:
    wm_Ne,fm_Ne = np.loadtxt('osmos_Ne.dat',usecols=(0,2),unpack=True)
    wm_Ne = air_to_vacuum(wm_Ne)
    wm.extend(wm_Ne)
    fm.extend(fm_Ne)

fm = np.array(fm)[np.argsort(wm)]
wm = np.array(wm)[np.argsort(wm)]


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

#ask if you want to only reduce sdss galaxies with spectra
try:
    sdss_check = str(sys.argv[2])
    if sdss_check == 'sdss':
        sdss_check = True
    else:
        raise Exception(sdss_check+' is not an accepted input. \'sdss\' is the only accepted input here.')
except IndexError:
    sdss_check = False

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
#########################################################
for curfile in os.listdir(datadir+clus_id+'/maskfiles/'):
    if fnmatch.fnmatch(curfile, '*.oms'):
        omsfile = curfile
with open(datadir+clus_id+'/maskfiles/'+clus_id+'_Mask1.txt','r') as fil:
    slit_X = []
    slit_Y = []
    slit_WIDTH = []
    slit_LENGTH = []
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
    SLIT_NUM = []
    RA = []
    DEC = []
    TYPE = []
    for line in fil:
        if line[:3] == 'MM_':
            mm_per_asec = float(line.split('=')[1].strip('\n\r'))
        if line[:4] == '[Obj':
            SLIT_NUM.append(int(line[7:].strip('\n\r[]')))
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


SLIT_NUM = np.asarray(SLIT_NUM)
correct_order_idx = np.argsort(SLIT_NUM)
SLIT_NUM = SLIT_NUM[correct_order_idx]
RA = np.asarray(RA)[correct_order_idx]
DEC = np.asarray(DEC)[correct_order_idx]
TYPE = np.asarray(TYPE)[correct_order_idx]

# All widths and locs are currently in mm's  -> want pixels
# X,Y in mm for mask is Y,-X for pixels on ccd  
#ie axes are flipped and one is inverted
mm_per_asec = 0.32
SLIT_X = binxpix_mid - np.array(slit_Y[1:-1])*(1/(xbin*pixscale*mm_per_asec))
SLIT_Y = binnedy + np.array(slit_X[1:-1])*(1/(ybin*pixscale*mm_per_asec)) - yshift
##SLIT_X = binxpix_mid + np.array(slit_X[1:-1])*(1/(xbin*pixscale*mm_per_asec))
##SLIT_Y = binypix_mid + np.array(slit_Y[1:-1])*(1/(ybin*pixscale*mm_per_asec))# +yshift
#SLIT_WIDTH = np.array(slit_WIDTH[1:])*(1/(ybin*pixscale*mm_per_asec))
#SLIT_LENGTH = np.array(slit_LENGTH[1:])*(1/(xbin*pixscale*mm_per_asec))
SLIT_WIDTH = np.array(slit_WIDTH[1:])*(1/(xbin*pixscale*mm_per_asec))
SLIT_LENGTH = np.array(slit_LENGTH[1:])*(1/(ybin*pixscale*mm_per_asec))
#SLIT_WIDTH = np.array(slit_LENGTH[1:])*(1/(ybin*pixscale*mm_per_asec))
#SLIT_LENGTH = np.array(slit_WIDTH[1:])*(1/(xbin*pixscale*mm_per_asec))
#pdb.set_trace()
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

d = ds9() #start pyds9 and set parameters
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
#skip_assign = 'n'
keys = np.arange(0,Gal_dat.SLIT_WIDTH.size,1).astype('string')
#if os.path.isfile(datadir+clus_id+'/'+clus_id+'_slittypes.pkl'):
#    reassign = (raw_input('Detected slit types file in path. Do you wish to use this (y) or remove and re-assign slit types (n)? '))
skip_assign = 'y'
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
#skip_cr_remov = 'n'
#if os.path.isfile(datadir+clus_id+'/data_products/science/'+clus_id+'_science.cr.fits'):
#    skip_cr_remov = (raw_input('Detected cosmic ray filtered file exists. Do you wish to use this (y) or remove and re-calculate (n)? '))
skip_cr_remov = 'y'
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
reassign = 'n'
if os.path.isfile(datadir+clus_id+'/'+clus_id+'_slit_pos_qual.tab'):
    reassign = (raw_input('Detected slit position and quality file in path. Do you wish to use this (y) or remove and re-adjust (n)? '))
if reassign == 'n':
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
                                cutscidat = flatfits_c.data[lowerbound:upperbound,:]
                                cutarcdat = flatfits_c.data[lowerbound:upperbound,:]
                                science_spec,arc_spec,gal_spec,gal_cuts,lower_lim,upper_lim = slit_find(cutflatdat,cutscidat,cutarcdat,lower_lim,upper_lim,int(Gal_dat.SLIT_LENGTH[i]),n_emptypixs,int(Gal_dat.SLIT_Y[i]))
                                spectra[keys[i]] = {'science_spec':science_spec,'arc_spec':arc_spec,'gal_spec':gal_spec,'gal_cuts':gal_cuts}
                                BOX_WIDTH[i] = upper_lim-lower_lim
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
    np.savetxt(datadir+clus_id+'/'+clus_id+'_slit_pos_qual.tab',np.array(list(zip(FINAL_SLIT_X,FINAL_SLIT_Y,SLIT_WIDTH,good_spectra)),dtype=[('float',float),('float2',float),('int',int),('str','|S1')]),delimiter='\t',fmt='%10.2f %10.2f %3d %s')
    pickle.dump(spectra,open(datadir+clus_id+'/'+clus_id+'_reduced_spectra.pkl','wb'))
else:
    FINAL_SLIT_X,FINAL_SLIT_Y,SLIT_WIDTH = np.loadtxt(datadir+clus_id+'/'+clus_id+'_slit_pos_qual.tab',dtype='float',usecols=(0,1,2),unpack=True)
    good_spectra = np.loadtxt(datadir+clus_id+'/'+clus_id+'_slit_pos_qual.tab',dtype='string',usecols=(3,),unpack=True)
    spectra = pickle.load(open(datadir+clus_id+'/'+clus_id+'_reduced_spectra.pkl','rb'))

Gal_dat['FINAL_SLIT_X'],Gal_dat['FINAL_SLIT_Y'],Gal_dat['SLIT_WIDTH'],Gal_dat['good_spectra'] = FINAL_SLIT_X,FINAL_SLIT_Y,SLIT_WIDTH,good_spectra

#Need to flip FINAL_SLIT_X coords to account for reverse wavelength spectra
Gal_dat['FINAL_SLIT_X_FLIP'] = binnedx - Gal_dat.FINAL_SLIT_X
#Gal_dat['FINAL_SLIT_X_FLIP'] = 4064 - Gal_dat.FINAL_SLIT_X
####################################################################


########################
#Wavelength Calibration#
########################
reassign = 'n'
#wave = np.zeros((len(Gal_dat),4064))
if os.path.isfile(datadir+clus_id+'/'+clus_id+'_stretchshift.tab'):
    reassign = (raw_input('Detected file with stretch and shift parameters for each spectra. Do you wish to use this (y) or remove and re-adjust (n)? '))
if reassign == 'n':
    #create write file
    f = open(datadir+clus_id+'/'+clus_id+'_stretchshift.tab','w')
    f.write('#X_SLIT_FLIP     Y_SLIT     SHIFT     STRETCH     QUAD     CUBE     FOURTH    FIFTH    WIDTH \n')
    
    #initialize polynomial arrays
    fifth,fourth,cube,quad,stretch,shift =  np.zeros((6,len(Gal_dat)))
    shift_est = 4.71e-6*(Gal_dat['FINAL_SLIT_X'] - (binnedx/2)-200)**2 + 4.30e-6*(Gal_dat['FINAL_SLIT_Y'] - (binnedy/2))**2 + 4469.72
    stretch_est = -9.75e-9*(Gal_dat['FINAL_SLIT_X'] - (binnedx/2)+100)**2 - 2.84e-9*(Gal_dat['FINAL_SLIT_Y'] - (binnedy/2))**2 + 0.7139
    quad_est = 8.43e-9*(Gal_dat['FINAL_SLIT_X'] - (binnedx/2)+100) + 1.55e-10*(Gal_dat['FINAL_SLIT_Y'] - (binnedy/2)) + 1.3403e-5
    cube_est = 7.76e-13*(Gal_dat['FINAL_SLIT_X'] - (binnedx/2)+100) + 4.23e-15*(Gal_dat['FINAL_SLIT_Y'] - (binnedy/2)) - 5.96e-9
    fifth_est,fourth_est = np.zeros((2,len(Gal_dat)))
    calib_data = arcfits_c.data
    p_x = np.arange(0,binnedx,1)
    ii = 0
    
    #do reduction for initial galaxy
    while ii <= stretch.size:
        if good_spectra[ii]=='y':
            f_x = np.sum(spectra[keys[ii]]['arc_spec'],axis=0)
            d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[ii])+' physical')
            d.set('regions command {box('+str(int(binnedy/2))+' '+str(Gal_dat.FINAL_SLIT_Y[ii])+' '+str(binnedx)+' '+str(Gal_dat.SLIT_WIDTH[ii])+') #color=green highlite=1}')
            
            #initial stretch and shift
            stretch_est[ii],shift_est[ii],quad_est[ii] = interactive_plot(p_x,f_x,stretch_est[ii],shift_est[ii],quad_est[ii],cube_est[ii],fourth_est[ii],fifth_est[ii],Gal_dat.FINAL_SLIT_X_FLIP[ii],wm,fm)

            #run peak identifier and match lines to peaks
            line_matches = {'lines':[],'peaks_p':[],'peaks_w':[],'peaks_h':[]}
            est_features = [fifth_est[ii],fourth_est[ii],cube_est[ii],quad_est[ii],stretch_est[ii],shift_est[ii]]
            xspectra = fifth_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**5 + fourth_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**4 + cube_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**3 + quad_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**2 + stretch_est[ii]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii]) + shift_est[ii]
            fydat = f_x[::-1] - signal.medfilt(f_x[::-1],171) #used to find noise
            fyreal = (f_x[::-1]-f_x.min())/10.0
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
                line_matches['peaks_p'].append(fxrpeak[np.argsort(np.abs(wm[j]-fxpeak))][0]) #closest peak (in pixels)
                line_matches['peaks_w'].append(fxpeak[np.argsort(np.abs(wm[j]-fxpeak))][0]) #closest peak (in wavelength)
                line_matches['peaks_h'].append(fypeak[np.argsort(np.abs(wm[j]-fxpeak))][0]) #closest peak (height)
            
            #Pick lines for initial parameter fit
            cal_states = {'Xe':True,'Ar':False,'HgNe':False,'Ne':False}
            fig,ax = plt.subplots(1)
            
            #maximize window
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
            plt.subplots_adjust(right=0.8,left=0.05,bottom=0.20)

            vlines = []
            for j in range(wm.size):
                vlines.append(ax.axvline(wm[j],color='r',alpha=0.5))
            line, = ax.plot(wm,np.zeros(wm.size),'ro')
            yspectra = (f_x[::-1]-f_x.min())/10.0
            fline, = plt.plot(xspectra,yspectra,'b',lw=1.5,picker=5)
            
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
            params,pcov = curve_fit(polyfour,(np.sort(browser.line_matches['peaks_p'])-Gal_dat.FINAL_SLIT_X_FLIP[ii]),np.sort(browser.line_matches['lines']),p0=[shift_est[ii],stretch_est[ii],quad_est[ii],cube_est[ii],1e-12,1e-12])
            #cube_est = cube_est + params[3]
            fourth_est = fourth_est + params[4]
            fifth_est = fifth_est + params[5]
            
            #make calibration and clip on lower anchor point. Apply to Flux as well
            wave_model =  params[0]+params[1]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])+params[2]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**2+params[3]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**3.0+params[4]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**4.0+params[5]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[ii])**5.0
            spectra[keys[ii]]['wave'] = wave_model
            spectra[keys[ii]]['wave2'] = wave_model[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]
            spectra[keys[ii]]['gal_spec2'] = ((np.array(spectra[keys[ii]]['gal_spec']).T[::-1])[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]).T
            
            flu = f_x - np.min(f_x)
            flu = flu[::-1][p_x >= np.sort(browser.line_matches['peaks_p'])[0]]
            Flux = flu/signal.medfilt(flu,201)
            fifth[ii],fourth[ii],cube[ii],quad[ii],stretch[ii],shift[ii] = params[5],params[4],params[3],params[2],params[1],params[0]
            plt.plot(spectra[keys[ii]]['wave2'],Flux/np.max(Flux[np.isfinite(Flux)]))
            plt.plot(wm,fm/np.max(fm),'ro')
            for j in range(browser.wm.size):
                plt.axvline(browser.wm[j],color='r')
            plt.xlim(3800,6000)
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

    #estimate stretch,shift,quad terms with sliders for 2nd - all galaxies
    for i in range(ii,len(Gal_dat)):
        print(('Calibrating',i,'of',stretch.size))
        if Gal_dat.good_spectra[i] == 'y':
            if sdss_check:
                if Gal_dat.spec_z[i] != 0.0: skipgal = False
                else: skipgal = True
            else: skipgal = False
            if not skipgal:
                p_x = np.arange(0,binnedx,1)
                f_x = np.sum(spectra[keys[i]]['arc_spec'],axis=0)
                d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[i])+' physical')
                d.set('regions command {box(2000 '+str(Gal_dat.FINAL_SLIT_Y[i])+' 4500 '+str(Gal_dat.SLIT_WIDTH[i])+') #color=green highlite=1}')
                #stretch_est[i],shift_est[i],quad_est[i] = interactive_plot(p_x,f_x,stretch_est[i-1],shift_est[i-1]-(Gal_dat.FINAL_SLIT_X_FLIP[i]*stretch_est[0]-Gal_dat.FINAL_SLIT_X_FLIP[i-1]*stretch_est[i-1]),quad[i-1],cube[i-1],fourth[i-1],fifth[i-1],Gal_dat.FINAL_SLIT_X_FLIP[i])
                reduced_slits = np.where(stretch != 0.0)
                stretch_est[i],shift_est[i],quad_est[i] = interactive_plot(p_x,f_x,stretch_est[i],shift_est[i],quad_est[i],cube_est[i],fourth_est[i],fifth_est[i],Gal_dat.FINAL_SLIT_X_FLIP[i],wm,fm)
                est_features = [fifth_est[i],fourth_est[i],cube_est[i],quad_est[i],stretch_est[i],shift_est[i]]

                #run peak identifier and match lines to peaks
                line_matches = {'lines':[],'peaks_p':[],'peaks_w':[],'peaks_h':[]}
                xspectra =  fifth_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**5 + fourth_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**4 + cube_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**3 + quad_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**2 + stretch_est[i]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i]) + shift_est[i]
                fydat = f_x[::-1] - signal.medfilt(f_x[::-1],171) #used to find noise
                fyreal = (f_x[::-1]-f_x.min())/10.0
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
                    line_matches['peaks_p'].append(fxrpeak[np.argsort(np.abs(wm[j]-fxpeak))][0]) #closest peak (in pixels)
                    line_matches['peaks_w'].append(fxpeak[np.argsort(np.abs(wm[j]-fxpeak))][0]) #closest peak (in wavelength)
                    line_matches['peaks_h'].append(fypeak[np.argsort(np.abs(wm[j]-fxpeak))][0]) #closest peak (height)
                
                #Pick lines for initial parameter fit
                cal_states = {'Xe':True,'Ar':False,'HgNe':False,'Ne':False}
                fig,ax = plt.subplots(1)
                
                #maximize window
                figManager = plt.get_current_fig_manager()
                figManager.window.showMaximized()
                plt.subplots_adjust(right=0.8,left=0.05,bottom=0.20)

                vlines = []
                for j in range(wm.size):
                    vlines.append(ax.axvline(wm[j],color='r'))
                line, = ax.plot(wm,fm/2.0,'ro',picker=5)# 5 points tolerance
                yspectra = (f_x[::-1]-f_x.min())/10.0
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

                
                #make calibration and clip on lower anchor point. Apply to Flux as well
                wave_model = params[0]+params[1]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])+params[2]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**2+params[3]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**3.0+params[4]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**4.0+params[5]*(p_x-Gal_dat.FINAL_SLIT_X_FLIP[i])**5.0
                spectra[keys[i]]['wave'] = wave_model
                spectra[keys[i]]['wave2'] = wave_model[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]
                spectra[keys[i]]['gal_spec2'] = ((np.array(spectra[keys[i]]['gal_spec']).T[::-1])[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]).T

                flu = f_x[p_x >= np.sort(browser.line_matches['peaks_p'])[0]] - np.min(f_x[p_x >= np.sort(browser.line_matches['peaks_p'])[0]])
                flu = flu[::-1]
                Flux = flu/signal.medfilt(flu,201)
                fifth[i],fourth[i],cube[i],quad[i],stretch[i],shift[i] = params[5],params[4],params[3],params[2],params[1],params[0]
                plt.plot(spectra[keys[i]]['wave2'],Flux/np.max(Flux))
                plt.plot(wm,fm/np.max(fm),'ro')
                for j in range(browser.wm.size):
                    plt.axvline(browser.wm[j],color='r')
                plt.xlim(3800,6000)
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

#Import template spectrum (SDSS early type) and continuum subtract the flux
early_type = pyfits.open('spDR2-023.fit')
coeff0 = early_type[0].header['COEFF0']
coeff1 = early_type[0].header['COEFF1']
early_type_flux = early_type[0].data[0] - signal.medfilt(early_type[0].data[0],171)
early_type_wave = 10**(coeff0 + coeff1*np.arange(0,early_type_flux.size,1))

#initialize data arrays
redshift_est = np.zeros(len(Gal_dat))
cor = np.zeros(len(Gal_dat))
HSN = np.zeros(len(Gal_dat))
KSN = np.zeros(len(Gal_dat))
GSN = np.zeros(len(Gal_dat))
SNavg = np.zeros(len(Gal_dat))
SNHKmin = np.zeros(len(Gal_dat))

sdss_elem = np.where(Gal_dat.spec_z > 0.0)[0]
sdss_red = Gal_dat[Gal_dat.spec_z > 0.0].spec_z
qualityval = {'Clear':np.zeros(len(Gal_dat))}

median_sdss_redshift = np.median(Gal_dat.spec_z[Gal_dat.spec_z > 0.0])
print(('Median SDSS redshift',median_sdss_redshift))

R = z_est()

for k in range(len(Gal_dat)):
    if Gal_dat.slit_type[k] == 'g' and Gal_dat.good_spectra[k] == 'y':
        if sdss_check:
            if Gal_dat.spec_z[k] != 0.0: skipgal = False
            else: skipgal = True
        else: skipgal = False
        if not skipgal:
            F1 = fftpack.rfft(Flux_science[k])
            cut = F1.copy()
            W = fftpack.rfftfreq(spectra[keys[k]]['wave2'].size,d=spectra[keys[k]]['wave2'][1001]-spectra[keys[k]]['wave2'][1000])
            cut[np.where(W>0.15)] = 0
            Flux_science2 = fftpack.irfft(cut)
            Flux_sc = Flux_science2 - signal.medfilt(Flux_science2,171)
            d.set('pan to 1150.0 '+str(Gal_dat.FINAL_SLIT_Y[k])+' physical')
            d.set('regions command {box(2000 '+str(Gal_dat.FINAL_SLIT_Y[k])+' 4500 '+str(Gal_dat.SLIT_WIDTH[k])+') #color=green highlite=1}')
            redshift_est[k],cor[k],ztest,corr_val,qualityval['Clear'][k] = R.redshift_estimate(early_type_wave,early_type_flux,spectra[keys[k]]['wave2'],Flux_science2,gal_prior=None)
            try:
                HSN[k],KSN[k],GSN[k] = sncalc(redshift_est[k],spectra[keys[k]]['wave2'],Flux_sc)
            except ValueError:
                HSN[k],KSN[k],GSN[k] = 0.0,0.0,0.0
            SNavg[k] = np.average(np.array([HSN[k],KSN[k],GSN[k]]))
            SNHKmin[k] = np.min(np.array([HSN[k],KSN[k]]))

    else:
        redshift_est[k] = 0.0
        cor[k] = 0.0

    if k in sdss_elem.astype('int') and redshift_est[k] > 0:
        print(('Estimate: %.5f'%(redshift_est[k]), 'SDSS: %.5f'%(sdss_red.values[np.where(sdss_elem==k)][0])))
    print(('z found for galaxy '+str(k+1)+' of '+str(len(Gal_dat))))
    print('')

#Add redshift estimates, SN, Corr, and qualityflag to the Dataframe
Gal_dat['est_z'],Gal_dat['cor'],Gal_dat['HSN'],Gal_dat['KSN'],Gal_dat['GSN'],Gal_dat['quality_flag'] = redshift_est,cor,HSN,KSN,GSN,qualityval['Clear']

plt.plot(Gal_dat['spec_z'],Gal_dat['est_z'],'ro')
#plt.plot(sdss_red,redshift_est2[sdss_elem.astype('int')],'bo')
#plt.plot(sdss_red,redshift_est3[sdss_elem.astype('int')],'o',c='purple')
plt.plot(sdss_red,sdss_red,'k')
plt.savefig(datadir+clus_id+'/redshift_compare.png')
plt.show()

f = open(datadir+clus_id+'/estimated_redshifts.tab','w')
f.write('#RA    DEC    Z_est    Z_sdss  correlation   H S/N    K S/N     G S/N  gal_gmag    gal_rmag    gal_imag\n')
for k in range(redshift_est.size):
    f.write(Gal_dat.RA[k]+'\t')
    f.write(Gal_dat.DEC[k]+'\t')
    f.write(str(Gal_dat.est_z[k])+'\t')
    f.write(str(Gal_dat.spec_z[k])+'\t')
    #if k in sdss_elem.astype('int'):
    #    f.write(str(sdss_red[sdss_elem==k].values[0])+'\t')
    #else:
    #    f.write(str(0.000)+'\t')
    f.write(str(cor[k])+'\t')
    f.write(str(HSN[k])+'\t')
    f.write(str(KSN[k])+'\t')
    f.write(str(GSN[k])+'\t')
    f.write(str(gal_gmag[k])+'\t')
    f.write(str(gal_rmag[k])+'\t')
    f.write(str(gal_imag[k])+'\t')
    f.write('\n')
f.close()

#Output dataframe
Gal_dat.to_csv(datadir+clus_id+'/results.csv')
