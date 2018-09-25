#!/C/Users/kremin/Anaconda3/python.exe
# coding: utf-8

# Basic Walkthrough:
#      1) Define everything
#      2) Create master bias file, Save master bias file  
#      3) Open all other files, sub master bias, save  (*c?.b.fits)
#      4) Remove cosmics from all file types except bias  (*c?.bc.fits)
#      5) Open flats and create master skyflat file, save
#      6) Open all remainging types and divide out master flat, then save  (*c?.bcf.fits)
#      7) Open all remaining types and stitch together, save  (*full.bcf.fits)
#      8) Use fibermap files to determine aperatures
#      9) Use aperatures to cut out same regions in thar,comp,science
#      10) Save all 256 to files with their header tagged name in filename, along with fiber num
#      11) Assume no curvature within tiny aperature region; fit profile to 2d spec and sum to 1d
#      12) Fit the lines in comp spectra, save file and fit solution
#      13) Try to fit lines in thar spectra, save file and fit solution
#      14) Apply to science spectra

# In[478]:


import os
import numpy as np
import pickle as pkl

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.ndimage.filters import median_filter



# In[479]:


from quickreduce_funcs import get_all_filedata, print_data_neatly,                               save_hdu,get_dict_temp


# ### Define input file numbers and other required information
# 
# Ex:
# 
#     Bias 597-626
#     ThAr 627,635
#     NeHgArXe 628,629,636,637
#     Science 631-634
#     Fibermaps 573-577
# 

# In[480]:


biass = np.arange(597,626+1).astype(int)
thar_lamps = np.asarray([627,635]).astype(int)
comp_lamps = np.asarray([628,629,636,637]).astype(int)
twiflats = np.arange(582,591+1).astype(int)
sciences = np.arange(631,634+1).astype(int)
fibermaps = np.arange(573,577+1).astype(int)


# In[505]:


instrument = 'M2FS'
mask_name = 'A02'
config = '11C'

from calibrations import load_calibration_lines_salt_dict as load_calibration

# from calibrations import load_calibration_lines_NIST_dict as load_calibration
# from calibrations import load_calibration_lines_nist_dict as load_calibration

cal_lamp_names = ['HgAr', 'NeAr' ,'Xe'] # ['Xe', 'Ar', 'HgNe', 'Hg', 'Ne'] # SALT
# cal_lamp_names = ['Hg','Ar','Ne','Xe'] #['Ar','He','Hg','Ne','ThAr','Th','Xe']  # NIST
# cal_lamp_names = ['HgAr', 'NeAr', 'Ar', 'Xe']
# cal_lamp_names = ['Xenon','Argon','Neon', 'HgNe']
# cal_lamp_names = ['Xe', 'Ar', 'HgNe', 'Hg', 'Ne']
# thar_lamp_name = ['Th','ThAr']
thar_lamp_name = ['ThAr']
cameras = ['r']
opamps = [1,2,3,4]
deadfibers=None
binning='2x2'
readout_speed='Slow'
m2fs_res_mode='LoRes'
filter=None

pairing_strategy = 'nearest'

# In[482]:


path_to_masks = os.path.abspath('../../OneDrive/Research/M2FSReductions')
mask_subdir = mask_name
raw_data_subdir =  'raw_data'
raw_data_loc=os.path.join(path_to_masks,mask_subdir,raw_data_subdir)
data_product_loc=os.path.join(path_to_masks,mask_subdir)
maskname=mask_name



# In[483]:


make_debug_plots = False
print_headers = True
cut_bias_cols = True
convert_adu_to_e = True
load_data_from_disk_each_step = False


# In[484]:


do_step = OrderedDict()
do_step['stitch'] = False  #1
do_step['bias'] = False #2
do_step['remove_crs']   = False #3
do_step['apcut'] = False  #4
do_step['wavecalib'] = True  #5
do_step['flat'] = True  #6
do_step['skysub'] = True #7
do_step['combine'] = True  #8
do_step['zfit'] = True  #9


# ###         Beginning of Code

# In[485]:
#start = np.array(list(do_step.keys()))[list(do_step.values())][0]
start = 'stitch'
for key,val in do_step.items():
    if val:
        start = key
        break

from directory_config import FieldData, FileManager, InstrumentState
filemanager=FileManager( raw_data_loc=raw_data_loc, data_product_loc=data_product_loc,\
                         maskname=mask_name)
instrument=InstrumentState(cameras=cameras,opamps=opamps,deadfibers=deadfibers,binning=binning,\
                 readout=readout_speed,resolution=m2fs_res_mode,filter=filter,configuration=config)

data = FieldData(sciences, biass, twiflats, fibermaps, \
                 comp_lamps, second_comp_filenums=thar_lamps,
                 filemanager=filemanager, instrument=instrument, startstep=start,\
                 obs_pairing_strategy=pairing_strategy, \
                 calib_lamps1=cal_lamp_names, calib_lamps2=thar_lamp_name)


for step,do_this_step in do_step.items():
    if do_this_step:
        print("Performing {}:".format(step))
        data.proceed_to(step=step)
        data.check_data_ready_for_current_step()#step=step)
        try:
            data.run_step()#step=step)
        except:
            outfile = os.path.join(data_product_loc, '_precrashdata.pkl')
            print("Run step failed to complete. Dumping data to {}".format(outfile))
            with open(outfile,'wb') as crashsave:
                pkl.dump(data.all_hdus,crashsave)
            raise
        try:
            data.save_data()#step=step)
        except:
            outfile = os.path.join(data_product_loc, '_precrashdata.pkl')
            print("Save data failed to complete. Dumping data to {}".format(outfile))
            with open(outfile,'wb') as crashsave:
                pkl.dump(data.all_hdus,crashsave)
            raise
    else:
        print("Skipping {}".format(step))