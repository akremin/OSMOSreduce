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
thar_lamps = np.asarray([627,635])
comp_lamps = np.asarray([628,629,636,637])
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

cal_lamp = ['HgAr', 'NeAr' ,'Xe'] # ['Xe', 'Ar', 'HgNe', 'Hg', 'Ne'] # SALT
# cal_lamp = ['Hg','Ar','Ne','Xe'] #['Ar','He','Hg','Ne','ThAr','Th','Xe']  # NIST
# cal_lamp = ['HgAr', 'NeAr', 'Ar', 'Xe']
# cal_lamp = ['Xenon','Argon','Neon', 'HgNe']
# cal_lamp = ['Xe', 'Ar', 'HgNe', 'Hg', 'Ne']
# thar_lamp = ['Th','ThAr']
thar_lamp = ['ThAr']
cameras = ['r']
opamps = [1,2,3,4]


# In[482]:


path_to_masks = os.path.abspath('../../OneDrive/Research/M2FSReductions')
mask_subdir = mask_name
raw_data_subdir =  'raw_data'
filename_template = {}
filename_template['raw'] = '{cam}{filenum:04d}c{opamp}.fits'


# In[483]:


make_debug_plots = False
print_headers = True
cut_bias_cols = True
convert_adu_to_e = True
load_data_from_disk_each_step = False


# In[484]:


do_step = OrderedDict()
do_step['stitch'] = False
do_step['bias'] = False
do_step['remove_crs']   = False
do_step['ffmerge'] = False
do_step['apcut'] = False
do_step['wavecalib'] = True
do_step['flat'] = False
do_step['combine'] = False
do_step['zfit'] = False


# ###         Beginning of Code

# In[485]:


date = np.datetime_as_string(np.datetime64('today', 'D'))


# In[486]:

mask = os.path.join(path_to_masks,mask_subdir)



# In[487]:

directory = {}
directory['mask'] =         os.path.abspath(mask)
directory['raw_data'] =     os.path.join(mask, raw_data_subdir)
directory['raw_stitched'] = os.path.join(mask,'raw_stitched')
directory['product'] =      os.path.join(mask,'data_products')
directory['twod'] =         os.path.join(mask,'twods')
directory['oned'] =         os.path.join(mask,'oneds')
directory['calibrated'] =   os.path.join(mask,'calibrated_oned')
directory['summedspec'] =   os.path.join(mask,'final_oned')
directory['zfit'] =         os.path.join(mask,'zfits')
directory['linelists'] =    os.path.join(os.curdir,'lamp_linelists','salt')

for dirpath in directory.values():
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


# In[488]:


filenumbers = {'bias':biass, 'thar':thar_lamps, 'comp': comp_lamps,
               'twiflat':twiflats, 'science': sciences, 'fibmap': fibermaps}


# In[489]:


filename_template['base'] =      '{cam}_{imtype}_{filenum:04d}_{maskname}_'
filename_template['stitched'] =  filename_template['base']+'stitched{tags}.fits'
filename_template['twod'] =      filename_template['base']+'{fibername}_2d{tags}.fits'
filename_template['oned'] =      filename_template['base']+'1d{tags}.fits'
filename_template['combined'] =  filename_template['base']+'{fibername}_combined_1d{tags}.fits'
filename_template['linelist'] =  '{lamp}.txt'

filename_template['master_stitched'] =  '{cam}_master{imtype}_{maskname}_stitched{tags}.fits'


# In[490]:


setup_info  =      {    'maskname':         mask_name,
                        'cameras':          cameras,
                        'opamps':           opamps, 
                        'deadfibers':       None          }

load_info   =       {   'datadir':          directory['raw_data'],
                        'template':         filename_template['raw'],
                        'tags':             ''            }

save_info   =       {   'date':             date,
                        'datadir':          directory['raw_stitched'],
                        'template':         filename_template['stitched'],
                        'tags':             ''             }

master_info  =      {   'master_template':  filename_template['master_stitched'],
                        'master_types':     []             }


# In[491]:


start = 'stitch'
for key,val in do_step.items():
    if val:
        start = key
        break


# ### How the hdudict is structured
# 
#     dict_of_hdus  
#          keys: science, comp, thar, flat, fibmap
#          vals: cameras_dict
#               
#          cameras_dict 
#                 keys: r, b
#                 vals: filenumbers_dict
#                            
#                 filenumbers_dict 
#                         keys: integer numbers
#                         vals: opampdict
#                                             
#                         opampdict 
#                                keys: ints 1, 2, 3, 4
#                                vals: hdus for each of 4 opamps

# In[492]:


#start = 'stitch'
if load_data_from_disk_each_step or ('stitch' == start):
    ## Load in data
    dict_of_hdus = get_all_filedata( filenum_dict=filenumbers, 
                                     **setup_info,**load_info,**master_info,
                                      cut_bias_cols=cut_bias_cols, 
                                      convert_adu_to_e = convert_adu_to_e )
    print_data_neatly(dict_of_hdus)


# In[493]:


if do_step['stitch']:
    stitched_hdu_dict = {}
    from quickreduce_funcs import stitch_these_camera_data
    for imtype,camdict in dict_of_hdus.items():
        stitched_hdu_dict[imtype] = {}
        for camera,filenumdict in camdict.items():
            stitched_hdu_dict[imtype][camera] = {}
            if imtype == 'bias':
                allfile_3darray,headers = [],[]
            for filenum,opampdict in filenumdict.items():
                if imtype == 'bias':
                    outhdu = stitch_these_camera_data(opampdict,filenum,camera,imtype,mask_name,
                                                      save_info,save_each=False)
                    headers.append(outhdu.header)
                    allfile_3darray.append(outhdu.data)
                else:
                    outhdu = stitch_these_camera_data(opampdict,filenum,camera,imtype,mask_name,
                                                      save_info, save_each=True,
                                                      make_plot=make_debug_plots)
                    stitched_hdu_dict[imtype][camera][filenum] = outhdu
                    
            if imtype == 'bias':
                filenum_3d_array = np.asarray(allfile_3darray)
                filenum_median_array = np.median(filenum_3d_array,axis=0)
                
                header = headers[0]
                del filenum_3d_array
                del headers
                
                header.add_history("Median Master Bias done by quickreduce on {}".format(date))

                median_outhdu = fits.PrimaryHDU(data=filenum_median_array ,header=header)
                outname = master_info['master_template'].format(cam=camera, imtype=imtype,maskname=mask_name, 
                                                               tags=save_info['tags'])
                filename = os.path.join(save_info['datadir'], outname)
                median_outhdu.writeto( filename ,overwrite=True)

                if make_debug_plots:
                    ## Plot the image
                    plt.figure()
                    plot_array = median_outhdu.data - np.min(median_outhdu.data) + 1e-4
                    plt.imshow(np.log(plot_array),'gray',origin='lowerleft')
                    plt.savefig(filename.replace('.fits','.png'),dpi=1200)
                    plt.show()

                stitched_hdu_dict[imtype][camera] = median_outhdu
            
    dict_of_hdus = stitched_hdu_dict


# In[494]:


## Now that stitching is complete. opamps = None
if 'bias' in filenumbers.keys():
    filenumbers.pop('bias')
#if 'bias' not in master_info['master_types']:
#    master_info['master_types'].append('bias')
    
setup_info['opamps'] = None
load_info['datadir'] = directory['raw_stitched']
save_info['datadir'] = directory['product']
load_info['tags'] = ''
save_info['tags'] = '.b'    
save_info['template'] = filename_template['stitched']
load_info['template'] = filename_template['stitched']

if load_data_from_disk_each_step or ('bias' == start):
        dict_of_hdus = get_all_filedata(filenum_dict=filenumbers, 
                                        **setup_info,**load_info,**master_info)
        print_data_neatly(dict_of_hdus)
        blank_dictofdicts = get_dict_temp(dict_of_hdus)


# In[495]:


if do_step['bias']:
    master_bias_dict = dict_of_hdus.pop('bias')
    debiased_hdus = blank_dictofdicts.copy()
    for imtype,camdict in dict_of_hdus.items():
        for camera,filenumdict in camdict.items():
            master_bias = master_bias_dict[camera].data.astype(float)
            for filenum in filenums:
                filnumarray = dict_of_hdus[imtype][camera][filenum].data.astype(float)
                header = dict_of_hdus[imtype][camera][filenum].header
                
                filnumarray -= master_bias
                
                header.add_history("Bias Subtracted done by quickreduce on {}".format(date))
                
                outhdu = fits.PrimaryHDU(data=filnumarray ,header=header)  
                
                save_hdu(outhdu, save_info, camera, imtype, mask_name,filenum)

                ## Plot the image
                if make_debug_plots:
                    plt.figure()
                    plot_array = outhdu.data - np.min(outhdu.data) + 1e-4
                    plt.imshow(np.log(plot_array),'gray',origin='lowerleft')
                    plt.savefig(filename.replace('.fits','.png'),dpi=1200)
                    plt.show()

                debiased_hdus[imtype][camera][filenum] = outhdu
            print("Completed bias subtraction for {}".format(imtype))
            print("Results saved to {}".format(save_info['datadir']))
    
    dict_of_hdus = debiased_hdus


# In[496]:


load_info['tags'] = '.b'
save_info['tags'] = '.bc' 
load_info['datadir'] = directory['product']
save_info['datadir'] = directory['product']
save_info['template'] = filename_template['stitched']
load_info['template'] = filename_template['stitched']


# In[497]:


if do_step['remove_crs']:
    import PyCosmic 
    for imtype in filenumbers.keys():
        for camera in common_info['cameras']:
            for filenum in filenumbers[imtype]:
                filename = load_info['template'].format(cam=camera, imtype=imtype, maskname=mask_name,
                                                             filenum=filenum, tags=load_info['tags'])
                filename = os.path.join(load_info['datadir'],filename)
                savefile = filename.replace(load_info['tags']+'.fits',save_info['tags']+'.fits')
                maskfile = filename.replace(load_info['tags']+'.fits',save_info['tags']+'.crmask.fits')
                print("\nFor image type: {}, shoe: {},   filenum: {}".format(imtype,camera,filenum))
                outdat,pycosmask,pyheader = PyCosmic.detCos(filename,maskfile,savefile,rdnoise='ENOISE',sigma_det=8,
                                                            gain='EGAIN',verbose=True,return_data=True)
                if make_debug_plots:
                    plot_cr_images(pycosmask,outdat,maskfile,filename)


# In[498]:


load_info['tags'] = '.bc'
save_info['tags'] = '.bc'  
load_info['datadir'] = directory['product']
save_info['datadir'] = directory['oned']
load_info['template'] = filename_template['stitched']
save_info['template'] = filename_template['oned']

if load_data_from_disk_each_step or ('apcut' == start):
        dict_of_hdus = get_all_filedata(filenum_dict=filenumbers, 
                                        **setup_info,**load_info,**master_info)
        print_data_neatly(dict_of_hdus)
        blank_dictofdicts = get_dict_temp(dict_of_hdus)


# In[499]:


if do_step['apcut']:
    from app_detection_helper_funcs import find_aperatures,cutout_1d_aperatures
    from app_detection_helper_funcs import cutout_1d_aperatures

    apcut_hdus = blank_dictofdicts.copy()
    aperatures = {}
    for camera in setup_info['cameras']:
        fib_hdus = dict_of_hdus['fibmap'][camera]
        first_hdu = list(fib_hdus.values())[0]
        sumd_fib_hdu = np.zeros(shape=first_hdu.data.shape)
        
        for val in fib_hdus.values():
            sumd_fib_hdu += val.data
        aperature = find_aperatures(sumd_fib_hdu,camera=camera,function_order=4,                                              deadfibers=setup_info['deadfibers'],                                                      resol_factor=int(100),nvertslices=int(2**6)      )  
        aperatures[camera] = aperature
        for imtype,camdict in dict_of_hdus.items():
            filenumdict = camdict[camera]
            for filenum,hdu in filenumdict.items():
                oneds = cutout_1d_aperatures(hdu.data,aperature)
                outhead = hdu.header.copy(strip=True)
                outhead.remove('datasec',ignore_missing=True)
                outhead.remove('trimsec',ignore_missing=True)
                outhead.remove('CHOFFX',ignore_missing=True)
                outhead.remove('CHOFFY',ignore_missing=True)
                outhead.remove('NOPAMPS',ignore_missing=True)
                 
                outhdu = fits.BinTableHDU(data=Table(data=oneds),header=outhead)
                save_hdu(outhdu, save_info, camera, imtype, mask_name,filenum)

                apcut_hdus[imtype][camera][filenum] = outhdu
                
                if make_debug_plots:
                    plt.figure()
                    for dat in oneds:
                        plt.plot(range(len(dat)),dat)
                    plt.show()
    dict_of_hdus = apcut_hdus


# In[500]:


filenumbers.pop('fibmap')
#master_info['master_types'].append('fibmap')

load_info['tags'] = '.bc'
save_info['tags'] = '.bc'  
load_info['datadir'] = directory['oned']
save_info['datadir'] = directory['oned']
load_info['template'] = filename_template['oned']
save_info['template'] = filename_template['oned']

if load_data_from_disk_each_step or ('wavecalib' == start):
        dict_of_hdus = get_all_filedata(filenum_dict=filenumbers,fibersplit=True,
                                        **setup_info,**load_info,**master_info)
        print_data_neatly(dict_of_hdus)
        blank_dictofdicts = get_dict_temp(dict_of_hdus)


# In[501]:


##loop through and rename, no more camera


# dict_of_hdus['thar']['r'][1]

# In[108]:


deltat = np.datetime64('now','m').astype(int)-np.datetime64('2018-06-01T00:00','m').astype(int)
print(deltat)
#filename = os.path.join(path_to_calibs,'calib_wave_coefs_{}_{}.dat'.format(camera, deltat))


# In[109]:


# head1 = dict_of_hdus['thar']['r'][627].header.copy(strip=True)
# head2 = dict_of_hdus['thar']['r'][635].header.copy(strip=True)
# for key in head1:
#     if key == 'COMMENT' or key == 'HISTORY':
#         continue
#     replace = 'r_'
#     if 'FIBER' in key:
#         replace += key.replace('FIBER','FIB')
#     elif len(key)>6:
#         if key[-1].isdigit():
#             if key[-2].isdigit():
#                 replace+=key[2:]
#             else:
#                 replace+=key[1:-1]
#         else:
#             replace+=key[:6]
#     else:
#         replace+=key
#     head1.rename_keyword(key,replace)
# for key in head2:
#     if key == 'COMMENT' or key == 'HISTORY':
#         continue
#     replace = 'b_'
#     if 'FIBER' in key:
#         replace += key.replace('FIBER','FIB')
#     elif len(key)>6:
#         if key[-1].isdigit():
#             if key[-2].isdigit():
#                 replace+=key[2:]
#             else:
#                 replace+=key[1:-1]
#         else:
#             replace+=key[:6]
#     else:
#         replace+=key
#     head2.rename_keyword(key,replace)
# head1.extend(head2)
# head1


# In[110]:


#dict_of_hdus['thar']['r'][627].header.copy(strip=True)


# In[476]:


# pseudocode

# take all comps

# loop through each aperature, fit to first comp
#                             fit to first thar
#                             loop through remainder of filenums (auto after first?)


# In[506]:

if do_step['wavecalib']:
    load_fromfile_if_possible = False
    import re
    timestamp = np.datetime64('now', 'm').astype(int) - np.datetime64('2018-06-01T00:00', 'm').astype(int)

    from wavelength_calibration_funcs import calibrate_pixels2wavelengths
    from calibrations import wavelength_fitting, interactive_wavelength_fitting

    from calibrations import save_calib_dict,locate_calib_dict

    # bounds = None
    bounds = ([-1e5,0.96,-1e-4,-1e-6,-1e-8,-1e-8],[1e5,1.2,1e-4,1e-6,1e-8,1e-8])
    complinelistdict = load_calibration(cal_lamp, wavemincut=4500, wavemaxcut=6600)
    tharlinelistdict = load_calibration(thar_lamp, wavemincut=4500, wavemaxcut=6600)

    calib_coefs = {}
    calib_coefs['comp'] = {key:{} for key in dict_of_hdus['comp'][setup_info['cameras'][0]].keys()}
    calib_coefs['thar'] = {key:{} for key in dict_of_hdus['thar'][setup_info['cameras'][0]].keys()}
    calib_coefs['interactive'] = {key:{} for key in setup_info['cameras']}
    
    for camera in setup_info['cameras']:
        comp_filenums = list(dict_of_hdus['comp'][camera].keys())
        thar_filenums = list(dict_of_hdus['thar'][camera].keys())

        ## Interactive
        fil = comp_filenums[0]
        if load_fromfile_if_possible:
            coef_table = locate_calib_dict('./', 'interactive',camera,config,fil)
            if coef_table is None:
                load_fromfile_if_possible = False
            else:
                calib_coef_table = coef_table

        first_comp = (dict_of_hdus['comp'][camera][fil]).data

        if not load_fromfile_if_possible:
            calib_coef_table = interactive_wavelength_fitting(first_comp,complinelistdict, \
                                                              default = (4522.6,1.0007,-1.6e-6), \
                                                              trust_initial = True)
            calib_coefs['interactive'][camera] = calib_coef_table
            save_calib_dict(calib_coef_table,'interactive',camera,config,fil,timestamp)

        calib_coefs['interactive'][camera] = calib_coef_table

        ## First pointed fit
        if load_fromfile_if_possible:
            coef_table = locate_calib_dict('./', 'compfit',camera,config,comp_filenums[0])
            if coef_table is None:
                load_fromfile_if_possible = False
            else:
                calib_coef_table = coef_table

        if not load_fromfile_if_possible:
            calib_coef_table, covs, selected_complinelists = wavelength_fitting(first_comp, complinelistdict,\
                                                                          calib_coef_table,select_lines = True, bounds=bounds)
            save_calib_dict(calib_coef_table, 'compfit', camera, config, comp_filenums[0], timestamp)

        calib_coefs['comp'][comp_filenums[0]][camera] = calib_coef_table

        ## Loop through pointed fits
        # for filenum in comp_filenums[1:]:
        #     if load_fromfile_if_possible:
        #         coef_table = locate_calib_dict('./', 'compfit', camera, config, filenum)
        #         if coef_table is None:
        #             load_fromfile_if_possible = False
        #         else:
        #             calib_coef_table = coef_table
        #
        #     if not load_fromfile_if_possible:
        #         comp = dict_of_hdus['comp'][camera][filenum].data
        #         calib_coef_table, covs = wavelength_fitting(comp, selected_complinelists, calib_coef_table)
        #         save_calib_dict(calib_coef_table, 'compfit', camera, config, filenum, timestamp)
        #
        #     calib_coefs['comp'][filenum][camera] = calib_coef_table

        if load_fromfile_if_possible:
            coef_table = locate_calib_dict('./', 'tharfit', camera, config, thar_filenums[0])
            if coef_table is None:
                load_fromfile_if_possible = False
            else:
                calib_coef_table = coef_table

        if not load_fromfile_if_possible:
            first_thar = (dict_of_hdus['thar'][camera][thar_filenums[0]]).data
            calib_coef_table, covs, selected_tharlinelists = wavelength_fitting(first_thar, tharlinelistdict, \
                                                                        calib_coef_table,select_lines = True)
            save_calib_dict(calib_coef_table, 'tharfit', thar_filenums[0], camera, config, timestamp)

        calib_coefs['thar'][thar_filenums[0]][camera] = calib_coef_table

        for filenum in thar_filenums[1:]:
            if load_fromfile_if_possible:
                coef_table = locate_calib_dict('./', 'tharfit', camera, config, filenum)
                if coef_table is None:
                    load_fromfile_if_possible = False
                else:
                    calib_coef_table = coef_table

            if not load_fromfile_if_possible:
                thar = dict_of_hdus['thar'][camera][filenum].data
                calib_coef_table, covs = wavelength_fitting(thar,selected_tharlinelists, calib_coef_table)
                save_calib_dict(calib_coef_table, 'tharfit', camera, config, filenum, timestamp)

            calib_coefs['thar'][filenum][camera] = calib_coef_table

    with open('calib_coefs.pkl','wb') as pklout:
        pkl.dump(calib_coefs,pklout)


# In[113]:


#tab = Table(first_comp)
#np.asarray(tab.columns[tab.colnames])


# In[114]:


dict_of_hdus = get_all_filedata(filenum_dict=filenumbers,fibersplit=True,
                                **setup_info,**load_info,**master_info)
print_data_neatly(dict_of_hdus)
blank_dictofdicts = get_dict_temp(dict_of_hdus)
with open('calib_coefs.pkl','rb') as pklin:
    calib_coefs = pkl.load(pklin)



# In[195]:


import seaborn
def fifthorder(xs,a,b,c,d,e,f):
    return a+b*xs+c*xs*xs+d*xs**3+e*xs**4+f*xs**5

from quickreduce_funcs import format_plot



# In[202]:

fnum = {'comp':628, 'thar':627}
cam = 'r'

best_fits = {}
best_fits['comp'] = calib_coefs['comp'][fnum['comp']][cam]
best_fits['thar'] = calib_coefs['thar'][fnum['thar']][cam]

# In[393]:


for calib_cof in ['comp','thar']:
    for ap in best_fits[calib_cof].colnames:
        for calib_spec in ['comp','thar']:
            #calib_spec = calib_cof
            plt.figure()
            fig, ax = plt.subplots(1,figsize=(12,8))
            flux = dict_of_hdus[calib_spec][cam][fnum[calib_spec]].data[ap]
            pixels = np.arange(flux.size).astype(np.float64)
            best_wave = fifthorder(pixels,*best_fits[calib_cof][ap].data)
            fline, = plt.plot(best_wave, flux/flux.max(), 'b')
            format_plot(ax,title='Pixel to Vacuum Wavelegth fit coefs:{}, spec:{}, {}'.format(calib_cof,                                                                            calib_spec,ap),                       xlabel=r'${\lambda}_{vac}\quad\mathrm{[\AA]}$',                       ylabel='Normalized Intensity')



# In[ ]:


from quickreduce_funcs import pair_exposures
pair_exposures(dict_of_hdus,cams_same=True,max_matches=1)['r'][631]


# In[ ]:


master_info['master_types'].append('twiflat')
#master_info['master_types'].append('fibmap')
filenumbers.pop('twiflat')
filenumbers.pop('fibmap')
load_info['tags'] = '.bc'
save_info['tags'] = '.bcf'
load_info['datadir'] = directory['product']
save_info['datadir'] = directory['product']

if load_data_from_disk_each_step or ('flat' == start):
        dict_of_hdus = get_all_filedata(filenum_dict=filenumbers, 
                                        **setup_info,**load_info,**master_info)
        print_data_neatly(dict_of_hdus)  
        blank_dictofdicts = get_dict_temp(dict_of_hdus)


# In[ ]:


if do_step['flat']:
    for imtype in ['twiflat','fibmap']:
        for camera in cr_removed_data[imtype].keys():
            filenum_3d_array = np.asarray(list(data[imtype][camera].values()))
            header =list(headers[imtype][camera].values())[0]
            
            for exposure in range(filenum_3d_array.shape[0]):
                ## get exposure and make sure there are no negative values
                current_exposure = filenum_3d_array[exposure,:,:]
                current_exposure -= np.min(current_exposure)
                ## get a median smoothed version to remove any peculiarities, find it's max
                median_exposure = median_filter(current_exposure,size=5)
                ## divide by the max of the median exposure to normalize (excluding outliers)
                current_exposure /= np.max(median_exposure)
                filenum_3d_array[exposure,:,:] = current_exposure
            
            filenum_summed_array = np.median(filenum_3d_array,axis=0)
            header.add_history("Summed Master {} done by quickreduce on {}".format(imtype,date))

            outhdu = fits.PrimaryHDU(data=filenum_summed_array ,header=header)
            outname = master_info['master_templates'].format(cam=camera, imtype=imtype,maskname=mask_name, 
                                                           tags=save_info['tags'])
            filename = os.path.join(save_info['datadir'], outname)
            outhdu.writeto( filename ,overwrite=True)

            ## Plot the image
            plt.figure()
            plot_array = outhdu.data - np.min(outhdu.data) + 1e-4
            plt.imshow(np.log(plot_array),'gray',origin='lowerleft')
            plt.savefig(filename.replace('.fits','.png'),dpi=1200)
            plt.show()

            headers[imtype][camera] = header
            data[imtype][camera] = filenum_summed_array
            
    master_twiflat_data = data['twiflat']
    flat_data, flat_headers = {}, {}
    for imtype in filenumbers.keys():
        flat_data[imtype] = {}
        flat_headers[imtype] = {}
        for camera,master_twiflat in master_twiflat_data.items():
            master_twiflat /= np.max(master_twiflat)
            flat_data[imtype][camera] = {}
            flat_headers[imtype][camera] = {}
            datadict = data[imtype][camera]
            
            headerdict = headers[imtype][camera]
            for filnum,filearray in datadict.items():
                filearray = filearray.astype(float)
                header = headerdict[filnum]
                filearray /= master_twiflat.astype(float)
                header.add_history("Flat correction done by quickreduce on {}".format(date))
                outhdu = fits.PrimaryHDU(data=filearray ,header=header)
                filename = save_info['template'].format(cam=camera, imtype=imtype, 
                                                             maskname=mask_name, 
                                                             filenum=filnum, \
                                                             tags=save_info['tags'])
                filename = os.path.join(save_info['datadir'], filename)
                outhdu.writeto( filename ,overwrite=True)

                ## Plot the image
                plt.figure()
                plot_array = outhdu.data - np.min(outhdu.data) + 1e-4
                plt.imshow(np.log(plot_array),'gray',origin='lowerleft')
                plt.savefig(filename.replace('.fits','.png'),dpi=1200)
                plt.show() 
                
                flatnd_data[imtype][camera][filnum] = header
                flatnd_headers[imtype][camera][filnum] = filearray

    print("Completed flattening for {}".format(imtype))
    print("Results saved to {}".format(save_info['datadir']))
    del cr_removed_data, cr_removed_headers


# In[ ]:


if load_data_from_disk_each_step or ('combine' == start):
        dict_of_hdus = get_all_filedata(filenum_dict=filenumbers, 
                                        **setup_info,**load_info,**master_info)
        print_data_neatly(dict_of_hdus)
        blank_dictofdicts = get_dict_temp(dict_of_hdus)


# In[ ]:


if load_data_from_disk_each_step or ('zfit' == start):
        dict_of_hdus = get_all_filedata(filenum_dict=filenumbers, 
                                        **setup_info,**load_info,**master_info)
        print_data_neatly(dict_of_hdus)
        blank_dictofdicts = get_dict_temp(dict_of_hdus)

