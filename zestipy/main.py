import numpy as np
from astropy.io import fits as pyfits
import matplotlib
#matplotlib.use('Qt4Agg')
#import matplotlib.pyplot as plt
matplotlib.use('Agg')
import os
from scipy import fftpack
import time
from zestipy.z_est import z_est
import zestipy.io as zpio
from zestipy.plotting_tools import summary_plot
import zestipy.workflow_funcs as zwf

#%matplotlib inline


set_directory_name_here = 'lt50.0'


def Redshift_Estimator_Auto(directoryname):
    #############################################
    #           Editable Params                 #
    #############################################
    # Print "this was a test" in ouput table, and
    # don't save output plots?  
    test = False
    
    # dir Name of Interest
    dirnm = directoryname #'lt30.0'
    use_prior = True
    # Import and skip specs you've already done?
    skip_prev_good = False
    
    # Skip test file objects for the above skips?
    use_tests = False
    
    
    # Choose your adventure: 
    # (1) Define bad specs you wish to skip, and run over all others
    # *OR* (2) define good specs and run over only those spectra
    
    # (1) List Bad Frames you want to skip in bad_specs
    bad_frames = []#range(312,500)
    #bad_frames = []
    #bad_frames = np.arange(100)
    
    # (2) You may also specify good frames instead of bad frames
    good_frames = []
    #good_frames = [18,21,22,23]
    #good_frames = np.arange(100)
    
    # **You can typically leave this be unless you have a custom run ** #
    # Create an instantiation of the z_est class for correlating spectra
    # To use default search of HK lines with no priors applied select True
    R = z_est(lower_w=3500.0,upper_w=9999.0,lower_z=0.05,upper_z=1.2,\
              z_res=3.0e-5,skip_initial_priors=True,auto_pilot=True,prior=use_prior)
    
    data_dir = '/nfs/kremin/DESI/quicksim/'
    
    #### relic of old code. Not useful at the moment
    # Import and skip specs you have already skipped?
    skip_prev_bad = True
    
    
    ############################################
    #        Main Body of the Code             #
    ############################################
    # Determine the name of the user for use in uniq_name
    if use_prior:
        username = 'auto_withprior'
    else:
        username = 'auto'
    
    # Avoid overwriting by appending unique name  
    # Initials_DayMonth_HourMinute    
    uniq_name = "%s_%s_" % (time.strftime("%H%M-%d%b", time.gmtime()),username)
    
    # Find path to the directory where this file is housed
    abs_cwd_path = data_dir    #os.getcwd() + '/'
    
    # Define Path to the dir
    #dirnm = 'group' + mask
    path_to_dir = os.path.join(abs_cwd_path,dirnm)
    
    
    # Display to terminal what the user has defined
    l1 = "##  test      = %s" % test
    l2 = "##  dir      = %s" % dirnm
    l3 = "##  uniq_name = %s" % uniq_name
    l4 = "##  abs_cwd_path = %s  ##" % abs_cwd_path
    
    print("\n\n" + len(l4)*'#')
    print(l1 + (len(l4)-len(l1)-2)*' ' + "##")
    print(l2 + (len(l4)-len(l2)-2)*' ' + "##")
    print(l3 +  (len(l4)-len(l3)-2)*' ' + "##")
    print(l4)
    print(len(l4)*'#' + "\n") 
    del l1,l2,l3,l4
    
    # Check the existance of all directories
    zpio.check_directories(abs_cwd_path,dirnm)
    
    # look for previously done results
    if skip_prev_good or skip_prev_bad:
        ignored_objects = zpio.get_file_specs(path_to_dir,skip_prev_good,skip_prev_bad,use_tests)
        ignored_objects = np.array(ignored_objects)
    
    if len(bad_frames)>0:
        use_bad_frames = True
        ignored_frames = np.array(bad_frames)
    elif len(good_frames)>0:
        use_bad_frames = False
        ignored_frames = np.array(good_frames)
    else:
        use_bad_frames = True
        ignored_frames = np.array([])
        
    
    # Display spectra being skipped
    print("Skipping user specified frames:")
    print(ignored_frames)
    print("Skipping previously done objects:")
    print(ignored_objects)
    
    
    
    
    
    #Import template spectrum (SDSS early type) and continuum subtract the flux
    try:
        early_type = pyfits.open(os.path.join(abs_cwd_path,'sdss_templates','spDR2-023.fit'))
    except IOError:
        print("There is no 'sdss_templates/spDR2-023.fit' in the cwd")
        print("cwd =", abs_cwd_path)
        raise("Exiting")
    
    # Declare the array for the template flux(es)
    early_type_flux = np.ndarray((3,len(early_type[0].data[0])))
    early_type_wave = np.ndarray((3,len(early_type[0].data[0])))
    coeff0 = early_type[0].header['COEFF0']
    coeff1 = early_type[0].header['COEFF1']
    early_type_flux[0,:] = early_type[0].data[0]
    early_type_wave[0,:] = 10**(coeff0 + coeff1*np.arange(0,early_type_flux[0,:].size,1))
    
    try:
        early_type = pyfits.open(os.path.join(abs_cwd_path,'sdss_templates','spDR2-024.fit'))
    except IOError:
        print("There is no 'sdss_templates/spDR2-024.fit' in the cwd")
        print("cwd =", abs_cwd_path)
        raise("Exiting")
    coeff0 = early_type[0].header['COEFF0']
    coeff1 = early_type[0].header['COEFF1']
    early_type_flux[1,:] = early_type[0].data[0]
    early_type_wave[1,:] = 10**(coeff0 + coeff1*np.arange(0,early_type_flux[1,:].size,1))
    
    try:
        early_type = pyfits.open(os.path.join(abs_cwd_path ,'sdss_templates','spDR2-025.fit'))
    except IOError:
        print("There is no 'sdss_templates/spDR2-025.fit' in the cwd")
        print("cwd =", abs_cwd_path)
        raise("Exiting")
    coeff0 = early_type[0].header['COEFF0']
    coeff1 = early_type[0].header['COEFF1']
    early_type_flux[2,:] = early_type[0].data[0]
    early_type_wave[2,:] = 10**(coeff0 + coeff1*np.arange(0,early_type_flux[2,:].size,1))
    
    
    
    
    # open the redshift output table and print the column names
    good_spec_file = os.path.join(path_to_dir,uniq_name + '{0}_speczs'.format(dirnm) + '.txt')
    with open(good_spec_file, 'w') as outf:
        if use_prior:
            outf.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ('Frame','Est_z', \
                'Correlation', 'Template', 'True_z', 'MOCK_PHOTOZ', 'Total_SN','MEAN_SN','Med_SN', \
                'Exp','I_Mag','Model') )
        else:
            outf.write("%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" % ('Frame','Est_z', \
                'Correlation', 'Template', 'True_z', 'Total_SN','MEAN_SN','Med_SN', \
                'Exp','I_Mag','Model') )
        
        # Define the path to the fits files and loop over files within that folder
        fits_path = os.path.join(path_to_dir,'fits')
    
        for curfile in os.listdir(fits_path):
            print(curfile)
        
            sn_stat1,sn_stat2,filframe,filext = curfile.split('.')
            sn_status = sn_stat1+'.'+sn_stat2
        
            if filext!='fits' and filext!='fits':
                continue
            if sn_status != dirnm:
                print("The fits file dir isn't matching the specified dir, skipping")
                continue
            if np.any(filframe==ignored_objects):
                print("Previously Done, Skipping")
                continue
            if use_bad_frames:
                if np.any(int(filframe)==ignored_frames):
                    print("Specified to skip in calling function\n")
                    continue
            else:
                if np.all(int(filframe)!=ignored_frames):
                    print("Not in list of good frames, so skipping\n")
                    continue
            del sn_status,filext
        
            # Import the spectrum
            fileName = os.path.join(fits_path,curfile)
            try:
                fits = pyfits.open(fileName)
            except IOError:
                print("File ",fileName," could not be opened")
                print("Moving on to the next spectra")
                continue
            dat = fits[0].data
            wave = dat[1].astype(float)
            raw_flux = dat[0].astype(float)
            # Mask out previously defined bad wavelength regions for the current spec
            masked_flux, masked_wave = zwf.mask_inf_regions(raw_flux,wave)
        
            if sum(masked_flux)==0:
                print("The sum of the flux was 0, skipping")
                continue
            cur_head = fits[0].header
            truez = float(cur_head['TRUE_Z'])
            total_sn = float(cur_head['TOTSQ_SN'])
            median_sn = float(cur_head['MED_SNR'])
            mean_sn = float(cur_head['MEAN_SNR'])
            exposure = float(cur_head['EXPTIME'])
            imag = float(cur_head['ABS_MAG'])
            model = 'lrg'#cur_head['MODEL']
            fits.close()
    
    
            if use_prior:
                std = 0.02*(1+truez)
                mock_photoz = np.random.randn(1)*std + truez
            else:
                mock_photoz = None
            # Declare variables for later
            temp = [0.,0.,0.,0.,0.]
            skip_spectra = False
            #        plt.figure()
            #        plt.plot(masked_wave,masked_flux,'b-')
            #        continue
            #Clean High Frequency noise
            F1 = fftpack.rfft(masked_flux)
            cut = F1.copy()
            W = fftpack.rfftfreq(masked_wave.size,d=masked_wave[(len(masked_wave)-1)]-masked_wave[(len(masked_wave)-2)])
            cut[np.where(W>0.15)] = 0
            Flux_Sci = fftpack.irfft(cut)
            Flux_Science, masked_wave = zwf.mask_neg_regions(Flux_Sci,masked_wave)
            if len(Flux_Science)==0:
                print("The length of Flux_Science was 0 so skipping\n")
                continue
            if len(masked_wave)==0:
                print("The length of masked_wave was 0 so skipping\n")
                continue
        	    # Find the redshifts
            for i in np.arange(len(early_type_flux[:,1])):
                redshift_est,corr,ztest,corr_val = R.redshift_estimate(early_type_wave[i,:], early_type_flux[i,:], masked_wave,Flux_Science,(i+23),gal_prior=mock_photoz)
                print("Template %d, Est Red = %f" % (i+23,redshift_est))
                # Check to see if the fit was better than the previos fits for this spec
                cor = np.max(corr)
                if (cor>=temp[1]):
                    temp = [redshift_est,cor,ztest,corr_val,i]
        
                           
            # Print the best redshift estimate to the terminal
            print("\n\tBest: Template %d, Est Red = %f\n" % (temp[4]+23,temp[0]),"\n")
        
        	# Print params to the output table
        
            # Find the RA and Decs of current object
        
            
            # Print the results into the output "good specs" text file
            if use_prior:
                outf.write("%04d\t%.4f\t%.4f\t%d\t%.1f\t%0.3f\t%.2f\t%.4f\t%.4f\t%.1f\t%.1f\t%s\n"  % (np.int(filframe),temp[0],temp[1],(np.int(temp[4])+23),truez,mock_photoz,total_sn,mean_sn,median_sn,exposure,imag,model))
            else:
                outf.write("%04d\t%.4f\t%.4f\t%d\t%.1f\t%.2f\t%.4f\t%.4f\t%.1f\t%.1f\t%s\n"  % (np.int(filframe),temp[0],temp[1],(np.int(temp[4])+23),truez,total_sn,mean_sn,median_sn,exposure,imag,model))
         
           	# Create a summary plot of the best z-fit
            if not test:
                plt_name = os.path.join(path_to_dir,'red_ests','redEst_%s_%s_Tmplt%d_%s.png' % (dirnm,filframe,np.int(temp[4]+23),uniq_name))
                summary_plot(masked_wave,Flux_Science,early_type_wave[temp[4],:],early_type_flux[temp[4],:],temp[0],temp[2],temp[3],plt_name,filframe,mock_photoz)    
        
        
        # If test, print that in the outputted text files
        if test:
            print("This was a test! Not valid results", file=outf)
    
    
        
        
    
    # if it was a test, ask the user if they want to save the outputted text files
    if test:
        YorN = 'b'
        print("\n\n\n\n\tWould you like to keep the test good spectra output?")
        while (YorN!='Y' and  YorN!='N' and YorN!='n' and YorN!='y'):
            YorN = input('\tYes (y) or No (n): ')
            if (YorN=='N' or YorN=='n'):
                os.remove(good_spec_file)




                
if __name__ == '__main__':
    directory_name = set_directory_name_here
    Redshift_Estimator_Auto(directory_name)
