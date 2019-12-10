import numpy as np
from numba import jit

from scipy.interpolate import interp1d
from scipy.stats import pearsonr
from matplotlib import gridspec
from zestipy.data_structures import waveform, redshift_data, load_sdss_templatefiles 
from zestipy.plotting_tools import plot_skylines
import matplotlib.pyplot as plt
from astropy.table import Table
from matplotlib.widgets import RadioButtons, Button
import sys
from types import *
import pdb


prior_coeff = 0.02


    
class z_est:
    def __init__(self,lower_w=3900.0,upper_w=7400.0,lower_z=0.1,upper_z=0.5,z_res=3.0e-5,prior_width = 0.02, use_zprior=False,skip_initial_priors=True,auto_pilot=True):
        '''
        Initialize redshift estimate parameters
        '''


        
        #set class attributes
        self.lower_w = lower_w
        self.upper_w = upper_w
        self.lower_z = lower_z
        self.upper_z = upper_z
        self.template_waveforms = np.array([])
        self.z_res = z_res
        self.auto = auto_pilot
        #create redshift array and initialize correlation value array
        self.ztest = np.arange(self.lower_z,self.upper_z,self.z_res)
        self.corr_val_i = np.zeros(self.ztest.size)
        self.qualityval = 0

        #set redshift prior flag
        if skip_initial_priors:
            if use_zprior:
                self.est_pre_z = '1'
                self.z_prior_width = prior_width
            else:
                self.est_pre_z = '3'
                self.z_prior_width = prior_width
            self.uline_n = 'HK'
            self.uline = 3950.0

        else:
            self.est_pre_z = input('(1) Use a known prior [Examples: median of known redshifts. Galaxy photoz measurements] \n'\
                            '(2) View spectrum and specify a redshift prior \n'\
                            '(3) No prior\n')

            #catch and correct false entry
            _est_enter = False
            self.uline_n = input('What is the name of a spectral line you wish to use to identify redshift priors? '\
                                                '[Default: HK]: ')
            if not self.uline_n:
                self.uline_n = 'HK'
            self.uline = input('Please list the approx. rest wavelength (in angstroms) of that line you seek to identify in your spectra '\
                                                '[Default: HK lines are at about 3950]: ')
            if self.uline:
                self.uline = np.float(self.uline)
            else:
                self.uline = 3950.0
            while not _est_enter:
                if self.est_pre_z == '1':
                    self.z_prior_width = prior_width
                    print('redshift prior width has been set to',self.z_prior_width)
                    _est_enter = True
                elif self.est_pre_z == '2':
                    self.z_prior_width = prior_width
                    print('redshift prior width has been set to',self.z_prior_width)
                    _est_enter = True
                elif self.est_pre_z == '3':
                    self.z_prior_width = prior_width
                    _est_enter = True
                else:
                    self.est_pre_z = input('Incorrect entry: Please enter either (1), (2), or (3).')
    
            #remind user to set the correct values in next step
            if self.est_pre_z == '1':
                print('Make sure to set the gal_prior argument to the value of the known redshift prior: '\
                    '[Example: z_est.redshift_estimate(gal_prior=0.1)]')
    
            #postconditions
            assert self.est_pre_z, "Must define redshift prior flag"
            assert self.est_pre_z == '1' or self.est_pre_z == '2' or self.est_pre_z == '3', \
                             "Incorrect string value for prior"



    def add_template(self,new_waveform):
        self.template_waveforms = np.append(self.template_waveforms,[new_waveform])
        
    def add_templates(self,new_waveforms):
        for new_waveform in new_waveforms:
            self.add_template(new_waveform)


    def add_sdsstemplates_fromfile(self,path_to_files='.',filenames=['spDR2-023.fit']):
        new_waveforms = load_sdss_templatefiles(path_to_files,filenames)
        self.add_templates(new_waveforms)


    def redshift_estimate(self,test_waveform,gal_prior=None):
        '''
        estimate redshift for object
        '''
        #manage redshift prior
        self.gal_prior = gal_prior
        
        test_flux = test_waveform.flux
        test_wave = test_waveform.wave

        #handle single redshift prior flag
        if self.est_pre_z == '1':
            if self.gal_prior:
                self.pre_z_est = self.gal_prior
            else:
                nospec = input('You said you are either using a spectroscopic or photometric redshift prior. '\
                                        'You need to specify a prior value! Either enter a number in now or type (q) to exit')
                if nospec == 'q':
                    sys.exit()
                elif not nospec:
                    sys.exit()
                else:
                    self.gal_prior = np.float(nospec)
                    self.pre_z_est = self.gal_prior

        #handle user prior flag
        if self.est_pre_z == '2':
            print('Take a look at the plotted galaxy spectrum and note, approximately, at what wavelength do you see the '+self.uline_n+' line. '\
                    'Then close the plot and enter that wavelength in angstroms.')
            plt.plot(test_wave,test_flux)
            plt.xlim(self.lower_w,self.upper_w)
            plt.show()
            line_init = input(self.uline_n+' approx. wavelength (A): ')
            self.pre_z_est = np.float(line_init)/self.uline - 1

        #handle no prior flag
        if self.est_pre_z == '3':
            self.pre_z_est = None

        redshift_output = self.__find_best_fit(self.pre_z_est,self.z_prior_width,test_waveform)
        return redshift_output




    def __find_best_fit(self,z_est,z_prior_width,test_waveform):
        if self.auto:
            bestfit_info = redshift_data()
        else:
            bestfit_info = redshift_data(qualityval=0)
            
        out_table = Table(names=['template','max_cor','redshift'],dtype=['S10',float,float])
        for template in self.template_waveforms:
            redshift_est,cor,ztest,corr_val = self._cross_cor(z_est,z_prior_width,test_waveform,template)
            
            if not self.auto:
                self.qualityval = 1
                self.first_pass = True
                user_zestimate = self._GUI_display(redshift_est,ztest,corr_val,test_waveform,template)
                if user_zestimate != None:
                    #try:
                    self.first_pass = False
                    redshift_est,cor,ztest,corr_val = self._cross_cor(user_zestimate,z_prior_width,test_waveform,template)
                    user_zestimate = self._GUI_display(redshift_est,ztest,corr_val,test_waveform,template)
                    if user_zestimate != None:
                        redshift_est = user_zestimate
                        if user_zestimate > self.lower_z and user_zestimate < self.upper_z:                     
                            cor = np.asarray(corr_val[np.argmin(np.abs(ztest-user_zestimate))])
                        else:
                            cor = np.zeros(1)
                    #except AttributeError:
                    #    pass   

            #print "Template %s, Est Red = %f" % (template.name,redshift_est)

            #redshift_est = self.spectra2.finalz
            cor = np.max(cor)
            if self.auto:
                if (cor>bestfit_info.max_cor):
                    bestfit_info.best_zest = redshift_est
                    bestfit_info.max_cor = cor
                    bestfit_info.ztest_vals = ztest
                    bestfit_info.corr_vals = corr_val
                    bestfit_info.template = template
            else:
                if self.qualityval > bestfit_info.qualityval:
                    bestfit_info.best_zest = redshift_est
                    bestfit_info.max_cor = cor
                    bestfit_info.ztest_vals = ztest
                    bestfit_info.corr_vals = corr_val
                    bestfit_info.template = template
                    bestfit_info.qualityval=self.qualityval
                elif (self.qualityval == bestfit_info.qualityval) \
                    and (cor>=bestfit_info.max_cor):
                    bestfit_info.best_zest = redshift_est
                    bestfit_info.max_cor = cor
                    bestfit_info.ztest_vals = ztest
                    bestfit_info.corr_vals = corr_val
                    bestfit_info.template = template
                    bestfit_info.qualityval=self.qualityval
                    
        bestfit_info.summary_table = out_table
        return bestfit_info




    def _cross_cor(self,z_est,coeff,test_waveform,temp_waveform):
        '''
        This function cross-correlates a continuum subtracted template spectrum with a continuum subtracted observed spectrum.
        It then returns an estimate of the redshift, the correlation value at that redshift, the array of redshifts tested,
        and the unnormalized correlation value.
        '''

        cont_subd_test_flux = test_waveform.continuum_subtracted_flux
        test_wave = test_waveform.masked_wave
        cont_subd_temp_flux = temp_waveform.continuum_subtracted_flux
        temp_wave = temp_waveform.masked_wave

        cut_at_lowerw = np.greater(test_wave,self.lower_w)    
        cut_at_higherw = np.less(test_wave,self.upper_w)

        index_of_85thpercentile = int(0.85*len(cont_subd_test_flux))
        science_85perc_wavelength = np.sort(test_wave)[index_of_85thpercentile]
        

        z_max = (science_85perc_wavelength/np.float(temp_waveform.min_lam))-1.
        
        z_max_mask = np.where(self.ztest<z_max)
        ztest = self.ztest[z_max_mask]
        corr_val_i = self.corr_val_i[z_max_mask]
        # Find

        #loop over each possible redshift to compute correlation values
        for i in range(ztest.size):
            z = ztest[i]
            #redshift the template wavelengths
            wshift = temp_wave*(1+z)
            min_wshift = temp_waveform.min_lam*(1+z)
            max_wshift = temp_waveform.max_lam*(1+z)
            #identify the wavelength diff between the lower wave limit and the redshifted template spectrum
            #if the limit is above the minimum wavelength of the redshifted template spectrum...
            if (min_wshift < self.lower_w):
                lower_bound = cut_at_lowerw
            else:
                lower_bound = np.greater(test_wave,min_wshift)

            if (max_wshift > self.upper_w):
                upper_bound = cut_at_higherw
            else:
                upper_bound = np.less(test_wave,max_wshift)

            #pdb.set_trace()
            bounds = np.bitwise_and(lower_bound,upper_bound)
            test_wave_vals = test_wave[bounds]
            test_flux_vals = cont_subd_test_flux[bounds]
            #del upper_bound, lower_bound

            #interpolate the redshifted template spectrum and estimate the flux at the observed spectrum wavelengths
            inter = interp1d(wshift,cont_subd_temp_flux)
            et_flux_range = inter(test_wave_vals)

            #calculate the pearson r correlation value between the observed and template flux
            corr_val_i[i] = pearsonr(et_flux_range,test_flux_vals)[0]

        #normalize the correlation values as a function of redshift
        where_finite = np.isfinite(corr_val_i)
        corr_val = corr_val_i[where_finite]#+1)/np.trapz((corr_val_i[where_finite]+1),ztest[where_finite])
        finite_ztest = ztest[where_finite]
        #multiply in prior to likelihood if specified
        if z_est:
            unc = coeff * (1 + z_est)
            zlowerbound = np.max([z_est-5*unc,self.lower_z])
            zupperbound = np.min([z_est+5*unc,self.upper_z])
            assert zlowerbound < zupperbound, "Lower bound of z must be below upper bound"
            if zupperbound < zlowerbound:
                print("Top hat prior failed, proceeding without prior")
                zupperbound = self.upper_z
                zlowerbound = self.lower_z
            #make redshift estimate
            zrange_mask = np.where((finite_ztest>zlowerbound)&(finite_ztest<zupperbound))[0]
        else:
            unc = coeff
            zrange_mask = np.where((finite_ztest>self.lower_z)&(finite_ztest<self.upper_z))[0]
        #pdb.set_trace()
        max_correlation_index = np.nanargmax(corr_val[zrange_mask])
        redshift_est = (finite_ztest[zrange_mask])[max_correlation_index] # NOTE: returns only first index even if multiple indices are equally to max value
        
        #save correlation value at maximum redshift likelihood
        cor = ((corr_val_i[where_finite])[zrange_mask])[max_correlation_index]

        return redshift_est, cor, finite_ztest,corr_val





    def _GUI_display(self,redshift_est,ztest,corr_val,test_waveform,template_waveform):
        wave = test_waveform.masked_wave
        flux_sci = test_waveform.masked_flux
        
        '''Display the spectrum and reference lines.'''
        self.fig = plt.figure(figsize=(10, 8)) 
        gs = gridspec.GridSpec(2, 1, height_ratios=[2, 1])
        ax2 = plt.subplot(gs[0])
        ax = plt.subplot(gs[1])
        plt.subplots_adjust(top=0.96,bottom=0.04,left=0.04,right=0.92)

        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        ax.plot(ztest,corr_val,'b')
        pspec_corr = ax.axvline(redshift_est,color='k',ls='--')
        ax.set_xlabel('Redshift')
        ax.set_ylabel('Correlation')

        self.pspec, = ax2.plot(wave,flux_sci)
        ax2.set_ylim(np.min(flux_sci),np.max(flux_sci))
        ax2.set_xlim(wave[0],wave[-1])
        ax2.plot()
        # Plot sky lines:
        # Red = HK
        # Purple = OII, Halpha
        # Black = sky
        # Blue = Emission
        # Orange = Absorption
        vlin_pspecs = plot_skylines(ax2,redshift_est)

        # Include information in plot            
        self.fig.text(0.923, 0.9, '%s' % template_waveform.name, bbox=dict(facecolor='white', alpha=1.),fontsize=18)
        self.fig.text(0.922, 0.8, 'Blue/ = Emission\nPurple   Lines', bbox=dict(facecolor='white', alpha=1.))
        self.fig.text(0.922, 0.78, 'Red/   = Absorption\nOrange  Lines', bbox=dict(facecolor='white', alpha=1.))
        self.fig.text(0.922, 0.74, 'Black = Sky\n         Lines', bbox=dict(facecolor='white', alpha=1.))

        # from left, from bottom, width, height
        rax = plt.axes([0.925, 0.43, 0.07, 0.22])
        
        # Setup User input box on plot
        if self.qualityval == 1:
            radio = RadioButtons(rax, ('1 - No Clue      ','2 - Slight\n    Chance', '3 - Maybe', '4 - Probably', '5 - Clear'))
        else:
            radio = RadioButtons(rax, ('1 - No Clue      ','2 - Slight\n    Chance', '3 - Maybe', '4 - Probably', '5 - Clear'),active=1)
        def qualfunc(label):
            if label == '5 - Clear':
                self.qualityval = 5
            elif label == '4 - Probably':
                self.qualityval = 4
            elif label == '3 - Maybe':
                self.qualityval = 3
            elif label == '2 - Slight\n    Chance':
                self.qualityval = 2
            else:
                self.qualityval = 1
        radio.on_clicked(qualfunc)
        # from left, from bottom, width, height
        closeax = plt.axes([0.93, 0.18, 0.06, 0.08])
        button = Button(closeax, 'Accept & Close', hovercolor='0.975')
        
        def closeplot(event):
            plt.close()
            
        button.on_clicked(closeplot)

        skip_spec_ax = plt.axes([0.93, 0.94, 0.06, 0.04])
        skip_button = Button(skip_spec_ax, 'skip spectra', hovercolor='0.975')
        
        def skip_spec(event):
            plt.close()
            self.qualityval = 0
            
        skip_button.on_clicked(skip_spec)
        
        #ax2.set_xlim(self.lower_w,self.upper_w)
        ax2.set_xlabel('Wavelength (A)')
        ax2.set_ylabel('Counts')

        # Setup the classification 
        if self.first_pass:
            line_est = Estimateline(self.pspec,ax2,self.uline_n)            
        else:
            spectra2 = DragSpectra(vlin_pspecs,pspec_corr,redshift_est,ax2)
            self.fig.canvas.mpl_connect('motion_notify_event',spectra2.on_motion)
            self.fig.canvas.mpl_connect('button_press_event',spectra2.on_press)
            self.fig.canvas.mpl_connect('button_release_event',spectra2.on_release)
        plt.show()
        if self.qualityval == 0:
            return None
        elif self.first_pass:
            if line_est.lam == 0:
                return None
            else:
                return line_est.lam/self.uline - 1.0
        else:
            return spectra2.finalz


class DragSpectra:
    '''Class to drage the spectra back and forth to match lines of interest'''
    def __init__(self,vlin_spectra,corr_spec,redshift_estimate,ax5):
        self.ax5 = ax5
        self.corr_spec = corr_spec
        self.yzs = self.corr_spec.get_data()[1]
        print('begin shift')
        self.vlin_spectra = vlin_spectra
        self.vline_ys = vlin_spectra[0].get_data()[1]
        self.pressed = False
        self.finalz = redshift_estimate
        #figure.canvas.mpl_connect('motion_notify_event',self.on_motion)
        #figure.canvas.mpl_connect('button_press_event',self.on_press)
        #figure.canvas.mpl_connect('button_release_event',self.on_release)

    def on_motion(self,evt):
        if self.pressed:
            #dx = evt.xdata - self.mouse_x 
            #print "%d %d" % (evt.xdata,self.mouse_x)
            newz = ((evt.xdata/self.mouse_x)*(1.+self.z_on_press))-1.  #((1. + (dx/self.mouse_x))*(1.+self.z0))-1.   
            newxs = self.vline_lams*(evt.xdata/self.mouse_x) # equivalent to spec_x*((1+newz)/(1+z0))
            for i in np.arange(len(self.vlin_spectra)):
                self.vlin_spectra[i].set_data([newxs[i], newxs[i]], self.vline_ys) 
                                        
            self.corr_spec.set_data([newz, newz], self.yzs)
            plt.draw()

    def on_press(self,evt):
        if evt.inaxes == self.ax5:
            self.mouse_x = evt.xdata
            self.z_on_press = self.corr_spec.get_data()[0][0]
            self.vline_lams = np.array([self.vlin_spectra[x].get_data()[0][0] for x in np.arange(len(self.vlin_spectra))])
            self.pressed = True
        else: return

    def on_release(self,evt):
        if evt.inaxes == self.ax5:
            self.pressed = False
            try:
                self.finalz = self.corr_spec.get_data()[0][0]
            except AttributeError:
                self.finalz = self.finalz
        else: return


class Estimateline:
    '''Class to manually estimate where lines are located'''
    def __init__(self,pspec,ax5,uline):
        print('If redshift calibration appears correct, hit "Accept and Close". '\
        'Otherwise, "right click" approx. where the '+uline+' line is in the plotted spectrum. '\
        'The program will re-correlate based on this guess.')
        self.ax5 = ax5
        self.cid3 = pspec.figure.canvas.mpl_connect('button_press_event',self.onclick)
        self.lam = 0
    def on_key_press(self,event):
        if event.key == 'shift':
            self.shift_is_held = True

    def on_key_release(self, event):
        if event.key == 'shift':
            self.shift_is_held = False

    def onclick(self,event):
        if event.inaxes == self.ax5:
            if event.button == 3:
                print('xdata=%f, ydata%f'%(event.xdata, event.ydata))
                self.lam = event.xdata
                plt.close()
            '''
            if event.button == 1:
                #if self.shift_is_held:
                #    print 'xdata=%f, ydata%f'%(event.xdata, event.ydata)
                #    self.lam = event.xdata
                #    plt.close()
                #else:
                plt.close()
            '''
        else: return


if __name__ == '__main__':
    R = z_est()
