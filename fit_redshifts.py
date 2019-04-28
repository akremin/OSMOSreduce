import os
import numpy as np

from astropy.table import Table

#sky_subd_sciences[ap] = [waves,diff,bool_mask]
from zestipy.data_structures import waveform
from zestipy.sncalc import sncalc
from zestipy.z_est import z_est
from zestipy.plotting_tools import summary_plot


def fit_redshifts_wrapper(input_dict):
    return fit_redshifts(**input_dict)


def fit_redshifts(sky_subd_sciences,mask_name,run_auto=True,prior = None,savetemplate_func=None):
    # 3.0e-5
    R = z_est(lower_w=4200.0, upper_w=6400.0, lower_z=0.10, upper_z=0.6, \
              z_res=1.0e-5, prior_width=0.02, use_zprior=False, \
              skip_initial_priors=True, \
              auto_pilot=True)

    template_names = ['spDR2-023.fit']#, 'spDR2-024.fit', 'spDR2-028.fit']
                        # ['spDR2-0'+str(x)+'.fit' for x in np.arange(23,31)]
    template_dir = 'sdss_templates'  # hack

    path_to_temps = os.path.abspath(os.path.join(os.curdir, template_dir))  # hack
    # Import template spectrum (SDSS early type) and continuum subtract the flux
    R.add_sdsstemplates_fromfile(path_to_temps, template_names)

    if run_auto:
        outnames = ['apperature','redshift_est', 'cor', 'template', 'SNavg', 'SNHKmin', 'HSN', 'KSN', 'GSN']
        types = ['S4',float,float,'S3',float,float,float,float,float]
    else:
        outnames = ['apperature','redshift_est', 'quality_val', 'cor', 'template', 'SNavg', 'SNHKmin', 'HSN', 'KSN', 'GSN']
        types = ['S4',float,int,float,'S3',float,float,float,float,float]
    outtable = Table(names=outnames,dtype=types)

    if not run_auto:
        quality_val = {}
    for ap in sky_subd_sciences.keys():
        waves, flux, boolmask = sky_subd_sciences[ap]
        test_waveform = waveform(waves, flux, ap, boolmask)

        redshift_outputs = R.redshift_estimate(test_waveform)
        redshift_est = redshift_outputs.best_zest
        cor = redshift_outputs.max_cor
        ztest = redshift_outputs.ztest_vals
        corr_val = redshift_outputs.corr_vals
        template = redshift_outputs.template.name

        if not run_auto:
            qualityval = redshift_outputs.qualityval
        try:
            HSN, KSN, GSN = sncalc(redshift_est, test_waveform.masked_wave,
                                               test_waveform.continuum_subtracted_flux)
        except ValueError:
            HSN, KSN, GSN = 0.0, 0.0, 0.0

        print("\n\n  {}:".format(ap))
        names = ['Z Best','Mac Cor','Templt','H S/N', 'K S/N', 'G S/N']
        vals = [redshift_est, cor, template, HSN, KSN, GSN]
        for name,val in zip(names,vals):
            if type(val) in [int,str]:
                print('---> {}:\t{}'.format(name, val))
            else:
                print('---> {}:\t{:06f}'.format(name,val))

        SNavg = np.average(np.array([HSN, KSN, GSN]))
        SNHKmin = np.min(np.array([HSN, KSN]))
        # Create a summary plot of the best z-fit
        comment = 'redEst_{}_Tmplt{}'.format(test_waveform.name, redshift_outputs.template.name)
        plt_name = savetemplate_func(cam='',ap=ap,imtype='science',step='zfit',comment=comment)
        summary_plot(test_waveform.wave, test_waveform.flux, redshift_outputs.template.wave, \
                     redshift_outputs.template.flux, redshift_outputs.best_zest, redshift_outputs.ztest_vals, \
                 redshift_outputs.corr_vals, plt_name, test_waveform.name, None)
        if run_auto:
            outtable.add_row([ap,redshift_est, cor, template, SNavg, SNHKmin, HSN, KSN, GSN])
        else:
            outtable.add_row([ap, redshift_est, quality_val, cor, template, SNavg, SNHKmin, HSN, KSN, GSN])

    return outtable