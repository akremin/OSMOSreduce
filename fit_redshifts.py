import os

import numpy as np
from astropy.table import Table

# sky_subd_sciences[ap] = [waves,diff,bool_mask]
from zestipy.data_structures import waveform
from zestipy.plotting_tools import summary_plot
from zestipy.sncalc import sncalc
from zestipy.z_est import z_est


def fit_redshifts_wrapper(input_dict):
    return fit_redshifts(**input_dict)


def fit_redshifts(sky_subd_sciences,mask_name,run_auto=True,prior = None,savetemplate_func=None):
    # 3.0e-5
    if run_auto:
        outnames = ['FIBNAME','redshift_est', 'cor', 'template', 'SNavg', 'SNHKmin', 'HSN', 'KSN', 'GSN']
        types = ['S4',float,float,'S3',float,float,float,float,float]
    else:
        outnames = ['FIBNAME','redshift_est', 'quality_val', 'cor', 'template', 'SNavg', 'SNHKmin', 'HSN', 'KSN', 'GSN']
        types = ['S4',float,int,float,'S3',float,float,float,float,float]
    outtable = Table(names=outnames,dtype=types)

    science_fiber_names = list(sky_subd_sciences.keys())
    if len(science_fiber_names)>0:
        first_ap = science_fiber_names[0]
    else:
        return outtable
    first_waves, flux, boolmask = sky_subd_sciences[first_ap]

    R = z_est(lower_w=first_waves.min()/(1+0.52), upper_w=first_waves.max()/(1+0.1), lower_z=0.10, upper_z=0.5, \
              z_res=1.0e-5, prior_width=0.02, use_zprior=False, \
              skip_initial_priors=True, \
              auto_pilot=True)
    del first_waves, flux, boolmask
    template_names = ['spDR2-023.fit', 'spDR2-024.fit']#, 'spDR2-028.fit']
                        # ['spDR2-0'+str(x)+'.fit' for x in np.arange(23,31)]
    template_dir = 'sdss_templates'  # hack

    path_to_temps = os.path.abspath(os.path.join(os.curdir, template_dir))  # hack
    # Import template spectrum (SDSS early type) and continuum subtract the flux
    R.add_sdsstemplates_fromfile(path_to_temps, template_names)



    if not run_auto:
        quality_val = {}
    for ap in sky_subd_sciences.keys():
        waves, flux, boolmask = sky_subd_sciences[ap]
        # mask = boolmask.copy()
        nmaskbins = 5 ## must be odd
        start = (nmaskbins - 1)
        half = start // 2
        mask = boolmask[start:].copy()
        for ii in range(1, start + 1):
            mask = (mask | boolmask[(start - ii):-ii])
        mask = np.append(np.append([True] * half, mask), [True] * half)
        test_waveform = waveform(waves, flux, ap, mask)
        # test_waveform = waveform(waves, flux, ap, boolmask)


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
        summary_plot(test_waveform.masked_wave, test_waveform.masked_flux, redshift_outputs.template.wave, \
                     redshift_outputs.template.flux, redshift_outputs.best_zest, redshift_outputs.ztest_vals, \
                 redshift_outputs.corr_vals, plt_name, test_waveform.name, None)
        if run_auto:
            outtable.add_row([ap,redshift_est, cor, template, SNavg, SNHKmin, HSN, KSN, GSN])
        else:
            outtable.add_row([ap, redshift_est, quality_val, cor, template, SNavg, SNHKmin, HSN, KSN, GSN])

    return outtable