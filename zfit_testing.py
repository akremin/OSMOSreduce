import os
import numpy as np
import pickle as pkl

from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from collections import OrderedDict
from scipy.ndimage.filters import median_filter

#sky_subd_sciences[ap] = [waves,diff,bool_mask]
from zestipy.data_structures import waveform, redshift_data, smooth_waveform
from sncalc import sncalc
from zestipy.z_est import z_est
from zestipy.plotting_tools import summary_plot


with open('skysubscience.pkl','rb') as pklin:
    sky_subd_sciences, galaxies, skies, check_skies = pkl.load(pklin)

mask_name = "A02"
run_auto = True
qualityval = None

savedir = os.path.join(os.curdir, 'zfits',mask_name)
if not os.path.exists(savedir):
    os.makedirs(savedir)

R = z_est(lower_w=4200.0, upper_w=6400.0, lower_z=0.05, upper_z=0.6, \
          z_res=3.0e-5, prior_width=0.02, use_zprior=False, \
          skip_initial_priors=True, \
          auto_pilot=True)

template_names = ['spDR2-023.fit', 'spDR2-024.fit',
                  'spDR2-028.fit']  # ['spDR2-0'+str(x)+'.fit' for x in np.arange(23,31)]
template_dir = 'sdss_templates'  # hack

path_to_temps = os.path.abspath(os.path.join(os.curdir, template_dir))  # hack
# Import template spectrum (SDSS early type) and continuum subtract the flux
R.add_sdsstemplates_fromfile(path_to_temps, template_names)

redshift_est, cor, template = {}, {}, {}
SNavg, SNHKmin, HSN, KSN, GSN = {}, {}, {}, {}, {}
for ap in sky_subd_sciences.keys():
    waves, flux, boolmask = sky_subd_sciences[ap]
    test_waveform = waveform(waves, flux, ap, boolmask)

    redshift_outputs = R.redshift_estimate(test_waveform)
    redshift_est[ap] = redshift_outputs.best_zest
    cor[ap] = redshift_outputs.max_cor
    ztest = redshift_outputs.ztest_vals
    corr_val = redshift_outputs.corr_vals
    template[ap] = redshift_outputs.template.name
    print((redshift_outputs.best_zest, redshift_outputs.max_cor, redshift_outputs.template.name))
    if not run_auto:
        qualityval['Clear'][ap] = redshift_outputs.qualityval
    try:
        HSN[ap], KSN[ap], GSN[ap] = sncalc(redshift_est[ap], test_waveform.wave,
                                           test_waveform.continuum_subtracted_flux)
    except ValueError:
        HSN[ap], KSN[ap], GSN[ap] = 0.0, 0.0, 0.0
    SNavg[ap] = np.average(np.array([HSN[ap], KSN[ap], GSN[ap]]))
    SNHKmin[ap] = np.min(np.array([HSN[ap], KSN[ap]]))
    # Create a summary plot of the best z-fit
    savestr = 'redEst_%s_Tmplt%s.png' % (test_waveform.name, redshift_outputs.template.name)
    plt_name = os.path.join(savedir, savestr)
    summary_plot(test_waveform.wave, test_waveform.flux, redshift_outputs.template.wave, \
                 redshift_outputs.template.flux, redshift_outputs.best_zest, redshift_outputs.ztest_vals, \
                 redshift_outputs.corr_vals, plt_name, test_waveform.name, None)