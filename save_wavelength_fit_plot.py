
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
from quickreduce_funcs import format_plot
from linebrowser import fifth_order_poly
#format_plot(ax, title=None, xlabel=None, ylabel=None, labelsize=16, \
# titlesize=None, ticksize=None, legendsize=None,  legendloc=None):


def create_saveplot(pixels,spectra,fitpix,fitlines,alllines,coefs,cov,savename):
    waves = fifth_order_poly(pixels,*coefs)
    fig = plt.figure(figsize=(20,25))
    ax_lineplot = plt.subplot2grid((5, 4), (0, 0), colspan=4,fig=fig)
    ax_loglineplot = plt.subplot2grid((5, 4), (1, 0), colspan=4,fig=fig)
    ax_alllineplot = plt.subplot2grid((5, 4), (2, 0), colspan=4,fig=fig)
    axfitpts = plt.subplot2grid((5, 4), (3, 0),colspan=2,rowspan=2,fig=fig)
    axcovar = plt.subplot2grid((5, 4), (3, 2),colspan=2,rowspan=2,fig=fig)


    ax_lineplot.plot(waves,spectra,'b-')
    for w in fitlines:
        ax_lineplot.axvline(w, color='r', alpha=0.5)
    format_plot(ax_lineplot, title='Fit with used lines', xlabel='Wavelength', ylabel='Flux', labelsize=16)

    ax_loglineplot.semilogy(waves,spectra,'b-')
    # ax_loglineplot.set_yscale('log')
    for w in fitlines:
        ax_loglineplot.axvline(w, color='r', alpha=0.5)
    format_plot(ax_loglineplot, title='Log Fit with used lines', xlabel='Wavelength', ylabel='Log Flux', labelsize=16)

    ax_alllineplot.plot(waves,spectra,'b-')
    for w in alllines:
       ax_alllineplot.axvline(w, color='gray', alpha=0.5)
    for w in fitlines:
        ax_alllineplot.axvline(w, color='r', alpha=0.5)
    format_plot(ax_alllineplot, title="Fit showing all lines", xlabel='Wavelength', ylabel='Flux', labelsize=16)

    axfitpts.scatter(fitpix,fitlines,'r')
    axfitpts.plot(np.arange(fitpix.min(),fitpix.max(),0.001),fifth_order_poly(fitpix,*coefs),'b-')
    format_plot(axfitpts, title="Fit versus Data", xlabel='Pixels', ylabel='Wavelength', labelsize=16)

    axcovar.matshow(cov)
    axcovar.set_xticklabels(['a', 'b', 'c', 'd', 'e', 'f'])
    axcovar.set_yticklabels(['a', 'b', 'c', 'd', 'e', 'f'])
    format_plot(axcovar, title='Covariance of Fit', xlabel='Param_j Unc', ylabel='Param_i Unc', labelsize=16)
    plt.savefig('{}.png'.format(savename),dpi=600)


