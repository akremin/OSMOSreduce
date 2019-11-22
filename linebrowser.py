

import numpy as np
from scipy.optimize import curve_fit
##  a zoom in window
##  mutlicursor
## And checkboxes
## and radio buttons
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt

from matplotlib.widgets import Slider, Button
from scipy import signal




import numpy as np
from calibration_helper_funcs import get_psf
from scipy.ndimage import gaussian_filter

def fifth_order_poly(p_x, zeroth, first, second, third, fourth, fifth):
    return zeroth + (first * p_x) + (second * np.power(p_x,2)) + (third * np.power(p_x,3)) + \
                   (fourth * np.power(p_x,4)) + (fifth * np.power(p_x,5))


class LineBrowser:
    def __init__(self, wm,fm,all_wms,mock_spec_w,mock_spec_f, f_x, coefs, bounds=None,edge_line_distance=0.,initiate=True, fibname=''):
        self.bounds = bounds
        self.coefs = np.asarray(coefs,dtype=np.float64)

        p_x = np.arange(len(f_x)).astype(np.float64)
        self.p_x = p_x
        xspectra = fifth_order_poly(p_x,*coefs)

        deviation = edge_line_distance
        good_waves = ((wm>(xspectra[0]-deviation))&(wm<(xspectra[-1]+deviation)))
        wm,fm = wm[good_waves],fm[good_waves]/3.
        good_waves = ((all_wms>(xspectra[0]-deviation))&(all_wms<(xspectra[-1]+deviation)))
        all_wms = all_wms[good_waves]
        del good_waves

        # fydat = f_x - signal.medfilt(f_x, 171)
        fyreal = (f_x - f_x.min()) #/ 10.0

        # noise = np.mean(f_x - signal.medfilt(f_x, 3)) # noise level
        # noise = np.std(fyreal[fyreal < np.quantile(fyreal, q=0.05)]) # noise level = variation in low flux pixels
        # noise = np.std(np.sort(fydat)[: (fydat.size // 2)]) # noise level = variation in lower half of flux pix
        noise = np.sqrt(np.median(fyreal)) # noise level = poisson of median
        # noise = np.sqrt(np.mean(fyreal)) # noise level = poisson of mean
        signal_threshhold = 3.*noise
        peaks,properties = signal.find_peaks(fyreal,prominence=(signal_threshhold, 1e9))  # find peaks
        fxpeak = xspectra[peaks]  # peaks in wavelength
        fxrpeak = p_x[peaks]  # peaks in pixels
        fypeak = fyreal[peaks]  # peaks heights

        # significant_peaks = fyrpeak > noise
        # peaks = peaks[significant_peaks]
        # fxpeak = fxpeak[significant_peaks]  # significant peaks in wavelength
        # fxrpeak = fxrpeak[significant_peaks]  # significant peaks in pixels
        # fypeak = fyrpeak[significant_peaks]  # significant peaks height


        line_matches = {'lines': [], 'peaks_p': [], 'peaks_w': [], 'peaks_h': []}
        for j in range(wm.size):
            line_matches['lines'].append(wm[j])  # line positions
            line_matches['peaks_p'].append(fxrpeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (in pixels)
            line_matches['peaks_w'].append(
                fxpeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (in wavelength)
            line_matches['peaks_h'].append(fypeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (height)

        yspectra = fyreal

        self.lastind = 0

        self.j = 0
        self.px = p_x
        self.all_wms = all_wms
        self.wm = wm
        self.fm = fm

        self.mock_spec_w = mock_spec_w
        self.mock_spec_f = mock_spec_f
        self.convd_mock_spec_f = None
        self.deviation = deviation
        self.xspectra = xspectra
        self.yspectra = yspectra
        self.peaks = peaks
        self.peaks_w = fxpeak
        self.peaks_p = fxrpeak
        self.peaks_h = fypeak
        self.line_matches = line_matches
        self.mindist_el, = np.where(self.peaks_w == self.line_matches['peaks_w'][self.j])
        self.fibname = fibname

        self.last = {'j':[],'lines':[],'peaks_h':[],'peaks_w':[],'peaks_p':[],'vlines':[],'wm':[],'fm':[]}
        if initiate:
            self.initiate_browser()

    def make_mock_spec(self):
        psf = get_psf(self.yspectra, step=1., prom_quantile=0.68, npeaks=6)
        convd_mock_spec_f = gaussian_filter(self.mock_spec_f, sigma=psf / (self.mock_spec_w[1] - self.mock_spec_w[0]),
                                            order=0)
        good_waves = ((self.mock_spec_w > (self.xspectra[0] - self.deviation)) & (
                    self.mock_spec_w < (self.xspectra[-1] + self.deviation)))
        self.mock_spec_w = self.mock_spec_w[good_waves]
        convd_mock_spec_f = convd_mock_spec_f[good_waves]
        self.convd_mock_spec_f = 0.1 * self.yspectra.max() * convd_mock_spec_f / convd_mock_spec_f.max()

    def initiate_browser(self):
        fig, ax = plt.subplots(1)
        plt.title(self.fibname)

        plt.subplots_adjust(right=0.8, left=0.05, bottom=0.20)

        if self.convd_mock_spec_f is not None:
            self.make_mock_spec()
            ax.fill_between(self.mock_spec_w, self.convd_mock_spec_f / 2., y2=0., color='gray', alpha=0.1)
        for w in self.all_wms:
            ax.axvline(w, color='gray', alpha=0.2)

        vlines = []
        for w in self.wm:
            vlines.append(ax.axvline(w, color='r', alpha=0.5))

        fline, = plt.plot(self.xspectra, self.yspectra, 'b', picker=5)
        line, = ax.plot(self.wm, np.zeros(self.wm.size), 'ro')
        est_f, = ax.plot(self.wm, self.fm / 2.0, 'ro', picker=5,alpha=0.8)  # 5 points tolerance

        self.fig = fig
        self.ax = ax
        self.est_f = est_f
        self.vlines = vlines
        self.fline = fline
        # self.text = ax.text(0.05, 0.95, 'Pick red reference line',transform=ax.transAxes, va='top')
        # self.selected,  = ax.plot([xs[0]], [ys[0]], 'o', ms=12, alpha=0.4,color='yellow', visible=False)
        self.selected = self.ax.axvline(self.line_matches['lines'][self.j], lw=3, alpha=0.5, color='red', ymin=0.5)
        self.selected_peak, = self.ax.plot(self.line_matches['peaks_w'][self.j], self.line_matches['peaks_h'][self.j],
                                           'o', mec='orange', markersize=12, alpha=0.8, mfc='None', mew=3, visible=True)
        self.selected_peak_line = self.ax.axvline(self.line_matches['lines'][self.j], color='cyan', lw=4, alpha=0.3,
                                                  ymax=0.6, visible=True)
        self.reset_lims()
        self.update_current(draw=False)

    def update_current(self,draw=True):
        if self.j >= len(self.line_matches['peaks_w']):
            print('done with plot')
            plt.close()
            return
        self.selected_peak.set_xdata(self.line_matches['peaks_w'][self.j])
        self.selected_peak.set_ydata(self.line_matches['peaks_h'][self.j])
        self.selected.set_xdata(self.line_matches['lines'][self.j])
        self.selected_peak_line.set_xdata(self.line_matches['lines'][self.j])
        self.mindist_el, = np.where(self.peaks_w == self.line_matches['peaks_w'][self.j])
        self.mindist_el = self.mindist_el[0]

        xlims = self.ax.xaxis.get_view_interval()
        ylims = self.ax.yaxis.get_view_interval()
        xwindowprob = self.line_matches['lines'][self.j] > (xlims[1]-100) or (self.line_matches['lines'][self.j] < (xlims[0]+100))
        y_too_low = self.line_matches['peaks_h'][self.j] > (ylims[1]*0.91)
        y_too_high = self.line_matches['peaks_h'][self.j] < (ylims[1] / 50.)
        ywindowprob = y_too_high or y_too_low
        if xwindowprob or ywindowprob:
            self.reset_lims(ywinbool=ywindowprob,yhighbool=y_too_high)
        if draw:
            self.fig.canvas.draw()

    def reset_lims(self,ywinbool=False,yhighbool=False):
        if not ywinbool:
            self.ax.set_xlim(self.line_matches['peaks_w'][self.j] - 100, self.line_matches['peaks_w'][self.j] + 500.0)
        xlims = self.ax.xaxis.get_view_interval()
        ylims = self.ax.yaxis.get_view_interval()
        y_in = self.yspectra[np.where((self.xspectra > xlims[0]) & (self.xspectra < xlims[1]))]
        if yhighbool:
            self.ax.set_ylim(np.min(y_in), 50*self.line_matches['peaks_h'][self.j])
        else:
            self.ax.set_ylim(np.min(y_in), 1.1*np.max(y_in))
        # if 1.8*np.median(y_in) < np.mean(y_in):
        #     self.ax.set_ylim(np.min(y_in), 5*np.mean(y_in) * 10)
        # else:
        #     self.ax.set_ylim(np.min(y_in), np.max(y_in) * 1.1)

    def onpress(self, event):
        if event.key not in ('n', 'm', 'j', 'b', 'u'): return
        if event.key == 'n':
            self.next_line()
        if event.key == 'm':
            self.replace()
        if event.key == 'j':
            self.delete()
        if event.key == 'b':
            self.back_line()
        if event.key == 'u':
            self.undo()
        return

    def onclick(self, event):
        if event.inaxes == self.ax:
            if event.button == 1:
                # the click locations
                x = event.xdata
                y = event.ydata

                self.mindist_el = np.argsort(np.abs(self.peaks_w - x))[0]
                self.update_circle()

    def update_circle(self):
        self.selected_peak.set_xdata([self.peaks_w[self.mindist_el]])
        self.selected_peak.set_ydata([self.peaks_h[self.mindist_el]])
        self.fig.canvas.draw()

    def replace(self):
        self.line_matches['peaks_p'][self.j] = self.peaks_p[self.mindist_el]
        self.line_matches['peaks_w'][self.j] = self.peaks_w[self.mindist_el]
        self.line_matches['peaks_h'][self.j] = self.peaks_h[self.mindist_el]
        self.next_line()
        return

    def back_line(self):
        if self.j >= 1:
            self.j -= 1
            self.update_current()
        else:
            return

    def next_go(self, event):
        self.next_line()

    def next_line(self):
        self.j += 1
        self.update_current()

    def finish(self, event):
        self.line_matches['peaks_p'] = self.line_matches['peaks_p'][:self.j]
        self.line_matches['peaks_w'] = self.line_matches['peaks_w'][:self.j]
        self.line_matches['peaks_h'] = self.line_matches['peaks_h'][:self.j]
        self.line_matches['lines'] = self.line_matches['lines'][:self.j]
        print('FINISHED GALAXY CALIBRATION')
        plt.close()
        return

    def delete_b(self, event):
        self.delete()

    def undo_b(self,event):
        self.undo()

    def replace_b(self, event):
        self.replace()

    def back_b(self,event):
        self.back_line()

    def undo(self):
        if len(self.last['j']) > 0:
            j = self.last['j'].pop()
            self.line_matches['peaks_p'].insert(j, self.last['peaks_p'].pop())
            self.line_matches['peaks_h'].insert(j, self.last['peaks_h'].pop())
            self.line_matches['peaks_w'].insert(j, self.last['peaks_w'].pop())
            self.line_matches['lines'].insert(j, self.last['lines'].pop())

            badline = self.last['vlines'].pop()
            badline.set_visible(True)
            self.vlines.insert(j, badline)

            self.wm = np.insert(self.wm, j, self.last['wm'].pop())
            self.fm = np.insert(self.fm, j, self.last['fm'].pop())

            self.back_line()
        else:
            return

    def delete(self):
        self.last['j'].append(self.j)
        self.last['lines'].append(self.line_matches['lines'].pop(self.j))
        self.last['peaks_p'].append(self.line_matches['peaks_p'].pop(self.j))
        self.last['peaks_w'].append(self.line_matches['peaks_w'].pop(self.j))
        self.last['peaks_h'].append(self.line_matches['peaks_h'].pop(self.j))

        badline = self.vlines.pop(self.j)
        badline.set_visible(False)
        self.last['vlines'].append(badline)

        self.last['wm'].append(self.wm[self.j])
        self.last['fm'].append(self.fm[self.j])

        self.wm = np.delete(self.wm, self.j)
        self.fm = np.delete(self.fm, self.j)
        self.update_current()
        return

    def plot(self):
        self.update_current()
        # maximize window
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()

        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.fig.canvas.mpl_connect('key_press_event', self.onpress)
        finishax = plt.axes([0.83, 0.85, 0.15, 0.1])
        finishbutton = Button(finishax, 'Finish', hovercolor='0.975')
        finishbutton.on_clicked(self.finish)
        closeax = plt.axes([0.83, 0.65, 0.15, 0.1])
        button = Button(closeax, 'Replace (m)', hovercolor='0.975')
        button.on_clicked(self.replace_b)
        nextax = plt.axes([0.83, 0.45, 0.15, 0.1])
        nextbutton = Button(nextax, 'Next (n)', hovercolor='0.975')
        nextbutton.on_clicked(self.next_go)
        deleteax = plt.axes([0.83, 0.25, 0.15, 0.1])
        delete_button = Button(deleteax, 'Delete (j)', hovercolor='0.975')
        delete_button.on_clicked(self.delete_b)
        undoax = plt.axes([0.83, 0.1, 0.15, 0.1])
        undo_button = Button(undoax, 'Undo (u)', hovercolor='0.975')
        undo_button.on_clicked(self.undo_b)
        plt.show()
        return

    def fit(self):
        # fit 5th order polynomial to peak/line selections
        return least_squares_fit(self.coefs,self.line_matches['peaks_p'],self.line_matches['lines'],self.bounds)

    def create_saveplot(self, coefs, cov, savename):
        if self.convd_mock_spec_f is None:
            self.make_mock_spec()
        coefs = np.asarray(coefs)
        from quickreduce_funcs import format_plot
        waves = np.polyval(coefs[::-1],self.p_x)
        fitlines = np.asarray(self.line_matches['lines'])
        fitlineloc = np.asarray(self.wm)
        fitheights = np.asarray(self.fm)
        dellines = np.asarray(self.all_wms)
        fitpix = np.asarray(self.line_matches['peaks_p'])
        fitwaves = np.polyval(coefs[::-1],fitpix)

        fig = plt.figure(figsize=(20., 25.),frameon=False)
        ax_lineplot_first = plt.subplot2grid((20, 16), (9, 0), colspan=16,rowspan=5, fig=fig)
        #ax_loglineplot = plt.subplot2grid((5, 4), (1, 0), colspan=4, fig=fig)
        ax_lineplot_second = plt.subplot2grid((20, 16), (15, 0), colspan=16, rowspan=5, fig=fig)
        axfitpts = plt.subplot2grid((20, 16), (0, 0), colspan=10, rowspan=4, fig=fig)#,sharex=True)
        axresid = plt.subplot2grid((20, 16), (5, 0), colspan=8, rowspan=3, fig=fig)
        axHisty = plt.subplot2grid((20, 16), (5, 8), colspan=2, rowspan=3, fig=fig)
        axcovar = plt.subplot2grid((20, 16), (4, 11), colspan=4, rowspan=4, fig=fig)
        axcovcb = plt.subplot2grid((20, 16), (4, 15), colspan=1, rowspan=4, fig=fig)
        axstats = plt.subplot2grid((20, 16), (0, 11), colspan=6, rowspan=3, fig=fig)
        #axfitpts = axresid.twinx()

        minlam,maxlam = waves.min(),waves.max()
        middlelam = int((maxlam+minlam)*0.5)

        for axi,lowlim,uplim,title in zip([ax_lineplot_first,ax_lineplot_second],\
                                          [minlam,middlelam],[middlelam,maxlam],\
                                          ["First Half of Range","Second Half of Range"]):
            axi.plot(waves, self.yspectra, 'b-')
            axi.plot(fitwaves, self.line_matches['peaks_h'], 'co', markersize=6, markeredgewidth=2, markerfacecolor='w', alpha=0.5)
            for w in dellines:
                axi.axvline(w, color='gray',linewidth=1 ,alpha=0.2)
            for w in fitlines:
                axi.axvline(w, color='r',linewidth=1, alpha=0.5)
            if self.mock_spec_w is not None:
                axi.fill_between(self.mock_spec_w, self.convd_mock_spec_f, y2=0., color='gray', alpha=0.1)
            axi.plot(fitlineloc, fitheights/2., 'r.', markersize=8, alpha=0.5)
            axi.set_xlim(left=lowlim, right=uplim)
            axi.set_ylim(0,1.1*np.max(self.yspectra[((waves>lowlim)&(waves<uplim))]))
            format_plot(axi, title=title, xlabel=r'Wavelength [$\mathrm{\AA}$]', ylabel='Counts', labelsize=16)


        axfitpts.plot(fitpix, fitlines-fitpix-coefs[0], 'r.',label='Fitted Points')
        highres_pix = np.arange(fitpix.min(), fitpix.max(), 0.1)
        coefs = np.asarray(coefs)
        axfitpts.plot(highres_pix, np.polyval(coefs[::-1],highres_pix)-highres_pix-coefs[0], 'b-',label='Fitted Line')
        format_plot(axfitpts, title="Fit versus Data", xlabel='Pixels', ylabel='Wavelength-(1*pix+{:0.01f})'.format(coefs[0])+r' [$\mathrm{\AA}$]', labelsize=16)
        axfitpts.legend(loc='best')

        # residuals = fitlines - np.polyval(coefs[::-1],fitpix)
        # axresid.plot(fitpix, residuals, 'r.')
        # axresid.plot(highres_pix, np.zeros(len(highres_pix)), 'k--')
        # format_plot(axresid, title="Residuals", xlabel='Pixels', ylabel=r'Line-Fit [$\mathrm{\AA}$]', labelsize=16)

        residuals = fitlines - np.polyval(coefs[::-1],fitpix)
        axresid.plot(np.polyval(coefs[::-1],fitpix), residuals, 'r.')
        axresid.plot(np.polyval(coefs[::-1],highres_pix), np.zeros(len(highres_pix)), 'k--')
        format_plot(axresid, title="Residuals", xlabel=r'Fit Wavelength [$\mathrm{\AA}$]', ylabel=r'Calibration-Fitted Peak [$\mathrm{\AA}$]', labelsize=16)

        axHisty.hist(residuals, bins=12, orientation='horizontal')
        axHisty.axhline(0., color='k',linestyle='--')
        axHisty.set_ylim(axresid.get_ylim())
        #axHisty.set_ylabel(None)
        axHisty.set_yticklabels([])
        #axHisty.set_title(r'Residuals $\mathrm{\AA}$',fontsize=16)
        axHisty.set_xlabel('Counts',fontsize=16)

        normd_cov = 0*cov.copy()
        for i,co in enumerate(coefs):
            for j,co2 in enumerate(coefs):
                normd_cov[i,j] = np.abs(cov[5-i,5-j]/(co*co2))
        # for i in np.arange(len(coefs)):
        #     for j in np.arange(i):
        #         normd_cov[i,j] = 0
        im = axcovar.imshow(np.log(normd_cov),origin='lower')#vmin=np.min(normd_cov),vmax=normd_cov[4,4],
        axcovar.set_xticklabels(['','a', 'b', 'c', 'd', 'e', 'f'])
        axcovar.set_yticklabels(['','a', 'b', 'c', 'd', 'e', 'f'])
        fig.colorbar(im, cax=axcovcb, orientation='vertical', shrink=0.8)
        format_plot(axcovar, title=r'Cov. of Fit: log$\left(\sigma_{x_ix_j}/x_ix_j\right)$', xlabel='Coefficient', ylabel='Coefficient', labelsize=16)

        axstats.set_frame_on(True)
        #axstats.set_xticklabels(None)
        axstats.set_xticks([])#None)
        #axstats.set_yticklabels(None)
        axstats.set_yticks([])#None)
        axstats.set_title("Residuals Statistics",fontsize=22)
        ax_settings = {'verticalalignment':'center', \
                       'transform': axstats.transAxes, 'fontsize':20 }  #'horizontalalignment':'center'
        axstats.text(0.1, 0.88, 'mean =\t\t   {:.3e}'.format(np.mean(residuals))+r' $\mathrm{\AA}$',**ax_settings)
        axstats.text(0.1, 0.68, 'median =\t  {:.3e}'.format(np.median(residuals))+r' $\mathrm{\AA}$',**ax_settings)
        axstats.text(0.1, 0.48, 'mean abs =\t {:.6f}'.format(np.mean(np.abs(residuals))) + r' $\mathrm{\AA}$', **ax_settings)
        axstats.text(0.1, 0.28, 'median abs =\t{:.6f}'.format(np.median(np.abs(residuals))) + r' $\mathrm{\AA}$', **ax_settings)
        axstats.text(0.1, 0.08, 'std dev. =\t    {:.06f}'.format(np.std(residuals))+r' $\mathrm{\AA}$',**ax_settings)

        if '/' in savename:
            plottitle = savename.split('/')[-1]
        elif '\\' in savename:
            plottitle = savename.split('\\')[-1]
        else:
            plottitle = savename
        if '.png' in plottitle:
            plottitle = plottitle.split('.')[0]
        plottitle.replace('_',' ')

        fig.suptitle(plottitle,fontsize=24)
        fig.savefig(savename, dpi=200)
        plt.close()
        del fig




def least_squares_fit(coefs,pixels,wavelengths,bounds):
    # fit 5th order polynomial to peak/line selections
    pixels = np.sort(pixels).astype(np.float64)
    waves = np.sort(wavelengths).astype(np.float64)
    try:
        # if bounds is None:
        #     params, pcov = curve_fit(fifth_order_poly, pixels, waves, p0=coefs, method='lm')
        # else:
        #     params, pcov = curve_fit(fifth_order_poly, pixels, waves, \
        #                              p0=coefs, bounds = bounds)

        guessed_waves = np.polyval(np.asarray(coefs)[::-1],pixels)
        dwaves = waves - guessed_waves

        # fit_poly = np.polynomial.polynomial.Polynomial.fit
        # if bounds is None:
        #     outseries, [resid, rank, sv, rcond] = fit_poly(pixels, dwaves, deg=5, full=True)
        #     params = outseries.convert().coef
        # else:
        #     outseries, [resid, rank, sv, rcond] = fit_poly(pixels, dwaves, deg=5, full=True, domain=bounds)
        #     params = outseries.convert().coef

        # print(sv,sv.shape)
        # cov = np.identity(len(params))*sv[::-1

        fit_poly = np.polyfit

        params, cov = fit_poly(pixels, dwaves, deg=5, full=False,cov=True)

        params = params[::-1] + np.asarray(coefs)
        outdwaves = waves- np.polyval(params[::-1],pixels)
        resid = np.dot(outdwaves,outdwaves)/len(outdwaves)

    except TypeError:
        print("Type error, fit failed, saving default")
        params = coefs
        cov = np.ones(shape=(5,5))*1.0e6
        resid = 1e6

    return params,cov, resid



