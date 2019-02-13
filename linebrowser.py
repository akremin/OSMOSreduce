

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


def fifth_order_poly(p_x, zeroth, first, second, third, fourth, fifth):
    return zeroth + (first * p_x) + (second * np.power(p_x,2)) + (third * np.power(p_x,3)) + \
                   (fourth * np.power(p_x,4)) + (fifth * np.power(p_x,5))


class LineBrowser:
    def __init__(self, wm, fm, f_x, coefs, all_wms, bounds=None,edge_line_distance=0.):
        self.bounds = bounds
        self.coefs = np.asarray(coefs,dtype=np.float64)

        p_x = np.arange(len(f_x)).astype(np.float64)
        self.p_x = p_x
        xspectra = fifth_order_poly(p_x,*coefs)

        deviation = edge_line_distance
        good_waves = ((wm>(xspectra[0]-deviation))&(wm<(xspectra[-1]+deviation)))
        wm,fm = wm[good_waves],fm[good_waves]
        good_waves = ((all_wms>(xspectra[0]-deviation))&(all_wms<(xspectra[-1]+deviation)))
        all_wms = all_wms[good_waves]
        del good_waves

        fydat = f_x - signal.medfilt(f_x, 171)
        fyreal = (f_x - f_x.min()) / 10.0

        peaks,properties = signal.find_peaks(fydat)  # find peaks
        fxpeak = xspectra[peaks]  # peaks in wavelength
        fxrpeak = p_x[peaks]  # peaks in pixels
        fypeak = fydat[peaks]  # peaks heights (for noise)
        fyrpeak = fyreal[peaks]  # peak heights
        noise = np.std(np.sort(fydat)[: (fydat.size //2)])  # noise level
        significant_peaks = fypeak > noise
        peaks = peaks[significant_peaks]
        fxpeak = fxpeak[significant_peaks]  # significant peaks in wavelength
        fxrpeak = fxrpeak[significant_peaks]  # significant peaks in pixels
        fypeak = fyrpeak[significant_peaks]  # significant peaks height

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

        self.xspectra = xspectra
        self.yspectra = yspectra
        self.peaks = peaks
        self.peaks_w = fxpeak
        self.peaks_p = fxrpeak
        self.peaks_h = fypeak
        self.line_matches = line_matches
        self.mindist_el, = np.where(self.peaks_w == self.line_matches['peaks_w'][self.j])


        self.last = {'j':[],'lines':[],'peaks_h':[],'peaks_w':[],'peaks_p':[],'vlines':[],'wm':[],'fm':[]}
        self.initiate_browser()

    def initiate_browser(self):
        fig, ax = plt.subplots(1)

        # maximize window
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplots_adjust(right=0.8, left=0.05, bottom=0.20)

        for w in self.all_wms:
            ax.axvline(w, color='gray', alpha=0.2)

        vlines = []
        for w in self.wm:
            vlines.append(ax.axvline(w, color='r', alpha=0.5))

        fline, = plt.plot(self.xspectra, self.yspectra, 'b', picker=5)
        line, = ax.plot(self.wm, np.zeros(self.wm.size), 'ro')
        est_f, = ax.plot(self.wm, self.fm / 2.0, 'ro', picker=5)  # 5 points tolerance

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
        self.update_current()

    def update_current(self):
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

        xlim = self.ax.xaxis.get_view_interval()
        ylim = self.ax.yaxis.get_view_interval()
        if self.line_matches['lines'][self.j] > (xlim[1]-100) or self.line_matches['lines'][self.j] < (xlim[0]+100):
            self.reset_lims()
        self.fig.canvas.draw()

    def reset_lims(self):
        self.ax.set_xlim(self.line_matches['peaks_w'][self.j] - 100, self.line_matches['peaks_w'][self.j] + 500.0)
        xlims = self.ax.xaxis.get_view_interval()
        y_in = self.yspectra[np.where((self.xspectra > xlims[0]) & (self.xspectra < xlims[1]))]
        self.ax.set_ylim(np.min(y_in), np.max(y_in) * 1.1)

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
            while self.j != j:
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
        from quickreduce_funcs import format_plot
        waves = fifth_order_poly(self.p_x, *coefs)
        fitlines = np.asarray(self.line_matches['lines'])
        dellines = np.asarray(self.all_wms)
        fitpix = np.asarray(self.line_matches['peaks_p'])

        fig = plt.figure(figsize=(20., 25.))
        ax_lineplot = plt.subplot2grid((5, 4), (0, 0), colspan=4, fig=fig)
        ax_loglineplot = plt.subplot2grid((5, 4), (1, 0), colspan=4, fig=fig)
        ax_alllineplot = plt.subplot2grid((5, 4), (2, 0), colspan=4, fig=fig)
        axfitpts = plt.subplot2grid((5, 4), (3, 0), colspan=2, rowspan=2, fig=fig)
        axcovar = plt.subplot2grid((5, 4), (3, 2), colspan=2, rowspan=2, fig=fig)

        ax_lineplot.plot(waves, self.yspectra, 'b-')
        for w in fitlines:
            ax_lineplot.axvline(w, color='r', alpha=0.5)
        format_plot(ax_lineplot, title='Fit with used lines', xlabel='Wavelength', ylabel='Flux', labelsize=16)

        ax_loglineplot.semilogy(waves, self.yspectra, 'b-')
        # ax_loglineplot.set_yscale('log')
        for w in fitlines:
            ax_loglineplot.axvline(w, color='r', alpha=0.5)
        format_plot(ax_loglineplot, title='Log Fit with used lines', xlabel='Wavelength', ylabel='Log Flux',
                    labelsize=16)

        ax_alllineplot.plot(waves, self.yspectra, 'b-')
        for w in dellines:
            ax_alllineplot.axvline(w, color='gray', alpha=0.5)
        for w in fitlines:
            ax_alllineplot.axvline(w, color='r', alpha=0.5)
        format_plot(ax_alllineplot, title="Fit showing all lines", xlabel='Wavelength', ylabel='Flux', labelsize=16)

        axfitpts.plot(fitpix, fitlines-fitpix-fitpix.min(), 'r.',label='pts-{}-1*pix'.format(fitpix.min()))
        highres_pix = np.arange(fitpix.min(), fitpix.max(), 0.1)
        axfitpts.plot(highres_pix, fifth_order_poly(highres_pix, *coefs)-highres_pix-fitpix.min(), 'b-',label='fit-{}-1*pix'.format(fitpix.min()))
        format_plot(axfitpts, title="Fit versus Data", xlabel='Pixels', ylabel='Wavelength', labelsize=16)
        plt.legend(loc='best')

        normd_cov = cov.copy()
        for i,co in enumerate(coefs):
            for j,co2 in enumerate(coefs):
                normd_cov[i,j] /= (co*co2)
        axcovar.matshow(normd_cov)
        axcovar.set_xticklabels(['','a', 'b', 'c', 'd', 'e', 'f'])
        axcovar.set_yticklabels(['','a', 'b', 'c', 'd', 'e', 'f'])

        format_plot(axcovar, title='Covariance of Fit', xlabel='Param_j Unc', ylabel='Param_i Unc', labelsize=16)
        plt.savefig('{}.png'.format(savename), dpi=600)
        plt.close()


def least_squares_fit(coefs,pixels,wavelengths,bounds):
    # fit 5th order polynomial to peak/line selections
    pixels = np.sort(pixels).astype(np.float64)
    waves = np.sort(wavelengths).astype(np.float64)
    try:
        if bounds is None:
            params, pcov = curve_fit(fifth_order_poly, pixels, waves, p0=coefs, method='lm')
        else:
            params, pcov = curve_fit(fifth_order_poly, pixels, waves, \
                                     p0=coefs, bounds = bounds)
    except TypeError:
        print("Type error, fit failed, saving default")
        params =  coefs
        pcov = np.ones(shape=(5,5))*1.0e6

    return params,pcov