import matplotlib

matplotlib.use('Qt5Agg')
from scipy.signal import argrelextrema


import os
from scipy import fftpack
os_slash = os.path.sep
import pickle as pkl

from astropy.table import Table

from testopt import interactive_plot
from helper_funcs import *

#from calibrations import load_calibration_lines
import pickle as pkl
matplotlib.use('Qt5Agg')

import numpy as np
try:
    matplotlib.use('Qt5Agg')
except:
    matplotlib.use('Qt4Agg')

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import scipy.signal as signal
from scipy import fftpack
from scipy.optimize import curve_fit
from astropy.coordinates import SkyCoord
import astropy.units as u

from testopt import interactive_plot, LineBrowser, polyfour
from calibrations import load_calibration_lines
from scipy.signal import argrelextrema


#comp,thar,
def calibrate_pixels2wavelengths(camera,path_to_fitinfo=os.path.join('.','wavelength_calibrations')):
    cal_lamp = ['HgNe', 'Argon', 'Neon','Xenon']
    wm, fm, cal_states = load_calibration_lines(cal_lamp)
    print("Loaded calibrations: {}".format(cal_lamp))

    ## hack !

    apcut_data = {}
    with open('pkldump.pkl', 'rb') as pdump:
        apcut_data = pkl.load(pdump)

    comp = apcut_data['comp']['r'][628]
    thar = apcut_data['thar']['r'][627]

    print("Loaded spectra")
    clus_id = 'A11'

    ## end hack

    fibernames =  list(comp.keys())
    nspecs = len(fibernames)

    # create write file
    # f = open(cluster_dir + clus_id + '_stretchshift.tab', 'w')
    ## outfile we want: X_SLIT_FLIP     Y_SLIT     SHIFT     STRETCH     QUAD     CUBE     FOURTH    FIFTH    WIDTH
    current_date_time = np.datetime64('now','m').astype(int)-np.datetime64('2018-06-01T00:00','m').astype(int)

    # initialize polynomial arrays
    fifth, fourth, cube, quad, stretch, shift = np.zeros((6, len(comp.keys())))

    calib_files = []
    calib_ages = []
    if os.path.exists(path_to_fitinfo):
        for fil in os.listdir(path_to_fitinfo):
            if 'calib_wave_coefs' in fil and '_{}_'.format(camera) in fil:
                filname = fil.split('.dat')[0]
                dtime = filname.split('_')[-1]
                calib_ages.append(int(dtime))
                calib_files.append(fil)



    ###### Initialize
    if len(calib_files) > 0:
        most_recent_calib = calib_files[np.argmax(calib_ages)]
        fit_info = Table.read(os.path.join(path_to_fitinfo,most_recent_calib))
        fifth_est, fourth_est, cube_est, quad_est, stretch_est, shift_est = \
                                                fit_info[['fifth', 'fourth', 'cube', 'quad', 'stretch', 'shift']]
    else:
        shift_est = 3850 - (np.abs(nspecs//2-np.arange(nspecs))*(30/(nspecs//2)))
        stretch_est = 2*np.ones((len(comp.keys())))
        cube_est,quad_est,fifth_est, fourth_est = np.zeros((4, len(comp.keys())))

    spectra = {}

    print("Created estimates of parameters")

    ## End Initialize

    # estimate stretch,shift,quad terms with sliders for 2nd - all galaxies
    for i in range(len(fibernames)):
        fibername = fibernames[i]
        compflux = comp[fibername]
        tharflux = thar[fibername]
        spectra[fibername] = {}
        p_x = np.arange(len(compflux))

        print('Calibrating', i, 'of', stretch.size)

        f_x = compflux
        # stretch_est[i],shift_est[i],quad_est[i] = interactive_plot(p_x,f_x,stretch_est[i-1],shift_est[i-1]-(\
        # len(compflux)//2*stretch_est[0]-Gal_dat.FINAL_SLIT_X_FLIP[i-1]*stretch_est[i-1]),quad[i-1],cube[i-1],\
        # fourth[i-1],fifth[i-1],len(compflux)//2)
        reduced_slits = np.where(stretch != 0.0)
        stretch_est[i], shift_est[i], quad_est[i] = interactive_plot(p_x, f_x, stretch_est[i], shift_est[i],
                                                                     quad_est[i], cube_est[i], fourth_est[i],
                                                                     fifth_est[i], len(compflux)//2,
                                                                     wm, fm,cal_states=cal_states)
        est_features = [fifth_est[i], fourth_est[i], cube_est[i], quad_est[i], stretch_est[i], shift_est[i]]

        # run peak identifier and match lines to peaks
        line_matches = {'lines': [], 'peaks_p': [], 'peaks_w': [], 'peaks_h': []}
        xspectra = fifth_est[i] * (p_x - len(compflux)//2) ** 5 + fourth_est[i] * (
                    p_x - len(compflux)//2) ** 4 + cube_est[i] * (
                               p_x - len(compflux)//2) ** 3 + quad_est[i] * (
                               p_x - len(compflux)//2) ** 2 + stretch_est[i] * (
                               p_x - len(compflux)//2) + shift_est[i]
        fydat = f_x - signal.medfilt(f_x, 171)  # used to find noise
        fyreal = (f_x - f_x.min()) / 10.0
        peaks = argrelextrema(fydat, np.greater)  # find peaks
        fxpeak = xspectra[peaks]  # peaks in wavelength
        fxrpeak = p_x[peaks]  # peaks in pixels
        fypeak = fydat[peaks]  # peaks heights (for noise)
        fyrpeak = fyreal[peaks]  # peak heights
        noise = np.std(np.sort(fydat)[:np.round(fydat.size * 0.5).astype(int)])  # noise level
        peaks = peaks[0][fypeak > noise]
        fxpeak = fxpeak[fypeak > noise]  # significant peaks in wavelength
        fxrpeak = fxrpeak[fypeak > noise]  # significant peaks in pixels
        fypeak = fyrpeak[fypeak > noise]  # significant peaks height
        for j in range(wm.size):
            line_matches['lines'].append(wm[j])  # line positions
            line_matches['peaks_p'].append(
                fxrpeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (in pixels)
            line_matches['peaks_w'].append(
                fxpeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (in wavelength)
            line_matches['peaks_h'].append(
                fypeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (height)

        line_matches['lines']   = np.asarray(line_matches['lines'])
        line_matches['peaks_p'] = np.asarray(line_matches['peaks_p'])
        line_matches['peaks_w'] = np.asarray(line_matches['peaks_w'])
        line_matches['peaks_h'] = np.asarray(line_matches['peaks_h'])

        # Pick lines for initial parameter fit
        if i ==0:
            cal_states = {'Xe': True, 'Ar': False, 'HgNe': False, 'Ne': False}
        else:
            cal_states = {'Xe': True, 'Ar': False, 'HgNe': False, 'Ne': False}
        fig, ax = plt.subplots(1)

        # maximize window
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        plt.subplots_adjust(right=0.8, left=0.05, bottom=0.20)

        vlines = []
        for j in range(wm.size):
            vlines.append(ax.axvline(wm[j], color='r'))
        line, = ax.plot(wm, fm / 2.0, 'ro', picker=5)  # 5 points tolerance
        yspectra = (f_x - f_x.min()) / 10.0
        fline, = plt.plot(xspectra, yspectra, 'b', lw=1.5, picker=5)
        estx = quad_est[i] * (line_matches['peaks_p'] - len(compflux)//2) ** 2 + stretch_est[i] * (
                    line_matches['peaks_p'] - len(compflux)//2) + shift_est[i]

        browser = LineBrowser(fig, ax, est_features, wm, fm, p_x - len(compflux)//2,
                              len(compflux)//2, vlines, fline, xspectra, yspectra, peaks, fxpeak,
                              fxrpeak, fypeak, line_matches, cal_states)
        fig.canvas.mpl_connect('button_press_event', browser.onclick)
        fig.canvas.mpl_connect('key_press_event', browser.onpress)
        finishax = plt.axes([0.83, 0.85, 0.15, 0.1])
        finishbutton = Button(finishax, 'Finish', hovercolor='0.975')
        finishbutton.on_clicked(browser.finish)
        closeax = plt.axes([0.83, 0.65, 0.15, 0.1])
        button = Button(closeax, 'Replace (m)', hovercolor='0.975')
        button.on_clicked(browser.replace_b)
        nextax = plt.axes([0.83, 0.45, 0.15, 0.1])
        nextbutton = Button(nextax, 'Next (n)', hovercolor='0.975')
        nextbutton.on_clicked(browser.next_go)
        deleteax = plt.axes([0.83, 0.25, 0.15, 0.1])
        delete_button = Button(deleteax, 'Delete (j)', hovercolor='0.975')
        delete_button.on_clicked(browser.delete_b)
        fig.canvas.draw()
        plt.show()

        # fit 5th order polynomial to peak/line selections
        try:
            params, pcov = curve_fit(polyfour,
                                     (np.sort(browser.line_matches['peaks_p']) - len(compflux)//2),
                                     np.sort(browser.line_matches['lines']),
                                     p0=[shift_est[i], stretch_est[i], quad_est[i], 1e-8, 1e-12, 1e-12])
            cube_est[i] = params[3]
            fourth_est[i] = params[4]
            fifth_est[i] = params[5]
        except TypeError:
            params = [shift_est[i], stretch_est[i], quad_est[i], cube_est[i - 1], fourth_est[i - 1],
                      fifth_est[i - 1]]

        # make calibration and clip on lower anchor point. Apply to Flux as well
        wave_model = params[0] + params[1] * (p_x - len(compflux)//2) + params[2] * (
                    p_x - len(compflux)//2) ** 2 + params[3] * (
                                 p_x - len(compflux)//2) ** 3.0 + params[4] * (
                                 p_x - len(compflux)//2) ** 4.0 + params[5] * (
                                 p_x - len(compflux)//2) ** 5.0
        spectra[fibername]['wave'] = wave_model
        spectra[fibername]['wave2'] = wave_model[p_x >= np.sort(browser.line_matches['peaks_p'])[0]]

        flu = f_x[p_x >= np.sort(browser.line_matches['peaks_p'])[0]] - np.min(
            f_x[p_x >= np.sort(browser.line_matches['peaks_p'])[0]])
        flu = flu[::-1]
        Flux = flu / signal.medfilt(flu, 201)
        fifth[i], fourth[i], cube[i], quad[i], stretch[i], shift[i] = params[5], params[4], params[3], params[2], \
                                                                      params[1], params[0]
        plt.plot(spectra[fibername]['wave2'], Flux / np.max(Flux))
        plt.plot(wm, fm / np.max(fm), 'ro')
        for j in range(browser.wm.size):
            plt.axvline(browser.wm[j], color='r')
        plt.xlim(3800, 6000)
        try:
            plt.savefig(clus_id + os_slash + 'figs' + os_slash + str(i) + '.wave.png')
        except:
            os.mkdir(clus_id + os_slash + 'figs')
            plt.savefig(clus_id + os_slash + 'figs' + os_slash + str(i) + '.wave.png')
        plt.close()

if __name__ == '__main__':
    print("Running calibrate_pixels2wavelengths")
    calibrate_pixels2wavelengths('r')