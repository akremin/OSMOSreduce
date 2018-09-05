# run peak identifier and match lines to peaks
line_matches = {'lines': [], 'peaks_p': [], 'peaks_w': [], 'peaks_h': []}
est_features = [fifth_est[ii], fourth_est[ii], cube_est[ii], quad_est[ii], stretch_est[ii], shift_est[ii]]
xspectra = fifth_est[ii] * (p_x - Gal_dat.FINAL_SLIT_X_FLIP[ii]) ** 5 + fourth_est[ii] * (
            p_x - Gal_dat.FINAL_SLIT_X_FLIP[ii]) ** 4 + cube_est[ii] * (p_x - Gal_dat.FINAL_SLIT_X_FLIP[ii]) ** 3 + \
           quad_est[ii] * (p_x - Gal_dat.FINAL_SLIT_X_FLIP[ii]) ** 2 + stretch_est[ii] * (
                       p_x - Gal_dat.FINAL_SLIT_X_FLIP[ii]) + shift_est[ii]
fydat = f_x[::-1] - signal.medfilt(f_x[::-1], 171)  # used to find noise
fyreal = (f_x[::-1] - f_x.min()) / 10.0
peaks = argrelextrema(fydat, np.greater)  # find peaks
fxpeak = xspectra[peaks]  # peaks in wavelength
fxrpeak = p_x[peaks]  # peaks in pixels
fypeak = fydat[peaks]  # peaks heights (for noise)
fyrpeak = fyreal[peaks]  # peak heights
noise = np.std(np.sort(fydat)[:np.round(fydat.size * 0.5)])  # noise level
peaks = peaks[0][fypeak > noise]
fxpeak = fxpeak[fypeak > noise]  # significant peaks in wavelength
fxrpeak = fxrpeak[fypeak > noise]  # significant peaks in pixels
fypeak = fyrpeak[fypeak > noise]  # significant peaks height
for j in range(wm.size):
    line_matches['lines'].append(wm[j])  # line positions
    line_matches['peaks_p'].append(fxrpeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (in pixels)
    line_matches['peaks_w'].append(fxpeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (in wavelength)
    line_matches['peaks_h'].append(fypeak[np.argsort(np.abs(wm[j] - fxpeak))][0])  # closest peak (height)

# Pick lines for initial parameter fit
cal_states = {'Xe': True, 'Ar': False, 'HgNe': False, 'Ne': False}
fig, ax = plt.subplots(1)

# maximize window
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.subplots_adjust(right=0.8, left=0.05, bottom=0.20)

vlines = []
for j in range(wm.size):
    vlines.append(ax.axvline(wm[j], color='r', alpha=0.5))
line, = ax.plot(wm, np.zeros(wm.size), 'ro')
yspectra = (f_x[::-1] - f_x.min()) / 10.0
fline, = plt.plot(xspectra, yspectra, 'b', lw=1.5, picker=5)

browser = LineBrowser(fig, ax, est_features, wm, fm, p_x - Gal_dat.FINAL_SLIT_X_FLIP[ii], Gal_dat.FINAL_SLIT_X_FLIP[ii],
                      vlines, fline, xspectra, yspectra, peaks, fxpeak, fxrpeak, fypeak, line_matches, cal_states)
