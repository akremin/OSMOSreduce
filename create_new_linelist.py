from astropy.io import fits
import numpy as np
from calibration_funcs import air_to_vacuum
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from scipy import signal

from astropy.table import Table
salt = Table.read('./lamp_linelists/salt/ThAr.csv',format='ascii.csv')
iraf = Table.read('./lamp_linelists/iraf/thar.dat',format='ascii.basic',names=['Wavelength'])
salt_air = salt['Wavelength']
iraf_air = iraf['Wavelength']

import os
cal_coef_path = os.path.abspath("../../Onedrive - umich.edu/Research/M2FSReductions/A02/calibrations/")

pixels = np.arange(2048).astype(np.float64)
p2 = pixels*pixels
p3 = pixels*p2
p4 = pixels*p3
p5 = pixels*p4
def pixels_to_waves(coefs):
    a,b,c,d,e,f = coefs
    return a + b*pixels + c*p2 + d*p3 + e*p4 + f*p5
def get_me_highres(coefs,flux,highres):
    waves = pixels_to_waves(coefs)
    interper = CubicSpline(x=waves,y=flux,extrapolate=False)
    hr_flux = interper(highres)
    hr_flux[np.isnan(hr_flux)] = 0.
    return hr_flux




bcal1 = fits.open(os.path.join(cal_coef_path,'b_calibration_full-ThAr_11C_627_320926.fits'))
bcal2 = fits.open(os.path.join(cal_coef_path,'b_calibration_full-ThAr_11C_635_321096.fits'))
bcal1 = fits.open(os.path.join(cal_coef_path,'b_calibration_full-ThAr_11C_627_321017.fits'))
rcal1 = fits.open(os.path.join(cal_coef_path,'r_calibration_full-ThAr_11C_627_320659.fits'))
rcal2 = fits.open(os.path.join(cal_coef_path,'r_calibration_full-ThAr_11C_635_320659.fits'))
bcoefs1 = bcal1['CALIB COEFS']
bcoefs2 = bcal2['CALIB COEFS']
rcoefs2 = rcal2['CALIB COEFS']
rcoefs1 = rcal1['CALIB COEFS']

cal_spec_path = os.path.abspath("../../Onedrive - umich.edu/Research/M2FSReductions/A02/oneds/")
cal_spec_template = os.path.join(cal_spec_path,'{}_fine_comp_{}_A02_1d_bc.fits')
rcalspec1 = fits.open(cal_spec_template.format('r',627))
rcalspec2 = fits.open(cal_spec_template.format('r',635))
bcalspec2 = fits.open(cal_spec_template.format('b',635))
bcalspec1 = fits.open(cal_spec_template.format('b',627))
bspec1 = bcalspec1['FLUX']
bspec2 = bcalspec2['FLUX']
rspec1 = rcalspec1['FLUX']
rspec2 = rcalspec2['FLUX']



fibers = np.sort(bspec1.data.names)
highres = np.arange(4400,6600,0.0001)
highres_flux = np.zeros(len(highres),dtype=np.float64)
for fib in fibers:
    coef = bcoefs1.data[fib]
    dat = bspec1.data[fib]
    highres_flux += get_me_highres(coef,dat,highres)
    coef = bcoefs2.data[fib]
    dat = bspec2.data[fib]
    highres_flux += get_me_highres(coef,dat,highres)

highres_flux /= (2*len(fibers))
# plt.figure(); plt.plot(highres,highres_flux); plt.show()
from linebrowser import LineBrowser

iraf_vac = air_to_vacuum(iraf_air)
salt_vac = air_to_vacuum(salt_air)
# mock_coefs = (highres[0],highres[1]-highres[0],0.,0.,0.,0.)
# browser = LineBrowser(salt_vac, np.zeros(len(salt_vac)), highres_flux, mock_coefs, iraf_vac)
# browser.plot()
peaks, properties = signal.find_peaks(highres_flux, height=(2000, None), width=(1, 100000), \
                                                   threshold=(None, None),
                                                   prominence=(1000, None), wlen=10000000)  # find peaks
fxpeak = highres[peaks]  # peaks in wavelength
fypeak = highres_flux[peaks]  # peaks heights (for noise)
# noise = np.std(np.sort(highres_flux)[: (highres_flux.size // 2)])  # noise level
# significant_peaks = fypeak > noise
# peaks = peaks[significant_peaks]
# fxpeak = fxpeak[significant_peaks]  # significant peaks in wavelength
# fypeak = fypeak[significant_peaks]  # significant peaks height

pw,ph = fxpeak,fypeak

overlap = 0.2
to_delete = []
listw, listh = pw.tolist(), ph.tolist()
for ii, (w, h) in enumerate(zip(pw, ph)):
    if ii == 0 or ii == len(pw) - 1:
        continue
    if ii in to_delete:
        continue
    if np.abs(pw[ii + 1] - w) < overlap or np.abs(pw[ii - 1] - w) < overlap:
        dup_inds = np.where(np.abs(w - pw) < overlap)[0]
        dup_hs = ph[dup_inds]
        selected = np.argmax(dup_hs)
        selected_ind = list(dup_inds).pop(selected)
        for del_ind in dup_inds:
            to_delete.append(del_ind)

for ind in to_delete[::-1]:
    listw.pop(ind)
    listh.pop(ind)

tab = Table(data=[listw,listh,['']*len(listh),['y']*len(listh)],names=['Wavelength','Intensity','Comment','Use'])
tab.write("ThAr_{}{}.csv".format(cam,filnum),format='ascii.csv',overwrite=True)