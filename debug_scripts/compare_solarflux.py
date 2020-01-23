
from astropy.table import Table
from astropy.io import fits
import os
import numpy as np
import matplotlib.pyplot as plt
from pyM2FS.calibration_helper_funcs import air_to_vacuum
from scipy.signal import medfilt
from scipy.interpolate import CubicSpline


cams = ['r','b']


waverange = np.arange(4274,6319,0.1)
calib_filnum = 1313

solar = Table.read("./lamp_linelists/SolarFluxAtlas.csv",format='ascii.csv')
twifluxes = {}
calib_coefs = {}

#filnums = range(1319,1330+1)
# flat_loc = "../data/B09/oneds/"
# for fil in os.listdir(flat_loc):
#     if 'twiflat' in fil and int(fil.split('_')[2]) in filnums:
#         flat = Table(fits.open('{}{}'.format(flat_loc,fil))['FLUX'].data)
#         twifluxes[(fil[0],int(fil.split('_')[2]))] = flat

filnums = [None]
flat_loc = "../data/B09/final_oned/"
for fil in os.listdir(flat_loc):
    if 'twiflat' in fil:
        flat = Table(fits.open('{}{}'.format(flat_loc,fil))['FLUX'].data)
        twifluxes[(fil[0],None)] = flat



matches = {'r':[],'b':[]}
for fil in os.listdir('../data/B09/calibrations/'):
    if 'calibration_full-ThAr' in fil and str(calib_filnum) in fil:
        matches[fil[0]].append(int(fil.split('_')[-1].split('.')[0]))
newest_r = np.max(matches['r'])
newest_b = np.max(matches['b'])
calib_coefs['r'] = Table(fits.open('../data/B09/calibrations/{}'.format('{}_calibration_full-ThAr_11J_{}_{}.fits'.format('r',calib_filnum,newest_r)))['CALIB COEFS'].data)
calib_coefs['b'] = Table(fits.open('../data/B09/calibrations/{}'.format('{}_calibration_full-ThAr_11J_{}_{}.fits'.format('b',calib_filnum,newest_b)))['CALIB COEFS'].data)







pix1 = np.arange(2048).astype(np.float64)
pix2 = pix1*pix1
pix3 = pix2*pix1
pix4 = pix2*pix2
pix5 = pix3*pix2
def wave(calcoefs):
    a,b,c,d,e,f = calcoefs
    return a + pix1*b + pix2*c + pix3*d + \
            pix4*e + pix5*f


all_subd_flux_tabs = {}
subd_flux_tabs = {}
subd_flux_uniform = []
wave_calibs_tabs = {}

for cam in cams:
    for filnum in filnums:
        flx_tab = twifluxes[(cam,filnum)]
        if filnum == filnums[0]:
            subd_flux_tab = flx_tab.copy()
            wave_calibs = flx_tab.copy()
            calib_coef = calib_coefs[cam]
            for fib in flx_tab.colnames:
                subd_flux_tab[fib] = subd_flux_tab[fib]*0.
                cal_ceof = calib_coef[fib]
                wav = wave(cal_ceof)
                wave_calibs[fib] = wav
            wave_calibs_tabs[cam] = wave_calibs
        cur_subd_flux_tab = flx_tab.copy()
        for fib in flx_tab.colnames:
            flx = flx_tab[fib]
            subd = flx#/flx.max()
            subd = subd - medfilt(subd,2*int(len(subd)/10) + 1)
            subd_flux_tab[fib] += subd
            cur_subd_flux_tab[fib] = subd
            interpolator = CubicSpline(wave_calibs[fib],subd)
            interpolated_fluxes = interpolator(waverange,extrapolate=False)
            subd_flux_uniform.append(interpolated_fluxes)

        all_subd_flux_tabs[(cam,filnum)] = cur_subd_flux_tab
    subd_flux_tabs[cam] = subd_flux_tab


plt.figure()
for cam in ['r','b']:
    for fib in subd_flux_tabs[cam].colnames:
        wav = wave_calibs_tabs[cam][fib]
        subd = subd_flux_tabs[cam][fib]
        plt.plot(wav,subd/np.max(np.abs(subd)),alpha=0.2)

solar_vac = air_to_vacuum(solar['Wave'])
xmin,xmax = plt.xlim()
cut = np.where((solar_vac > xmin) & (solar_vac<xmax))
cut_vac = solar_vac[cut]
smoothing_scale = 2000
cut_flx = np.convolve(solar['RawFlux'],np.ones(smoothing_scale)/smoothing_scale,mode='same')[cut]
# cut_flx = np.array(solar['RawFlux'])[cut]
divisor = len(cut_flx) // (2*len(subd))
padd = int((divisor * np.ceil(len(cut_flx)/divisor) ) - len(cut_flx))
cut_flx = np.concatenate([np.array([cut_flx[0]]*padd),cut_flx])
cut_vac = np.concatenate([np.array([cut_vac[0]]*padd),cut_vac])
cut_flx = np.mean(cut_flx.reshape((-1,divisor)),axis=1)
cut_vac = np.mean(cut_vac.reshape((-1,divisor)),axis=1)
cut_subd = cut_flx - medfilt(cut_flx,2*int(len(cut_flx)/10) + 1)
plt.plot(cut_vac,cut_subd/np.max(np.abs(cut_subd)),'k-',alpha=1.0)
plt.show()