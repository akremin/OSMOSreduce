from scipy.interpolate import CubicSpline
import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from scipy.optimize import curve_fit


def poly11(x, a, b, c, d, e, f, g, h, i, j, k, l):
    return (a + b * x + c * np.power(x, 2) + \
            d * np.power(x, 3) + e * np.power(x, 4) + f * np.power(x, 5) + \
            g * np.power(x, 6) + h * np.power(x, 7) + i * np.power(x, 8) + \
            j * np.power(x, 9) + k * np.power(x, 10) + l * np.power(x, 11))


def flatten_data(fiber_fluxes,waves):
    from scipy.ndimage import median_filter
    from scipy.signal.windows import gaussian
    interp_scale = 371
    nloops=1
    waves_arr = []
    orig_fluxs_arr = []
    names = []
    pixels = np.arange(len(fiber_fluxes[fiber_fluxes.colnames[0]])).astype(np.float64)
    for name in waves.colnames:
        a,b,c,d,e,f = waves[name]
        lams = a + b*pixels+ c*np.power(pixels,2)+ d*np.power(pixels,3) +\
               e*np.power(pixels, 4)+ f*np.power(pixels,5)
        waves_arr.append(lams)

        origflux = fiber_fluxes[name].copy()
        ## remove massive absorption line near end of solar spectrum
        prior_range = origflux[((lams>6760)&(lams<6880))]
        med_filler = np.median(prior_range)
        max_filler = np.max(prior_range)
        absorb_range_mask = ((lams > 6830) & (lams < 7080))
        nfills = np.sum(absorb_range_mask)
        midpoint = 250/2.
        quad = (-(max_filler-med_filler)*((np.arange(nfills).astype(float)-midpoint)**2)/(midpoint**2)) + (2*max_filler-med_filler)
        origflux[absorb_range_mask] = quad

        # medflux = median_filter(origflux, interp_scale, mode='constant', cval=np.quantile(origflux[-1 * interp_scale:],0.5))
        # flux = medflux.copy()
        # gwindow = gaussian(interp_scale, int(np.ceil(interp_scale / 6)))
        # gwindow = gwindow / gwindow.sum()
        # flux2 = np.convolve(medflux, gwindow, mode='same')
        # flux2[:interp_scale//2],flux2[-1*interp_scale//2:] = medflux[:interp_scale//2],medflux[-1*interp_scale//2:]
        #
        # flux = np.convolve(flux2, gwindow, mode='same')
        # flux[:interp_scale//2],flux[-1*interp_scale//2:] = medflux[:interp_scale//2],medflux[-1*interp_scale//2:]

        # for mult in range(1,10):
        #     int_val = 2 * (interp_scale // np.power(2, mult)) + 1
        #     gwindow = gaussian(int_val, int(np.ceil(int_val / 6)))
        #     gwindow = gwindow / gwindow.sum()
        #     flux2 = np.convolve(flux, gwindow, mode='same')
        #     flux[int_val//2:int_val] = flux2[int_val//2:int_val]
        #     flux[-1 * int_val:-1 * int_val//2] = flux2[-1 * int_val:-1 * int_val//2]
        # flux[:10], flux[-10:] = flux[10],flux[-10]


        coefs,cov = curve_fit(poly11,lams,origflux,p0=[5600,1,0.01,0,0,0,0,0,0,0,0,0])
        flux = poly11(lams,*coefs)

        orig_fluxs_arr.append(flux)

        # plt.figure()
        # plt.plot(lams,origflux,alpha=0.4)
        # plt.plot(lams,flux,alpha=0.4)
        # plt.show()
        names.append(name)

    waves_array = np.array(waves_arr)
    orig_flux_array = np.array(orig_fluxs_arr)
    del waves_arr, orig_fluxs_arr

    ## Create a grid of wavelengths to interpolate data onto
    min_lam, max_lam = np.min(waves_array[:,0]),np.max(waves_array[:,-1])
    min_shared_lam, max_shared_lam = np.max(waves_array[:,0]),np.min(waves_array[:,-1])
    outwaves = np.arange(min_shared_lam,max_shared_lam,0.1)

    ## Initialize the fitted wavelength grade arrays
    adj_flux_array = np.ndarray(shape=(orig_flux_array.shape[0],outwaves.shape[0]))
    adj_smth_flux_array = np.ndarray(shape=(orig_flux_array.shape[0],outwaves.shape[0]))

    ## Take in fibers row by row and interpolate them to the consistent wavelength grid
    for ii in range(len(names)):
        ## Get the original data to be interpolated
        flux_rough = orig_flux_array[ii,:].astype(np.float64)
        flux = flux_rough.copy()
        # flux = median_filter(flux_rough,interp_scale,mode='constant',cval=np.median(flux_rough[-1*interp_scale:]))
        # last_int = interp_scale
        # for mult in range(3):
        #     int_val = 2*(interp_scale//int(np.power(2,mult+2)))+1
        #     flux2 = median_filter(flux_rough,int_val,mode='constant',cval=np.median(flux[-1*int_val:]))
        #     flux[:last_int] = flux2[:last_int]
        #     flux[-1*last_int:] = flux2[-1*last_int:]
        #     last_int = int_val

        wave = waves_array[ii,:].astype(np.float64)

        ## Interpolate the data
        interpol = CubicSpline(x=wave,y=flux)
        outflux = interpol(outwaves)
        adj_flux_array[ii,:] = outflux

        ## Also save a smoothed version (because line peaks may differ slightly
        ## and cause issues in dividing out to get acceptances
        ## Want interp scale to be roughly half, but need to force it to be odd for medfiltering
        flux = outflux.copy()
        med = flux
        # med = median_filter(flux, interp_scale, mode='constant', cval=np.quantile(flux[-1 * interp_scale:],0.25))
        # for mult in range(3):
        #     int_val = 2 * (interp_scale // int(np.power(2, mult + 2))) + 1
        #     flux2 = median_filter(flux, int_val, mode='constant', cval=np.median(flux[-1 * int_val:]))
        #     med[:int_val] = flux2[:int_val]
        #     med[-1 * int_val:] = flux2[-1 * int_val:]
        adj_smth_flux_array[ii,:] = med


    ## Determine the brightest fiber and normalize on that
    # med_smth_flux = np.median(adj_smth_flux_array[:,200:-200],axis=1)
    # median = np.median(med_smth_flux)
    # top = np.argwhere(med_smth_flux==median)[0][0]
    # sumd_smth_flux = np.median(adj_smth_flux_array, axis=1)
    # fiber_acceptances = (sumd_smth_flux/sumd_smth_flux[top])*(med_smth_flux/median)

    sumd_smth_flux = np.median(adj_smth_flux_array, axis=1)
    top = np.argmax(sumd_smth_flux)
    fiber_acceptances = (sumd_smth_flux/sumd_smth_flux[top])

    ## Initiate final arrays
    normalized = np.ndarray(shape=adj_smth_flux_array.shape)
    final_flux_array = np.ndarray(shape=(orig_flux_array.shape))
    cols = []

    ## Fill final flattened images. One normalized (normalized) on lambda grid,
    ## the final_flux_array spline-fit back to the original pixel grid
    for ii in range(normalized.shape[0]):
        ## Normalized the data to the brightest fiber, then smooth out division incongruities
        outflux = adj_smth_flux_array[ii,:]/adj_smth_flux_array[top,:]
        ninterp = (2*interp_scale)+1
        medflux = median_filter(outflux,size=ninterp,mode='constant',cval=np.quantile(outflux[-1 * ninterp:],0.25))
        #
        # int_val = (4 * interp_scale) + 1
        # gwindow = gaussian(int_val, int(np.ceil(int_val / 6)))
        # gwindow = gwindow / gwindow.sum()
        # flux2 = np.convolve(medflux, gwindow, mode='same')
        # flux2[:int_val//2],flux2[-1*int_val//2:] = medflux[:int_val//2],medflux[-1*int_val//2:]
        # int_val = (4 * interp_scale) + 1
        # gwindow = gaussian(int_val, int(np.ceil(int_val / 6)))
        # gwindow = gwindow / gwindow.sum()
        # flux = np.convolve(flux2, gwindow, mode='same')
        # flux[:int_val//2],flux[-1*int_val//2:] = medflux[:int_val//2],medflux[-1*int_val//2:]
        # for mult in range(16):
        #     int_val = 2 * (interp_scale // np.power(2, mult)) + 1
        #     gwindow = gaussian(int_val, int(np.ceil(int_val / 6)))
        #     gwindow = gwindow / gwindow.sum()
        #     flux2 = np.convolve(flux, gwindow, mode='same')
        #     flux[int_val//2:int_val] = flux2[int_val//2:int_val]
        #     flux[-1 * int_val:-1 * int_val//2] = flux2[-1 * int_val:-1 * int_val//2]
        # flux[:100], flux[-100:] = flux[100],flux[-100]
        # flux = medfilt(adj_smth_flux_array[ii,:]/adj_smth_flux_array[top,:],interp_scale)
        flux = outflux
        normalized[ii, :] = flux

        ## Interpolate the normalized data
        interpol = CubicSpline(x=outwaves,y=flux,extrapolate=False)

        ## Refit the data back to the original pixel grid
        final_wave = waves_array[ii,:]
        final_flux = interpol(final_wave)

        #################### Short term hack
        # plt.figure()
        # plt.plot(outwaves,adj_smth_flux_array[ii,:]/np.median(adj_smth_flux_array[top,:]),label='Orig',alpha=0.2)
        # plt.plot(outwaves,adj_smth_flux_array[top,:]/np.median(adj_smth_flux_array[top,:]),label='Divisor',alpha=0.2)

        # plt.plot(outwaves, adj_smth_flux_array[ii, :] / adj_smth_flux_array[top, :], label='Ratiod', alpha=0.4)
        # plt.plot(outwaves, medflux, label='Med {}'.format(interp_scale), alpha=0.4)
        # plt.plot(outwaves, flux, label='Gaus Smthd {}'.format(interp_scale), alpha=0.4)
        # plt.plot(final_wave, final_flux, label='Interpd', alpha=0.4)
        # # plt.ylim(0.75, 0.9)
        # plt.legend()
        # plt.show()
        ##############################################################3

        ## Data was only fit in shared wavelength regime.
        ## For the outer pixels, give a linear normalization from the last fit value
        ## change lienarly outward to the mean normalization value at the ends (first or last pixel)
        nanlocs = np.where(np.isnan(final_flux))[0]

        # smoothed_orig = medfilt(orig_flux_array[ii, :], 193)
        # norm_smth_orig = smoothed_orig/fiber_acceptances[ii]

        midpt = (len(final_flux)//2)
        lower_nanlocs = nanlocs[nanlocs<midpt]
        higher_nanlocs = nanlocs[nanlocs > midpt]
        if len(lower_nanlocs)>1:
            lower_lastnan = lower_nanlocs[-1]
            lower_first_val = final_flux[lower_lastnan+1]
            lower_slope = (lower_first_val - fiber_acceptances[ii])/(lower_lastnan+1)
            if lower_slope < 0:
                lower_fill_values = np.ones(len(lower_nanlocs)).astype(float)*lower_first_val
            else:
                lower_fill_values = lower_nanlocs * lower_slope + fiber_acceptances[ii]
        elif len(lower_nanlocs)==1:
            lower_lastnan = lower_nanlocs[0]
            lower_first_val = final_flux[lower_lastnan+1]
            lower_fill_values = np.array([lower_first_val])
        else:
            lower_fill_values = np.array([])
        if len(higher_nanlocs)>1:
            higher_firstnan = higher_nanlocs[0]
            higher_lastval = final_flux[higher_firstnan-1]
            higher_slope = (fiber_acceptances[ii] - higher_lastval ) / (len(final_flux)-higher_firstnan-1)
            if higher_slope > 0.:
                higher_slope = 0.
            higher_fill_values = (higher_nanlocs-higher_nanlocs[0])*higher_slope + higher_lastval
        elif len(higher_nanlocs)==1:
            higher_firstnan = higher_nanlocs[0]
            higher_lastval = final_flux[higher_firstnan-1]
            higher_fill_values = np.array([higher_lastval])
        else:
            higher_fill_values = np.array([])
        fill_values = np.append(lower_fill_values,higher_fill_values)
        final_flux[nanlocs] = fill_values

        ## Store outputs
        cols.append(Column(data=final_flux,name=names[ii]))
        final_flux_array[ii,:] = final_flux

    # plt.figure()
    # plt.imshow(normalized, 'gray', aspect='auto', origin='lowerleft')#[:,:150])#[:,150:-150])
    # plt.figure()
    # plt.imshow(final_flux_array, 'gray', aspect='auto', origin='lowerleft')#[:,:150])#[:,150:-150])
    # plt.show()
    # plt.figure()
    # plt.imshow(adj_flux_array,  'gray', aspect='auto', origin='lowerleft')
    # plt.show()
    # plt.figure()
    # plt.imshow(adj_smth_flux_array,  'gray', aspect='auto', origin='lowerleft')
    # plt.figure()
    # plt.imshow(adj_flux_array-adj_smth_flux_array,  'gray', aspect='auto', origin='lowerleft')
    # plt.figure()
    # plt.imshow(normalized, 'gray', aspect='auto', origin='lowerleft')
    # plt.colorbar()
    # plt.show()

    final_table = Table(cols)
    return final_table, final_flux_array


if __name__ == '__main__':
    fiberfluxes = fits.open('../../odrive/Research/M2FSReductions/A02/oneds/r_twiflat_0591_A02_1d.bc.fits')[1].data
    waves = fits.open('./calibration_interactive_r_11C_628_130833.fits')[1].data
    final_table, final_flux_array = flatten_data(fiberfluxes,waves)
    plt.figure()
    plt.imshow(final_flux_array, 'gray', aspect='auto', origin='lowerleft')#[:,:150])#[:,150:-150])
    plt.show()


