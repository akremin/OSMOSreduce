from scipy.interpolate import CubicSpline
import numpy as np
from astropy.table import Table, Column
from astropy.io import fits
import matplotlib.pyplot as plt
from scipy.signal import medfilt



def flatten_data(fiber_fluxes,waves):
    waves_arr = []
    orig_fluxs_arr = []
    names = []
    for tet in range(1,9):
        for fib in range(1,17):
            name = 'r{}{:02d}'.format(tet,fib)
            a,b,c,d,e,f = waves[name]

            pixels = np.arange(len(fiberfluxes['r101']))
            lams = a + b*pixels+ c*np.power(pixels,2)+ d*np.power(pixels,3) +\
                   e*np.power(pixels, 4)+ f*np.power(pixels,5)
            waves_arr.append(lams)

            flux = fiberfluxes[name]
            orig_fluxs_arr.append(flux)
            names.append(name)

    waves_array = np.array(waves_arr)
    orig_flux_array = np.array(orig_fluxs_arr)
    del waves_arr, orig_fluxs_arr

    ## Create a grid of wavelengths to interpolate data onto
    min_lam, max_lam = np.min(waves_array[:,0]),np.max(waves_array[:,-1])
    min_shared_lam, max_shared_lam = np.max(waves_array[:,0]),np.min(waves_array[:,-1])
    outwaves = np.arange(min_shared_lam,max_shared_lam,0.2)

    ## Initialize the fitted wavelength grade arrays
    adj_flux_array = np.ndarray(shape=(orig_flux_array.shape[0],outwaves.shape[0]))
    adj_smth_flux_array = np.ndarray(shape=(orig_flux_array.shape[0],outwaves.shape[0]))

    ## Take in fibers row by row and interpolate them to the consistent wavelength grid
    for ii in range(128):
        ## Get the original data to be interpolated
        flux = orig_flux_array[ii,:]
        wave = waves_array[ii,:]

        ## Interpolate the data
        interpol = CubicSpline(x=wave,y=flux)
        outflux = interpol(outwaves)
        adj_flux_array[ii,:] = outflux

        ## Also save a smoothed version (because line peaks may differ slightly
        ## and cause issues in dividing out to get acceptances
        med = medfilt(outflux,193)
        adj_smth_flux_array[ii,:] = med


    ## Determine the brightest fiber and normalize on that
    sumd_smth_flux = np.sum(adj_smth_flux_array,axis=1)
    top = np.argmax(sumd_smth_flux)
    fiber_acceptances = sumd_smth_flux/sumd_smth_flux[top]

    ## Initiate final arrays
    normalized = np.ndarray(shape=adj_smth_flux_array.shape)
    final_flux_array = np.ndarray(shape=(orig_flux_array.shape))
    cols = []

    ## Fill final flattened images. One normalized (normalized) on lambda grid,
    ## the final_flux_array spline-fit back to the original pixel grid
    for ii in range(normalized.shape[0]):
        ## Normalized the data to the brightest fiber, then smooth out division incongruities
        flux = medfilt(adj_smth_flux_array[ii,:]/adj_smth_flux_array[top,:],193)
        normalized[ii, :] = flux

        ## Interpolate the normalized data
        interpol = CubicSpline(x=outwaves,y=flux,extrapolate=False)

        ## Refit the data back to the original pixel grid
        final_wave = waves_array[ii,:]
        final_flux = interpol(final_wave)

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


