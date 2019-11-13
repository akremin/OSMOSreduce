



def run_automated_calibration_wrapper(input_dict):
    return run_automated_calibration(**input_dict)

def run_automated_calibration(coarse_comp, complinelistdict):#, last_obs=None):
    assumed_pixscal_angstrom,calibration_wave_step = 1.,0.01
    wavelow,wavehigh = 4000,8000
    sumsqs,def_wavestart = 10., 5000.
    prom_quantile_psfcalc = 0.68
    npeaks_psfcalc =  6
    elements = ['Ne','Hg','Ar']

    ## Make sure the information is in astropy table format
    coarse_comp = Table(coarse_comp)

    cut_waves, cut_flux = create_simple_line_spectra(elements, complinelistdict, wave_low=wavelow, wave_high=wavehigh, \
                                                     clab_step=calibration_wave_step, atm_weights={'Ne': 1., 'Hg': 0.2, 'Ar': 0.8})

    calib_coefs = OrderedDict()

    for key in coarse_comp.colnames():  # ['r116','r101']:# keys:
        print("\n\n{}:\n".format(key))
        current_flux = np.array(coarse_comp[key].data, copy=True)

        sigma = get_psf(current_flux, step=assumed_pixscal_angstrom, prom_quantile=prom_quantile_psfcalc, npeaks=npeaks_psfcalc)
        print("PSF: {}".format(sigma))

        convd_cut_flux = gaussian_filter(cut_flux, sigma=sigma / calibration_wave_step, order=0)

        topb = 1.0
        if sumsqs < 0.4:
            topa, da = params[0], 100.
        else:
            topa, da = def_wavestart, 1000.

        best = fit_coarse_spectrum(current_flux, topa, da, topb, cut_waves, convd_cut_flux, \
                                   assumed_pixscal_angstrom, calibration_wave_step)
        sumsqs = best['metric']
        params = best['coefs']

        calib_coefs[key] = best.copy()

    return calib_coefs

def fit_coarse_spectrum(current_flux, topa, da, topb, cut_waves, convd_cut_flux, step,clab_step):
    topcor = 0.
    pix = np.arange(len(current_flux)).astype(np.float64)

    for a in np.arange(topa - da, topa + da, 0.2):
        test_waves = a + topb * pix
        wave_mask = ((test_waves >= cut_waves.min()) & (test_waves <= cut_waves.max()))
        test_waves = test_waves[wave_mask]
        if len(test_waves) == 0:
            continue
        else:
            cal_line_inds = np.round(test_waves / clab_step, 0).astype(int) - int(np.round(cut_waves.min() / clab_step))
            corcoef = np.corrcoef(current_flux[wave_mask], convd_cut_flux[cal_line_inds])[0, 1]
        if corcoef > topcor:
            topcor = corcoef
            topa = a

    fiber_wave_cut = ((cut_waves >= topa) & (cut_waves <= topa + topb * pix[-1]))
    fibcut_convd_flux = convd_cut_flux[fiber_wave_cut]
    fibcut_waves = cut_waves[fiber_wave_cut]
    peaks, props = find_peaks(current_flux, height=(1.4 * np.mean(current_flux), 1e9),
                              width=(2.355 / step, 6 * 2.355 / step))
    cpeaks, cparam = find_peaks(fibcut_convd_flux, height=(1.4 * np.mean(fibcut_convd_flux), 1e9),
                                width=(2.355 / clab_step, 12 * 2.355 / clab_step))
    peak_wavelengths = topa + topb * peaks
    dwaves_mat = np.abs(fibcut_waves[cpeaks].reshape((len(cpeaks), 1)) - peak_wavelengths.reshape((1, len(peaks))))

    dwaves = np.min(dwaves_mat, axis=0).flatten()

    wave_dist_cut_size = 20.
    nbad = np.sum(dwaves >= wave_dist_cut_size)
    print(nbad)
    if len(peaks) - nbad > 7:
        subset_peaks = peaks[dwaves < wave_dist_cut_size]
        subset_dwaves = dwaves[dwaves < wave_dist_cut_size]
    else:
        subset_peaks = peaks
        subset_dwaves = dwaves

    params, cov = curve_fit(pix_to_wave_explicit_coefs2, subset_peaks.astype(np.float64), subset_dwaves, p0=(0., 0., 0.))
    ddwaves = pix_to_wave(subset_peaks, *params) - subset_dwaves

    sumsqs = np.dot(ddwaves, ddwaves) / len(ddwaves)
    print("Fit Sumsq: {}".format(sumsqs))

    params = np.array([params[0], params[1], params[2], 0., 0., 0.]) + np.asarray([topa, topb, 0., 0., 0., 0.])

    print("Recovered: {}".format(params))

    best = {}
    best['metric'] = sumsqs
    best['nlines'] = len(subset_dwaves)
    best['clines'] = fibcut_waves[cpeaks]
    best['pixels'] = subset_peaks
    best['coefs'] = params
    return best
