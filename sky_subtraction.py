from astropy.io import fits
import numpy as np
import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.interpolate import CubicSpline
from scipy.signal import medfilt, find_peaks
from scipy.signal.windows import gaussian as gaussian_kernel
from scipy.ndimage import median_filter
from collections import OrderedDict
from calibrations import Calibrations
from observations import Observations
import matplotlib.pyplot as plt


from quickreduce_funcs import smooth_and_dering
def gauss(lams, offset, mean, sig, amp):
    return offset + (amp * np.exp(-(lams - mean) * (lams - mean) / (2 * sig * sig))) / np.sqrt(
        2 * np.pi * sig * sig)

def linear_gauss(lams, offset, linear, mean, sig, amp):
    return offset + linear * (lams - lams.min()) + (
            amp * np.exp(-(lams - mean) * (lams - mean) / (2 * sig * sig))) / np.sqrt(
        2 * np.pi * sig * sig)

def sumd_gauss(lams, offset, mean, sig1, sig2, amp1, amp2):
    return gauss(lams, offset, mean, sig1, amp1) + gauss(lams, offset, mean, sig2, amp2)

def doublet_gauss(lams, offset, mean1, mean2, sig1, sig2, amp1, amp2):
    return gauss(lams, offset, mean1, sig1, amp1) + gauss(lams, offset, mean2, sig2, amp2)


def to_sigma(s_right,  nsigma,fwhm_to_sigma,s_left):
    return nsigma * fwhm_to_sigma * (s_right - s_left)

def replace_nans(flux_array,wave_pix_multiplier = 4):
    nanlocs = np.isnan(flux_array)
    outfluxarray = flux_array.copy()
    if np.any(nanlocs):
        background = flux_array.copy()
        background[nanlocs] = 0.
        background = median_filter(median_filter(background, size=371*wave_pix_multiplier,mode='nearest'), size=371*wave_pix_multiplier,mode='nearest')
        outfluxarray[nanlocs] = background[nanlocs]
    return outfluxarray

def subtract_sky(galflux,skyflux,gallams,galmask):
    pix_per_wave = int(np.ceil(1/np.nanmedian(gallams[1:]-gallams[:-1])))
    continuum_median_kernalsize = 371
    if int(np.ceil(pix_per_wave)) % 2 == 0:
        continuum_median_kernalsize = (continuum_median_kernalsize*pix_per_wave) + 1
    else:
        continuum_median_kernalsize = continuum_median_kernalsize * pix_per_wave

    npixels = len(galflux)
    pixels = np.arange(npixels).astype(float)
    masked = galmask.copy()

    galflux = replace_nans(galflux, wave_pix_multiplier=pix_per_wave)
    skyflux = replace_nans(skyflux, wave_pix_multiplier=pix_per_wave)

    gcont = median_filter(galflux, size=continuum_median_kernalsize,mode='nearest')
    scont = median_filter(skyflux, size=continuum_median_kernalsize,mode='nearest')
    gal_contsub = galflux - gcont
    sky_contsub = skyflux - scont
    median_sky_continuum_height = np.nanmedian(scont)
    s_peak_inds, s_peak_props = find_peaks(sky_contsub, height=(sky_contsub.max() / 10, None), width=(0.5*pix_per_wave, 8*pix_per_wave), \
                                           threshold=(None, None),
                                           prominence=(sky_contsub.max() / 5, None), wlen=int(24*pix_per_wave))

    g_peak_inds, g_peak_props = find_peaks(gal_contsub, height=(gal_contsub.max() / 10, None), width=(0.5*pix_per_wave, 8*pix_per_wave), \
                                           threshold=(None, None),
                                           prominence=(gal_contsub.max() / 5, None), wlen=int(24*pix_per_wave))

    g_peak_inds_matched = []
    if len(g_peak_inds) == 0:
        g_peak_inds, g_peak_props = find_peaks(gal_contsub, height=(gal_contsub.max() / 20, None), width=(0.5*pix_per_wave, 8*pix_per_wave), \
                                               threshold=(None, None),
                                               prominence=(gal_contsub.max() / 10, None), wlen=int(24*pix_per_wave))
        if len(g_peak_inds) == 0:
            return (gal_contsub + gcont), skyflux*(np.median(gcont/scont)), gcont, np.zeros(len(galflux)).astype(bool)

    for peak in s_peak_inds:
        ind = np.argmin(np.abs(gallams[g_peak_inds] - gallams[peak]))
        g_peak_inds_matched.append(g_peak_inds[ind])

    g_peak_inds_matched = np.asarray(g_peak_inds_matched).astype(int)

    # s_peak_fluxes = sky_contsub[s_peak_inds]
    # g_peak_fluxes = gal_contsub[g_peak_inds_matched]
    s_peak_fluxes = skyflux[s_peak_inds]
    g_peak_fluxes = galflux[g_peak_inds_matched]
    # differences = g_peak_fluxes - s_peak_fluxes
    # normd_diffs = differences/s_peak_fluxes
    # median_normd_diff = np.median(normd_diffs)
    peak_ratio = g_peak_fluxes / s_peak_fluxes
    median_ratio = np.median(peak_ratio)
    # print(peak_ratio, median_ratio)

    skyflux *= median_ratio
    scont = median_filter(skyflux, size=continuum_median_kernalsize,mode='nearest')
    sky_contsub = skyflux - scont

    sky_too_large = np.where(scont>gcont)[0]
    if len(sky_too_large) > 0.2*len(gal_contsub):
        adjusted_skyflux_ratio = np.median(gcont[sky_too_large]/scont[sky_too_large])
        skyflux *= adjusted_skyflux_ratio
        scont = median_filter(skyflux, size=continuum_median_kernalsize,mode='nearest')
        sky_contsub = skyflux - scont

    remaining_sky = skyflux.copy()

    s_peak_inds, s_peak_props = find_peaks(sky_contsub, height=(30, None), width=(0.1*pix_per_wave, 10*pix_per_wave), \
                                           threshold=(None, None),
                                           prominence=(10, None), wlen=int(101*pix_per_wave))
    g_peak_inds, g_peak_props = find_peaks(gal_contsub, height=(30, None), width=(0.1*pix_per_wave, 10*pix_per_wave), \
                                           threshold=(None, None),
                                           prominence=(10, None), wlen=int(101*pix_per_wave))

    outgal = gal_contsub.copy()
    line_pairs = []
    for ii in range(len(s_peak_inds)):
        pair = {}

        lam1 = gallams[s_peak_inds[ii]]
        ind1 = np.argmin(np.abs(gallams[g_peak_inds] - lam1))

        if np.abs(gallams[g_peak_inds[ind1]] - gallams[s_peak_inds[ii]]) > 3.0*pix_per_wave:
            continue

        pair['gal'] = {
            'peak': g_peak_inds[ind1], 'left': g_peak_props['left_ips'][ind1], \
            'right': g_peak_props['right_ips'][ind1] + 1, 'height': g_peak_props['peak_heights'][ind1], \
            'wheight': g_peak_props['width_heights'][ind1]
        }
        pair['sky'] = {
            'peak': s_peak_inds[ii], 'left': s_peak_props['left_ips'][ii], \
            'right': s_peak_props['right_ips'][ii] + 1, 'height': s_peak_props['peak_heights'][ii], \
            'wheight': s_peak_props['width_heights'][ii]
        }

        line_pairs.append(pair)

    sky_smthd_contsub = np.convolve(sky_contsub, [1 / 15., 3 / 15., 7 / 15., 3 / 15., 1 / 15.], 'same')
    gal_smthd_contsub = np.convolve(gal_contsub, [1 / 15., 3 / 15., 7 / 15., 3 / 15., 1 / 15.], 'same')
    for pair in line_pairs:
        need_to_mask = False
        g1_peak = pair['gal']['peak']
        s1_peak = pair['sky']['peak']

        itterleft = int(s1_peak)
        keep_going = True
        nextset = np.arange(1, 4*pix_per_wave).astype(int)
        while keep_going:
            if itterleft == 0:
                g_select = False
                s_select = False
            elif itterleft > 2*pix_per_wave:
                g_select = np.any(gal_smthd_contsub[itterleft - nextset] < gal_smthd_contsub[itterleft])
                s_select = np.any(sky_smthd_contsub[itterleft - nextset] < sky_smthd_contsub[itterleft])
            else:
                to_start = itterleft
                endcut = -3*pix_per_wave + to_start
                g_select = np.any(
                    gal_smthd_contsub[itterleft - nextset[:endcut]] < gal_smthd_contsub[itterleft])
                s_select = np.any(
                    sky_smthd_contsub[itterleft - nextset[:endcut]] < sky_smthd_contsub[itterleft])

            over_zero_select = ((gal_smthd_contsub[itterleft] > -10.) & (sky_smthd_contsub[itterleft] > -10.))
            if g_select and s_select and over_zero_select:
                itterleft -= 1
            else:
                keep_going = False

        itterright = int(s1_peak)
        keep_going = True
        nextset = np.arange(1, 4*pix_per_wave).astype(int)
        while keep_going:
            if (len(pixels) - itterright) == 1:
                g_select = False
                s_select = False
            elif (len(pixels) - itterright) >= 4*pix_per_wave:
                g_select = np.any(gal_smthd_contsub[itterright + nextset] < gal_smthd_contsub[itterright])
                s_select = np.any(sky_smthd_contsub[itterright + nextset] < sky_smthd_contsub[itterright])
            else:
                to_end = len(pixels) - itterright
                endcut = -4*pix_per_wave + to_end
                g_select = np.any(
                    gal_smthd_contsub[itterright + nextset[:endcut]] < gal_smthd_contsub[itterright])
                s_select = np.any(
                    sky_smthd_contsub[itterright + nextset[:endcut]] < sky_smthd_contsub[itterright])

            over_zero_select = ((gal_smthd_contsub[itterright] > -10.) & (sky_smthd_contsub[itterright] > -10.))
            if g_select and s_select and over_zero_select:
                itterright += 1
            else:
                keep_going = False

        slower_wave_ind = int(itterleft)
        supper_wave_ind = int(itterright) + 1
        extended_lower_ind = np.clip(slower_wave_ind - 10, 0, npixels - 1)
        extended_upper_ind = np.clip(supper_wave_ind + 10, 0, npixels - 1)

        g_distrib = gal_contsub[slower_wave_ind:supper_wave_ind].copy()
        min_g_distrib = g_distrib.min()
        g_distrib = g_distrib - min_g_distrib + 0.00001
        # g_lams = gallams[slower_wave_ind:supper_wave_ind]
        # g_dlams = gallams[slower_wave_ind+1:supper_wave_ind+1]-\
        #           gallams[slower_wave_ind:supper_wave_ind]
        # integral_g = np.dot(g_distrib,g_dlams)
        integral_g = np.sum(g_distrib)/pix_per_wave
        normd_g_distrib = g_distrib / integral_g

        s_distrib = sky_contsub[slower_wave_ind:supper_wave_ind].copy()
        min_s_distrib = s_distrib.min()
        s_distrib = s_distrib - min_s_distrib + 0.00001
        # s_lams = gallams[slower_wave_ind:supper_wave_ind]
        # s_dlams = gallams[slower_wave_ind+1:supper_wave_ind+1]-\
        #           gallams[slower_wave_ind:supper_wave_ind]
        # integral_s = np.dot(s_distrib,s_dlams)
        integral_s = np.sum(s_distrib)/pix_per_wave

        # local_vals = gal_contsub[slower_wave_ind-10*:supper_wave_ind]
        # dint = g_distrib*(1 - (integral_s / integral_g))
        ## Distrib is cont subtracted, so this requires masking if the original peak
        ## is at least twice as tall as the continuum
        if np.nanmax(s_distrib) > median_sky_continuum_height:
            need_to_mask = True

        if integral_s > (1.1 * integral_g):
            need_to_mask = True
            integral_s = (1.1 * integral_g)

        if integral_s > (30.0 + integral_g):
            need_to_mask = True
            integral_s = (30.0 + integral_g)

        if integral_g > (60.0 + integral_s):
            need_to_mask = True
            integral_s = integral_g - 60.0

        sky_g_distrib = normd_g_distrib * integral_s
        if len(sky_g_distrib) > 3:
            nzpad = 5
            nkern = 3
            subd = gal_contsub[slower_wave_ind:supper_wave_ind].copy() - sky_g_distrib
            zeropadded = np.pad(subd,(nzpad,nzpad),'constant',constant_values=0.)
            gkern = gaussian_kernel(nkern,nkern/6.,sym=True)
            norm_gkern = gkern/gkern.sum()
            removedlineflux = np.convolve(zeropadded, norm_gkern, 'same')[nzpad:-nzpad]
            del subd,zeropadded
        else:
            removedlineflux = (gal_contsub[slower_wave_ind:supper_wave_ind].copy() - sky_g_distrib)

        if slower_wave_ind > 100:
            prior_vals = gal_contsub[slower_wave_ind-100:supper_wave_ind-100].copy()
            prior_vals = prior_vals[np.bitwise_not(masked[slower_wave_ind-100:supper_wave_ind-100])]
            if len(prior_vals)>10:
                plqart, pmed, puqart = np.nanquantile(prior_vals,[0.25,0.5,0.75])
                prior_var = (puqart-plqart)*0.5
                removedlineflux = np.clip(removedlineflux,a_min=1.5*prior_vals.min(),a_max= 1.5*prior_vals.max())
                removedlineflux = np.clip(removedlineflux, a_min=pmed-3*prior_var, a_max=pmed+3*prior_var)

        doplots = False
        dips_low = np.any((gal_contsub[slower_wave_ind:supper_wave_ind] - sky_g_distrib) < (-60))
        all_above = np.all((gal_contsub[slower_wave_ind:supper_wave_ind]) > (-60))
        if (dips_low and all_above):
            need_to_mask = True
        if doplots:  # or (dips_low and all_above):
            speaks, sheights, swheights = gallams[int(s1_peak)], pair['sky']['height'], pair['sky']['wheight']
            gpeaks, gheights, gwheights = gallams[int(g1_peak)], pair['gal']['height'], pair['gal']['wheight']
            slefts, glefts = gallams[int(pair['sky']['left'])], gallams[int(pair['gal']['left'])]
            srights, grights = gallams[int(pair['sky']['right'])], gallams[int(pair['gal']['right'])]

            plt.subplots(1, 3)

            plt.subplot(131)
            plt.plot(gallams[extended_lower_ind:extended_upper_ind], \
                     sky_contsub[extended_lower_ind:extended_upper_ind], label='sky', alpha=0.4)
            plt.plot(speaks, sheights, 'k*', label='peaks')
            plt.plot(slefts, swheights, 'k>')
            plt.plot(srights, swheights, 'k<')
            plt.xlim(gallams[extended_lower_ind], gallams[extended_upper_ind])

            ymin, ymax = plt.ylim()
            plt.vlines(gallams[slower_wave_ind], ymin, ymax)
            plt.vlines(gallams[supper_wave_ind - 1], ymin, ymax)
            plt.legend(loc='best')

            plt.subplot(132)
            plt.plot(gallams[extended_lower_ind:extended_upper_ind], \
                     gal_contsub[extended_lower_ind:extended_upper_ind], label='gal', alpha=0.4)
            plt.plot(gpeaks, gheights, 'k*', label='peaks')
            plt.plot(glefts, gwheights, 'k>')
            plt.plot(grights, gwheights, 'k<')
            plt.xlim(gallams[extended_lower_ind], gallams[extended_upper_ind])
            ymin, ymax = plt.ylim()
            plt.vlines(gallams[slower_wave_ind], ymin, ymax)
            plt.vlines(gallams[supper_wave_ind - 1], ymin, ymax)
            plt.legend(loc='best')

            plt.subplot(133)
            plt.plot(gallams[slower_wave_ind:supper_wave_ind],
                     gal_contsub[slower_wave_ind:supper_wave_ind] - sky_g_distrib, label='new sub',
                     alpha=0.4)
            plt.plot(gallams[slower_wave_ind:supper_wave_ind], removedlineflux, label='new smth sub', alpha=0.4)

            plt.plot(gallams[slower_wave_ind:supper_wave_ind], sky_g_distrib, alpha=0.4,
                     label='transformed sky')
            plt.plot(gallams[extended_lower_ind:extended_upper_ind],
                     gal_contsub[extended_lower_ind:extended_upper_ind], alpha=0.2, label='gal')
            plt.plot(gallams[slower_wave_ind:supper_wave_ind], s_distrib, alpha=0.4, label='sky')

            plt.plot(speaks, sheights, 'k*', label='sky peak')
            plt.plot(slefts, swheights, 'k>')
            plt.plot(srights, swheights, 'k<')

            plt.plot(gpeaks, gheights, 'c*', label='gal peak')
            plt.plot(glefts, gwheights, 'c>')
            plt.plot(grights, gwheights, 'c<')
            # plt.plot(lams,gauss(lams,*gfit_coefs),label='galfit')
            plt.xlim(gallams[extended_lower_ind], gallams[extended_upper_ind])
            ymin, ymax = plt.ylim()
            plt.vlines(gallams[slower_wave_ind], ymin, ymax)
            plt.vlines(gallams[supper_wave_ind - 1], ymin, ymax)
            # min1,min2 = plt.ylim()
            # plt.ylim(min1,ymax)
            plt.legend(loc='best')

            plt.show()
            if (dips_low and all_above):
                print("That didn't go well")
        # 1/np.sqrt(2*np.pi*sig*sig)
        # print(*gfit_coefs,*sfit_coefs)
        outgal[slower_wave_ind:supper_wave_ind] = removedlineflux

        ## remove the subtracted sky from that remaining in the skyflux
        remaining_sky[slower_wave_ind:supper_wave_ind] = scont[slower_wave_ind:supper_wave_ind]
        maskbuffer = 0
        if need_to_mask:
            masked[(slower_wave_ind-maskbuffer):(supper_wave_ind+maskbuffer)] = True

    outgal = outgal + gcont - remaining_sky

    return outgal, remaining_sky, gcont, masked










def investigate_app_naming(self, cam):
    from scipy.signal import medfilt

    if len(self.final_calibration_coefs.keys()) == 0:
        self.get_final_wavelength_coefs()
    observation_keys = list(self.observations.observations.keys())
    first_obs = observation_keys[0]
    sci_filnum, throw, throw1, throw2 = self.observations.observations[first_obs]
    npixels = len(self.all_hdus[(cam, sci_filnum, 'science', None)].data)
    pixels = np.arange(npixels).astype(np.float64)
    pix2 = pixels * pixels
    pix3 = pix2 * pixels
    pix4 = pix3 * pixels
    pix5 = pix4 * pixels
    target_sky_pair = self.targeting_sky_pairs[cam]
    ##hack!
    scis = {}
    for obs in observation_keys[1:]:
        sci_filnum, ccalib, fcalib, comparc_ind = self.observations.observations[obs]
        sci_hdu = self.all_hdus.pop((cam, sci_filnum, 'science', None))

        if obs == observation_keys[1]:
            combined_table = Table(sci_hdu.data).copy()
            if self.twostep_wavecomparc:
                calib_filnum = fcalib
            else:
                calib_filnum = ccalib
            comparc_data = self.final_calibration_coefs[(cam, calib_filnum)]
            header = sci_hdu.header
        else:
            itter_tab = Table(sci_hdu.data)
            for col in itter_tab.colnames:
                combined_table[col] += itter_tab[col]

    sci_data = combined_table
    skyfibs = list(np.unique((list(target_sky_pair.values()))))

    cards = dict(header)
    for key, val in cards.items():
        if key[:5].upper() == 'FIBER':
            if val.strip(' \t').lower() == 'unplugged':
                fib = '{}{}'.format(cam, key[5:])
                if fib not in self.instrument.deadfibers:
                    skyfibs.append(fib)

    skyfits = {}
    plt.figure()
    mins, maxs = [], []

    # HACK!
    skyfibs = comparc_data.colnames
    for skyfib in skyfibs:
        a, b, c, d, e, f = comparc_data[skyfib]
        skylams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
        skyflux = np.array(
            sci_data[skyfib])  # medfilt(sci_data[skyfib] - medfilt(sci_data[skyfib], 371), 3)
        skyflux[np.isnan(skyflux)] = 0.
        plt.plot(skylams, skyflux, label=skyfib, alpha=0.4)
        skyfit = CubicSpline(x=skylams, y=skyflux, extrapolate=False)
        skyfits[skyfib] = skyfit

        mins.append(skylams.min())
        maxs.append(skylams.max())

    skyllams = np.arange(np.min(mins), np.max(maxs), 0.1).astype(np.float64)
    del mins, maxs

    master_skies = []
    meds = []
    for skyfib in skyfibs:
        skyfit = skyfits[skyfib]
        outskyflux = skyfit(skyllams)
        outskyflux[np.isnan(outskyflux)] = 0.
        # outskyflux[np.isnan(outskyflux)] = 0.
        med = np.nanmedian(outskyflux)
        meds.append(med)
        # corrected = smooth_and_dering(outskyflux)
        master_skies.append(outskyflux)

    # median_master_sky = np.median(master_skies, axis=0)
    # mean_master_sky = np.mean(master_skies, axis=0)
    # master_sky = np.zeros_like(mean_master_sky)
    # master_sky[:300] = mean_master_sky[:300]
    # master_sky[-300:] = mean_master_sky[-300:]
    # master_sky[300:-300] = median_master_sky[300:-300]
    master_sky = np.nanmedian(master_skies, axis=0)
    master_sky[np.isnan(master_sky)] = 0.
    # medmed = np.median(meds)
    # master_sky *= medmed
    # del meds, medmed

    masterfit = CubicSpline(x=skyllams, y=master_sky, extrapolate=False)
    plt.plot(skyllams, master_sky, 'k-', label='master', linewidth=4)
    plt.legend(loc='best')
    plt.show()
    nonzeros = np.where(master_sky > 0.)[0]
    first_nonzero_lam = skyllams[nonzeros[0]]
    last_nonzero_lam = skyllams[nonzeros[-1]]
    del nonzeros

    bin_by_tet = True
    if bin_by_tet:
        median_arrays1 = {int(i): [] for i in range(1, 9)}
        median_arrays2 = {int(i): [] for i in range(1, 9)}

        for ii, name in enumerate(skyfibs):
            tet = int(name[1])
            fib = int(name[2:4])
            if fib > 8:
                median_arrays2[tet].append(master_skies[ii])
            else:
                median_arrays1[tet].append(master_skies[ii])

        plt.figure()
        for (key1, arr1), (key2, arr2) in zip(median_arrays1.items(), median_arrays2.items()):
            if len(arr1) > 0:
                med1 = np.nanmedian(np.asarray(arr1), axis=0)
                plt.plot(skyllams, med1, label="{}_1".format(key1), alpha=0.4)
            if len(arr2) > 0:
                med2 = np.nanmedian(np.asarray(arr2), axis=0)
                plt.plot(skyllams, med2, label="{}_2".format(key2), alpha=0.4)
        plt.legend(loc='best')
        plt.show()

    print("\n\n\n\n\n")
    sums = []
    gal_is_sky = 0.
    sky_is_gal = 0.
    unp_is_sky = 0.
    unp_is_gal = 0.
    correct_s, correct_g = 0., 0.
    cutlow = 3161840
    cuthigh = 3161840
    for skyfib, spec in zip(skyfibs, master_skies):
        subs = spec[((skyllams > 5660) & (skyllams < 5850))]
        # subs = subs[np.bitwise_not(np.isnan(subs))]
        # subs = subs[subs>0.]
        summ = np.sum(subs)
        sums.append(summ)
        truename = "FIBER{}{:02d}".format(9 - int(skyfib[1]), 17 - int(skyfib[2:]))
        objname = sci_hdu.header[truename]
        if summ < cutlow:
            print(truename, skyfib, summ, 'sky', objname)
            if 'GAL' == objname[:3]:
                gal_is_sky += 1
            elif objname[0] == 'u':
                unp_is_sky += 1
            else:
                correct_s += 1
        elif summ > cuthigh:
            print(truename, skyfib, summ, 'gal', objname)
            if 'SKY' == objname[:3]:
                sky_is_gal += 1
            elif objname[0] == 'u':
                unp_is_gal += 1
            else:
                correct_g += 1

    sums = np.array(sums)
    plt.figure()
    plt.hist(sums, bins=60)
    plt.show()
    print(len(sums))
    print(np.where(sums < cutlow)[0].size, np.where(sums > cuthigh)[0].size)
    print(len(np.unique(list(target_sky_pair.values()))))
    print(gal_is_sky, unp_is_sky, correct_s, gal_is_sky + unp_is_sky + correct_s)
    print(sky_is_gal, unp_is_gal, correct_g, sky_is_gal + unp_is_gal + correct_g)
    for galfib, skyfib in target_sky_pair.items():
        a, b, c, d, e, f = comparc_data[galfib]
        gallams = a + b * pixels + c * pix2 + d * pix3 + e * pix4 + f * pix5
        galflux = np.asarray(sci_data[galfib])
        galflux[np.isnan(galflux)] = 0.

        lamcut = np.where((gallams < first_nonzero_lam) | (gallams > last_nonzero_lam))[0]
        galflux[lamcut] = 0.
        galflux[np.isnan(galflux)] = 0.

        skyflux = masterfit(gallams)
        skyflux[np.isnan(skyflux)] = 0.
        skyflux[lamcut] = 0.

        gcont = medfilt(galflux, 371)
        scont = medfilt(skyflux, 371)
        gal_contsub = galflux - gcont
        sky_contsub = skyflux - scont