import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from scipy.interpolate import CubicSpline
from scipy.ndimage import median_filter
from scipy.signal import find_peaks
from scipy.signal.windows import gaussian as gaussian_kernel
from scipy.ndimage import gaussian_filter

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

def find_continuum(spec, pix_per_wave, csize = 701, quant = 0.9):
    modspec = np.clip(spec,-100,np.nanquantile(spec,quant))
    nanlocs = np.where(np.bitwise_not(np.isnan(modspec)))[0]
    if len(nanlocs)>0 and nanlocs[0]>0:
        modspec[:nanlocs[0]] = modspec[nanlocs[0]]
    if len(nanlocs)>0 and nanlocs[-1]<(len(spec)-1):
        modspec[nanlocs[-1]:] = modspec[nanlocs[-1]]
    medfit = buffered_smooth(modspec, csize+2, csize, 'median')
    for attempt in range(0,100,2):
        devs = (modspec-medfit)
        dev_quant = np.nanquantile(devs,quant-(attempt/200.))
        modspec[devs>dev_quant] = medfit[devs>dev_quant] + dev_quant
        medfit = buffered_smooth(modspec, csize+10, csize, 'median')
    sig = 8 * pix_per_wave
    nzpad = int(4 * sig)
    cont = buffered_smooth(medfit, nzpad, sig, 'gaussian')
    return cont + np.nanquantile(spec-cont,0.16)

def buffered_smooth(arr,bufsize,smoothsize,smoothtype):
    zeropadded = np.pad(arr, (bufsize, bufsize), 'constant',
                        constant_values=(np.median(arr[:bufsize]), np.median(arr[-bufsize:])))
    if smoothtype.lower()[0] == 'g':
        cont = gaussian_filter(zeropadded, sigma=smoothsize, order=0)[bufsize:-bufsize]
    else:
        cont = median_filter(zeropadded, size=smoothsize, mode='nearest')[bufsize:-bufsize]
    return cont

def subtract_sky(galflux,skyflux,gallams,galmask):
    pix_per_wave = int(np.ceil(1/np.nanmedian(gallams[1:]-gallams[:-1])))
    continuum_median_kernalsize = 280#371

    if int(np.ceil(pix_per_wave)) % 2 == 0:
        continuum_median_kernalsize = (continuum_median_kernalsize*pix_per_wave) + 1
    else:
        continuum_median_kernalsize = continuum_median_kernalsize * pix_per_wave

    npixels = len(galflux)
    pixels = np.arange(npixels).astype(float)
    masked = galmask.copy()

    galflux = replace_nans(galflux, wave_pix_multiplier=pix_per_wave)
    skyflux = replace_nans(skyflux, wave_pix_multiplier=pix_per_wave)

    gcont = find_continuum(galflux.copy(), pix_per_wave=pix_per_wave, csize=continuum_median_kernalsize)
    scont = find_continuum(skyflux.copy(), pix_per_wave=pix_per_wave, csize=continuum_median_kernalsize)
    gal_contsub = galflux - gcont
    sky_contsub = skyflux - scont
    # plt.figure()
    # plt.plot(gallams,gcont,alpha=0.2,label='gcont')
    # plt.plot(gallams, scont, alpha=0.2, label='scont')
    # plt.plot(gallams, galflux, alpha=0.2, label='gal')
    # plt.plot(gallams, skyflux, alpha=0.2, label='sky')
    # plt.plot(gallams, gal_contsub, alpha=0.2, label='gal contsub')
    # plt.plot(gallams, sky_contsub, alpha=0.2, label='sky contsub')
    # plt.legend()
    # plt.figure()
    # plt.plot(gallams,galflux-skyflux)
    # plt.show()
    median_sky_continuum_height = np.nanmedian(scont)
    g_peak_inds, g_peak_props = find_peaks(gal_contsub, height=(gal_contsub.max() / 8, None),
                                           width=(0.5 * pix_per_wave, 8 * pix_per_wave), \
                                           threshold=(None, None),
                                           prominence=(gal_contsub.max() / 8, None), wlen=int(24 * pix_per_wave))


    if len(g_peak_inds) == 0:
        g_peak_inds, g_peak_props = find_peaks(gal_contsub, height=(gal_contsub.max() / 20, None),
                                               width=(0.5 * pix_per_wave, 8 * pix_per_wave), \
                                               threshold=(None, None),
                                               prominence=(gal_contsub.max() / 20, None), wlen=int(24 * pix_per_wave))
        if len(g_peak_inds) == 0:
            print("Couldn't identify any peaks, returning scaled sky for direct subtraction")
            return (gal_contsub + gcont), skyflux * (np.median(gcont / scont)), gcont, np.zeros(len(galflux)).astype(
                bool)

    for runnum in range(10):
        s_peak_inds, s_peak_props = find_peaks(sky_contsub, height=(sky_contsub.max() / 8, None), width=(0.5*pix_per_wave, 8*pix_per_wave), \
                                               threshold=(None, None),
                                               prominence=(sky_contsub.max() / 8, None), wlen=int(24*pix_per_wave))

        g_peak_inds_matched = []
        for peak in s_peak_inds:
            ind = np.argmin(np.abs(gallams[g_peak_inds] - gallams[peak]))
            g_peak_inds_matched.append(g_peak_inds[ind])

        g_peak_inds_matched = np.asarray(g_peak_inds_matched).astype(int)

        s_peak_fluxes = sky_contsub[s_peak_inds]
        g_peak_fluxes = gal_contsub[g_peak_inds_matched]
        s_peak_total_fluxes = skyflux[s_peak_inds]
        # g_peak_fluxes = galflux[g_peak_inds_matched]
        differences = g_peak_fluxes - s_peak_fluxes
        peak_ratios = differences / s_peak_total_fluxes
        # print(np.median(peak_ratios),peak_ratios)
        if np.median(peak_ratios) < 0.001:
            break
        # normd_diffs = differences/s_peak_fluxes
        # median_normd_diff = np.median(normd_diffs)
        # peak_ratios = g_peak_fluxes / s_peak_fluxes
        median_ratio = 1.0+np.median(peak_ratios)

        skyflux *= median_ratio
        scont = find_continuum(skyflux.copy(), pix_per_wave=pix_per_wave, csize=continuum_median_kernalsize)
        sky_contsub = skyflux - scont

    not_masked = np.bitwise_not(galmask)
    sky_too_large = np.where(scont[not_masked][50:-50]>gcont[not_masked][50:-50])[0]
    flux_too_large = np.where(skyflux[not_masked][50:-50] > galflux[not_masked][50:-50])[0]
    if len(sky_too_large) > 0.4*len(gal_contsub[not_masked][50:-50]) and skyflux.max() > 2000. and \
        len(flux_too_large) > 0.2*len(gal_contsub[not_masked][50:-50]):
        # plt.subplots(2,1,sharex=True)
        # plt.subplot(211)
        # plt.plot(gallams, skyflux,alpha=0.2,label='scaled orig sky')
        # plt.plot(gallams, galflux, alpha=0.2, label='orig gal')
        # plt.plot(gallams, scont, alpha=0.2, label='scaled cont sky')
        # plt.plot(gallams, gcont, alpha=0.2, label='cont gal')
        # plt.legend()
        # plt.subplot(212)
        # plt.plot(gallams, sky_contsub, alpha=0.2, label='scaled contsub sky')
        # plt.plot(gallams, gal_contsub, alpha=0.2, label='cont sub gal')
        # plt.legend()
        # plt.show()

        adjusted_skyflux_ratio = np.median(gcont[not_masked][50:-50][sky_too_large]/scont[not_masked][50:-50][sky_too_large])
        skyflux *= adjusted_skyflux_ratio
        scont = find_continuum(skyflux.copy(), pix_per_wave=pix_per_wave, csize=continuum_median_kernalsize)
        sky_contsub = skyflux - scont
        print("Needed to rescale the sky because the initial adjustment was too large: ", adjusted_skyflux_ratio)

    remaining_sky = skyflux.copy()
    sprom = np.quantile(sky_contsub, 0.8)
    gprom = np.quantile(gal_contsub, 0.8)
    s_peak_inds, s_peak_props = find_peaks(sky_contsub, height=(None, None), width=(0.1*pix_per_wave, 10*pix_per_wave), \
                                           threshold=(None, None),
                                           prominence=(sprom, None), wlen=24*pix_per_wave)
    g_peak_inds, g_peak_props = find_peaks(gal_contsub, height=(None, None), width=(0.1*pix_per_wave, 10*pix_per_wave), \
                                           threshold=(None, None),
                                           prominence=(gprom, None), wlen=24*pix_per_wave)

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

    # sky_smthd_contsub = np.convolve(sky_contsub, [1 / 15., 3 / 15., 7 / 15., 3 / 15., 1 / 15.], 'same')
    # gal_smthd_contsub = np.convolve(gal_contsub, [1 / 15., 3 / 15., 7 / 15., 3 / 15., 1 / 15.], 'same')
    sky_smthd_contsub = gaussian_filter(sky_contsub, sigma=0.25 * pix_per_wave, order=0)
    gal_smthd_contsub = gaussian_filter(gal_contsub, sigma=0.5 * pix_per_wave, order=0)
    sky_smthd_contsub = sky_smthd_contsub * sky_contsub.sum() / sky_smthd_contsub.sum()
    gal_smthd_contsub = gal_smthd_contsub * gal_contsub.sum() / gal_smthd_contsub.sum()
    # plt.subplots(2,1,sharex=True)
    # plt.subplot(211)
    # plt.plot(gallams, skyflux,alpha=0.2,label='scaled orig sky')
    # plt.plot(gallams, galflux, alpha=0.2, label='orig gal')
    # plt.plot(gallams, scont, alpha=0.2, label='scaled cont sky')
    # plt.plot(gallams, gcont, alpha=0.2, label='cont gal')
    # plt.legend()
    # plt.subplot(212)
    # plt.plot(gallams, sky_contsub, alpha=0.2, label='scaled contsub sky')
    # plt.plot(gallams, gal_contsub, alpha=0.2, label='cont sub gal')
    # plt.legend()
    # plt.show()

    print("Identified {} lines to subtract".format(len(line_pairs)))
    seen_before = np.zeros(len(outgal)).astype(bool)
    for pair in line_pairs:
        need_to_mask = False
        g1_peak = pair['gal']['peak']
        s1_peak = pair['sky']['peak']

        itterleft = int(np.min([s1_peak,g1_peak]))
        keep_going = True
        n_angs = 1
        nextset = np.arange(1, n_angs*pix_per_wave).astype(int)
        while keep_going:
            if itterleft < np.max([1,(n_angs*pix_per_wave)//2]):
                if itterleft < 0:
                    itterleft = 0
                g_select = False
                s_select = False
            elif itterleft > n_angs*pix_per_wave:
                g_select = np.any(gal_smthd_contsub[itterleft - nextset] < gal_smthd_contsub[itterleft])
                s_select = np.any(sky_smthd_contsub[itterleft - nextset] < sky_smthd_contsub[itterleft])
            else:
                to_start = itterleft
                endcut = -(n_angs-1)*pix_per_wave + to_start
                g_select = np.any(
                    gal_smthd_contsub[itterleft - nextset[:endcut]] < gal_smthd_contsub[itterleft])
                s_select = np.any(
                    sky_smthd_contsub[itterleft - nextset[:endcut]] < sky_smthd_contsub[itterleft])

            over_zero_select = True#((gal_smthd_contsub[itterleft] > -10.) & (sky_smthd_contsub[itterleft] > -10.))
            if (g_select or s_select) and over_zero_select:
                itterleft -= np.max([1,(n_angs*pix_per_wave)//2])
            else:
                keep_going = False

        itterright = int(np.max([s1_peak,g1_peak]))
        keep_going = True
        nextset = np.arange(1, n_angs*pix_per_wave).astype(int)
        while keep_going:
            if (len(pixels) - itterright) == 1:
                g_select = False
                s_select = False
            elif (len(pixels) - itterright) >= n_angs*pix_per_wave:
                g_select = np.any(gal_smthd_contsub[itterright + nextset] < gal_smthd_contsub[itterright])
                s_select = np.any(sky_smthd_contsub[itterright + nextset] < sky_smthd_contsub[itterright])
            else:
                to_end = len(pixels) - itterright - 1
                endcut = -n_angs*pix_per_wave + to_end
                g_select = np.any(
                    gal_smthd_contsub[itterright + nextset[:endcut]] < gal_smthd_contsub[itterright])
                s_select = np.any(
                    sky_smthd_contsub[itterright + nextset[:endcut]] < sky_smthd_contsub[itterright])

            over_zero_select = True#((gal_smthd_contsub[itterright] > -10.) & (sky_smthd_contsub[itterright] > -10.))
            if (g_select or s_select) and over_zero_select:
                itterright += np.max([1,(n_angs*pix_per_wave)//2])
            else:
                keep_going = False

        slower_wave_ind = int(itterleft)
        supper_wave_ind = int(itterright) + 1

        if np.any(seen_before[slower_wave_ind:supper_wave_ind]):
            print("some of these have already been seen")
            if np.sum(seen_before[slower_wave_ind:supper_wave_ind])>(supper_wave_ind-slower_wave_ind-2):
                continue
            locs = np.where(np.bitwise_not(seen_before[slower_wave_ind:supper_wave_ind]))[0]
            print("was: ",slower_wave_ind,supper_wave_ind)
            supper_wave_ind = slower_wave_ind + locs[-1] + 1
            slower_wave_ind += locs[0]
            print('now: ',slower_wave_ind,supper_wave_ind)
        if slower_wave_ind == supper_wave_ind:
            print("Lower and upper inds somehow were the same. Skipping this peak")
            print("was: ",slower_wave_ind,supper_wave_ind)
            continue

        seen_before[slower_wave_ind:supper_wave_ind] = True

        # extended_lower_ind = np.clip(slower_wave_ind - 10, 0, npixels - 1)
        # extended_upper_ind = np.clip(supper_wave_ind + 10, 0, npixels - 1)

        g_distrib = gal_contsub[slower_wave_ind:supper_wave_ind].copy()
        if np.any(g_distrib<0.):
            min_g_distrib = g_distrib.min()
        else:
            min_g_distrib = 0.
        g_distrib = g_distrib - min_g_distrib
        integral_g = np.sum(g_distrib)#/pix_per_wave
        normd_g_distrib = g_distrib / integral_g

        s_distrib = sky_contsub[slower_wave_ind:supper_wave_ind].copy()
        if np.any(s_distrib < 0.):
            min_s_distrib = s_distrib.min()
        else:
            min_s_distrib = 0.
        s_distrib = s_distrib - min_s_distrib
        integral_s = np.sum(s_distrib)#/pix_per_wave

        # local_vals = gal_contsub[slower_wave_ind-10*:supper_wave_ind]
        # dint = g_distrib*(1 - (integral_s / integral_g))
        ## Distrib is cont subtracted, so this requires masking if the original peak
        ## is at least twice as tall as the continuum
        # if np.nanmax(s_distrib) > median_sky_continuum_height:
        #     need_to_mask = True
        ## Only if it's twice the GALAXY continuum
        # if np.nanmax(s_distrib) > np.nanmedian(gcont[slower_wave_ind:supper_wave_ind]):
        #     need_to_mask = True

        mean_s_ppix = np.mean(s_distrib)
        mean_g_ppix = np.mean(g_distrib)
        if integral_s > (1.1 * integral_g):
            need_to_mask = True
            integral_s = (1.1 * integral_g)

        if integral_s < (0.9 * integral_g):
            need_to_mask = True
            integral_s = (0.9 * integral_g)

        if mean_s_ppix > (30.0 + mean_g_ppix):
            need_to_mask = True
            integral_s = (30.0*len(s_distrib)) + integral_g

        if mean_g_ppix > (30.0 + mean_s_ppix):
            need_to_mask = True
            integral_s = integral_g - 30.0*len(s_distrib)

        # if integral_s > 1000:
        #     print("lessgo")
        sky_g_distrib = normd_g_distrib * integral_s
        subd = gal_contsub[slower_wave_ind:supper_wave_ind].copy() - sky_g_distrib

        if len(subd) == 0:
            print(slower_wave_ind,supper_wave_ind,subd, gal_contsub[slower_wave_ind:supper_wave_ind].copy(), sky_g_distrib,normd_g_distrib, s_distrib,g_distrib)
            continue
        if np.max(subd) > 300. or np.min(subd) < -100.:
            need_to_mask = True

        if len(sky_g_distrib) > pix_per_wave:
            nkern = (0.5*pix_per_wave)
            nzpad = int(nkern * 5)
            # if supper_wave_ind+1 < len(gal_contsub):
            #     zeropadded = np.pad(subd,(nzpad,nzpad),'constant',constant_values=(outgal[slower_wave_ind-1],outgal[supper_wave_ind+1]))
            # else:
            #     zeropadded = np.pad(subd,(nzpad,nzpad),'constant',constant_values=(outgal[slower_wave_ind-1],outgal[supper_wave_ind]))
            zeropadded = np.pad(subd,(nzpad,nzpad),'constant',constant_values=(outgal[slower_wave_ind-1],0.))
            zpad_smthd = gaussian_filter(zeropadded,sigma=nkern, order=0)
            removedlineflux = zpad_smthd[nzpad:-nzpad] #* subd.sum() / np.sum(zpad_smthd[nzpad:-nzpad])
            del zeropadded,zpad_smthd#,subd
        else:
            removedlineflux = subd#gal_contsub[slower_wave_ind:supper_wave_ind].copy() - sky_g_distrib

        # if need_to_mask:
        #     plt.figure()
        #     plt.plot(gallams[slower_wave_ind:supper_wave_ind],gal_contsub[slower_wave_ind:supper_wave_ind],alpha=0.2,label='gal')
        #     if np.abs(np.sum(gal_contsub[slower_wave_ind:supper_wave_ind]-outgal[slower_wave_ind:supper_wave_ind])) < 0.1:
        #         plt.plot(gallams[slower_wave_ind:supper_wave_ind], outgal[slower_wave_ind:supper_wave_ind],alpha=0.2, label='out gal')
        #
        #     if min_g_distrib < 0.:
        #         plt.plot(gallams[slower_wave_ind:supper_wave_ind], g_distrib, alpha=0.2,label='gal adj')
        #     plt.plot(gallams[slower_wave_ind:supper_wave_ind],sky_contsub[slower_wave_ind:supper_wave_ind],alpha=0.2,label='sky')
        #     if min_s_distrib < 0.:
        #         plt.plot(gallams[slower_wave_ind:supper_wave_ind], s_distrib, alpha=0.2,label='sky adj')
        #     plt.plot(gallams[slower_wave_ind:supper_wave_ind], sky_g_distrib, alpha=0.2,label='sky nrmd')
        #     plt.plot(gallams[slower_wave_ind:supper_wave_ind], subd, alpha=0.2, label='subd')
        #     plt.plot(gallams[slower_wave_ind:supper_wave_ind], removedlineflux, alpha=0.2, label='smthd subd')
        #     plt.plot(gallams[slower_wave_ind:supper_wave_ind], gal_contsub[slower_wave_ind:supper_wave_ind]-sky_contsub[slower_wave_ind:supper_wave_ind], alpha=0.2, label='naive')
        #     plt.legend()
        #     plt.show()


        # 1/np.sqrt(2*np.pi*sig*sig)
        # print(*gfit_coefs,*sfit_coefs)
        outgal[slower_wave_ind:supper_wave_ind] = removedlineflux

        ## remove the subtracted sky from that remaining in the skyflux
        remaining_sky[slower_wave_ind:supper_wave_ind] = scont[slower_wave_ind:supper_wave_ind]
        maskbuffer = 0
        if need_to_mask:
            masked[(slower_wave_ind-maskbuffer):(supper_wave_ind+maskbuffer)] = True

    remaining_sky_devs = remaining_sky - scont
    twosig = np.nanquantile(np.abs(remaining_sky_devs),0.9)
    test_bools = ((remaining_sky_devs>(4*twosig)) & np.bitwise_not(masked))
    angstrom_buffer = 0.5
    if np.any(test_bools):
        print("Found points that didn't get removed properly",np.sum(test_bools)*pix_per_wave*2*angstrom_buffer)
        locs = np.where(test_bools)[0]
        for maskiter in np.arange(int(np.floor(-angstrom_buffer*pix_per_wave)),int(np.ceil(angstrom_buffer*pix_per_wave))+1):
            it_locs = np.clip(locs+int(maskiter),0,len(masked)-1)
            masked[it_locs] = True
        #
        # plt.figure()
        # plt.plot(gallams,remaining_sky,alpha=0.2,label='sky')
        # plt.plot(gallams[locs], remaining_sky[locs], '.', alpha=0.2, label='sky flagged')
        # plt.plot(gallams,outgal + gcont,alpha=0.2,label='gal')
        # plt.legend()
        # plt.figure()
        # plt.plot(gallams,remaining_sky_devs,alpha=0.2,label='sky')
        # plt.plot(gallams[locs], remaining_sky_devs[locs], '.', alpha=0.2, label='sky flagged')
        # plt.plot(gallams,outgal,alpha=0.2,label='gal')
        # plt.legend()
        # plt.show()

    outgal = outgal + gcont - remaining_sky

    return outgal, remaining_sky, gcont, scont, masked










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

        gcont = find_continuum(galflux.copy(), pix_per_wave=4., csize=701)
        scont = find_continuum(skyflux.copy(), pix_per_wave=4., csize=701)
        gal_contsub = galflux - gcont
        sky_contsub = skyflux - scont