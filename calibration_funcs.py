
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
##  a zoom in window
##  mutlicursor
## And checkboxes
## and radio buttons
from matplotlib.widgets import MultiCursor
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider, Button

from astropy.table import Table

from scipy.signal import medfilt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
import numpy as np
from scipy.signal import argrelmax
from collections import OrderedDict
deltat = np.datetime64('now' ,'m').astype(int) -np.datetime64('2018-06-01T00:00' ,'m').astype(int)
print(deltat)



import numpy as np

def run_automated_calibration(coarse_comp, complinelistdict, last_obs=None):
    calibrationline_fluxcut = 0.
    n_attempts_beforebad = 100
    n_resolution_iters = 100

    # elements = ['Hg', 'Ne']
    euc_tolerance_perpix = 0.08  # 0.14
    frac_reduction, stepfrac = 0.6, 0.1

    ## 5063.36264  9.98328419e-01 -1.51722887e-06
    abound_mean, abound_hw = 5000.0, 600.0  # 5063.36264, 40.0
    bbound_mean, bbound_hw = 1.0, 0.1  # 9.98328419e-01, 0.01
    cbound_mean, cbound_hw = 0.0, 2.0e-5  # -1.51722887e-06, 2.0e-6#

    ## Make sure the information is in astropy table format
    coarse_comp = Table(coarse_comp)

    # select_complinelistdict = {element: complinelistdict[element] for element in elements if element in complinelistdict.keys()}
    # wm, fm = get_wm_fm(select_complinelistdict, cut=calibrationline_fluxcut)
    wm, fm = get_wm_fm(complinelistdict, cut=calibrationline_fluxcut)

    mat_wm = wm.reshape((wm.size, 1)).T

    frac = 1.0 / frac_reduction
    specs_to_run = np.array(coarse_comp.colnames)  # [::20]:['r216','r301','r302','r309','r310','r311','r312']
    calib_coefs = OrderedDict()
    for attempt in range(n_attempts_beforebad):
        last_good = {'euc': 1e8, 'a': abound_mean, 'b': bbound_mean, 'c': cbound_mean}
        n_baddist_skip = int(attempt)
        bad_specs = []
        for specname in specs_to_run:
            specflux, pix, ph, ppix = get_peaks(coarse_comp, specname=specname)

            mat_ppix = ppix.reshape((ppix.size, 1))

            if last_obs is not None and specname in last_obs.colnames:
                best = {}
                best['a'],best['b'],best['c'] = last_obs[specname][0:3]
            else:
                best = last_good.copy()
            best['euc'], best['nlines'] = 1e8, 1
            best['clines'],best['pixels'] = [],[]
            euc_tolerance = euc_tolerance_perpix * (len(ppix) - n_baddist_skip)

            nochange = 0
            for iter in np.arange(n_resolution_iters):
                frac = frac_reduction * frac
                aas = make_range(best['a'], abound_hw * frac, step_fraction=stepfrac)
                bs = make_range(best['b'], bbound_hw * frac, step_fraction=stepfrac)
                cs = make_range(best['c'], cbound_hw * frac, step_fraction=stepfrac)

                old_bests = np.array([best['a'], best['b'], best['c']], copy=True)
                for c in cs:
                    cterm = (c * mat_ppix * mat_ppix)
                    for b in bs:
                        bcterm = (b * mat_ppix) + cterm
                        for a in aas:
                            guess = a + bcterm
                            dists_locs = np.argmin(np.abs(mat_wm - guess), axis=1)
                            dists = np.abs(wm[dists_locs] - guess.flatten())
                            if n_baddist_skip > 0:
                                sorted_inds = np.argsort(dists)[:-n_baddist_skip]  # [:int(0.5*len(ppix))]
                                cdists = dists[sorted_inds]
                            else:
                                cdists = dists
                            euc_dist = np.sqrt(np.dot(cdists, cdists))

                            if euc_dist < best['euc']:
                                best['euc'] = euc_dist
                                best['nlines'] = len(cdists)
                                if n_baddist_skip > 0:
                                    best['clines'] = wm[dists_locs][sorted_inds]
                                    best['pixels'] = ppix[sorted_inds]
                                else:
                                    best['clines'] = wm[dists_locs]
                                    best['pixels'] = ppix
                                best['a'] = a
                                best['b'] = b
                                best['c'] = c
                if np.all(np.abs(old_bests - np.array([best['a'], best['b'], best['c']])) / old_bests < 0.00001):
                    # print(specname,(old_bests - np.array([best['a'], best['b'], best['c']])) / old_bests )
                    nochange += 1
                else:
                    nochange = 0
                if best['euc'] < euc_tolerance or nochange > 5:
                    break
            calib_coefs[specname] = best.copy()
            print(specname, '---->\tmin_dist={:0.2f}, with nlines={:02d} ({:0.4f} dist/line)\n\t\t\ta=\t{:0.2f}\t\tb=\t{:0.04f}\t\tc=\t{:0.04e}\n\t\t\tMet Threshold?: {}'.format(best['euc'],best['nlines'],best['euc']/best['nlines'],best['a'],best['b'],best['c'],best['euc'] < euc_tolerance))
            if best['euc'] < 2 * euc_tolerance:
                last_good = best.copy()
            if best['euc'] > 4 * euc_tolerance:
                if best['nlines'] > 3:  # 3 coefs, reducing nlines by 1 each iteration
                    if len(bad_specs)+1 > 20:
                        current_loop_loc = np.where(specs_to_run==specname)[0][0]
                        bad_specs.extend(specs_to_run[current_loop_loc:].tolist())
                        specs_to_run = np.array(bad_specs, copy=True)
                        print("\nMore than 20 bad specs found in this iteration. Possibly too many lines used. Rerunning with fewer.")
                        print("\n\n\tRerunning: {}".format(specs_to_run))
                        break
                    else:
                        bad_specs.append(specname)
                else:
                    print(
                        "{} only has 3 lines remaining. A better fit cannot be made. Not continuing with that fiber".format(
                            specname))
                # plt.figure()
                # plt.title(specname)
                # plt.plot(get_waves_fromdict(pix, best), specflux, label='r816')
                # plt.plot(get_waves_fromdict(ppix, best), ph, 'r*')
                # best_guess = get_waves_fromdict(ppix, best)
                # matched_wm_inds = np.argmin(np.abs(mat_wm - guess), axis=1)
                # calibflux = fm[matched_wm_inds]
                # spec_adjusted_calibflux = specflux.max() * calibflux / calibflux.max()
                # plt.plot(wm[matched_wm_inds], spec_adjusted_calibflux, 'b^')
                # plt.show()
            del best
            ## Don't divide by frac, meaning all iterations after the first
            ## will start with smaller step sizes by that factor
            frac = 1.

        specs_to_run = np.array(bad_specs, copy=True)
        if len(bad_specs) == 0:
            break
        else:
            print("\n\n\tRerunning: {}".format(specs_to_run))

    return calib_coefs


def poly5(x,a,b,c,d,e,f):
    return a+ b*x + c*x*x + d*np.power(x,3) + \
           e * np.power(x, 4) + f*np.power(x,5)
def poly2(x,a,b,c):
    return a+ b*x + c*x*x

def make_range(mean_val,bound_hw,step_fraction= 0.1):
    lower_bound = mean_val-bound_hw
    upper_bound = mean_val+bound_hw
    step_size = step_fraction*bound_hw
    return np.arange(lower_bound,upper_bound+step_size,step_size)

def get_waves_fromdict(pixels,diction):
    a,b,c = diction['a'],diction['b'],diction['c']
    return a+b*pixels+c*pixels*pixels


def get_wm_fm(complinelistdict,cut= 10000.):
    wm, fm = [], []

    # plt.figure()
    for element, (xe0, xe1) in complinelistdict.items():
        xe1f = xe1.copy()
        xe1f[xe1f < 1.] = 1.
        xe0c = xe0[xe1f > cut].copy()
        xe1c = xe1f[xe1f > cut].copy()
        wm.extend(xe0c.tolist())
        fm.extend(xe1c.tolist())
    wm,fm = np.array(wm),np.array(fm)
    # fm = fm[((wm>5200)&(wm<7100))]
    # wm = wm[((wm>5200)&(wm<7100))]
    sortd = np.argsort(wm)
    return wm[sortd],fm[sortd]

def get_peaks(coarse_comp,specname='r816'):
    min_height = 200.
    spec = coarse_comp[specname]
    pix = np.arange(len(spec)).astype(float)
    nppix = 0.
    itter = 1
    maxspec = np.max(spec)
    while nppix < 22:
        ppix, peaks = find_peaks(spec, height=maxspec / (2+itter), prominence=maxspec / (2+itter),wlen=100)
        nppix = len(ppix)
        itter += 1
    ph = np.array(peaks['peak_heights'])
    proms = np.array(peaks['prominences'])
    ppix = np.array(ppix)
    return spec,pix,ph[proms>min_height],ppix[proms>min_height]


def coarse_calib_configure_tables(dict_of_dicts):
    maxlines = np.max([val['nlines'] for val in dict_of_dicts.values()])
    coeftab,metrictab,linestab,pixtab = Table(),Table(),Table(),Table()

    for fib,fibdict in dict_of_dicts.items():
        coefcolvals = np.array([fibdict['a'], fibdict['b'], fibdict['c'], 0., 0., 0.])
        coefcol = Table.Column(name=fib,data=coefcolvals)
        coeftab.add_column(coefcol)

        metcol = Table.Column(name=fib,data=np.array([fibdict['euc']]))
        metrictab.add_column(metcol)

        lines = np.append(fibdict['clines'],np.zeros(maxlines-fibdict['nlines']))
        pixels = np.append(fibdict['pixels'],np.zeros(maxlines-fibdict['nlines']))

        linecol = Table.Column(name=fib,data=lines)
        linestab.add_column(linecol)

        pixelcol = Table.Column(name=fib,data=pixels)
        pixtab.add_column(pixelcol)

    out = {'coefs': coeftab,
           'metric': metrictab,
           'clines': linestab,
           'pixels': pixtab  }
    return out


def compare_outputs(raw_data,table1,table2,save_template='{fiber}.png',save_plots=True,show_plots=False):
    def waves(pixels, a, b, c,d,e,f):
        return a + (b * pixels) + (c * pixels * pixels)+\
               (d * np.power(pixels,3)) + (e * np.power(pixels,4))+ \
               (f * np.power(pixels, 5))
    fib1s = set(table1.colnames)
    fib2s = set(table2.colnames)
    matches = fib1s.intersection(fib2s)

    for match in matches:
        pixels = np.arange(len(raw_data[match])).astype(np.float64)
        a1,b1,c1,d1,e1,f1 = table1[match]
        a2, b2, c2, d2, e2, f2 = table2[match]
        waves1 = waves(pixels, a1,b1,c1,d1,e1,f1)
        waves2 = waves(pixels, a2, b2, c2, d2, e2, f2)
        dwaves = waves1-waves2
        print("\n"+match)
        print("--> Max deviation: {}  mean: {}  median: {}".format(dwaves[np.argmax(np.abs(dwaves))], np.mean(np.abs(dwaves)), np.median(np.abs(dwaves))))
        plt.figure()
        plt.plot(pixels, dwaves, 'r-')
        plt.title("{}  max dev={}".format(match,np.max(np.abs(dwaves))))
        plt.ylabel("Fit 1 - Fit 2 [Angstrom]")
        plt.xlabel("Pixel")
        if save_plots:
            plt.savefig(save_template.format(fiber=match))
        if show_plots:
            plt.show()
        plt.close()
    return matches

def automated_calib_wrapper_script(input_dict):
    return run_automated_calibration(**input_dict)


def aperature_number_pixoffset(fibnum,camera='r'):
    if type(fibnum) is str:
        strpd_fibnum = fibnum.strip('rb')
        if strpd_fibnum.isnumeric():
            tet = np.int8(strpd_fibnum[0]) - 1.
            fib = np.int8(strpd_fibnum[1:]) - 1.
        else:
            return 0.
    elif np.isscalar(fibnum):
        tet = fibnum // 16
        fib = fibnum % 16

    if camera.lower() != 'r':
        orientation = 1.
    else:
        orientation = -1.
    c1, c2, c3, c4, c5 = 1.023, 54.058, -6.962, 1.985, -0.5560
    outval_mag = (c1) + (c2 * tet) + (c3 * tet * tet) + (c4 * fib) + (c5 * tet * fib)
    return orientation * outval_mag

def aperature_pixoffset_between2(fibnum,camera='r'):
    if type(fibnum) is str:
        strpd_fibnum = fibnum.strip('rb')
        if strpd_fibnum.isnumeric():
            tet = np.int8(strpd_fibnum[0]) - 1.
            fib = np.int8(strpd_fibnum[1:]) - 1.
        else:
            return 0.
        fibnum = 16*tet+fib
    elif not np.isscalar(fibnum):
        return 0.

    if fibnum == 0:
        return 0.
    else:
        return aperature_number_pixoffset(fibnum,camera)-aperature_number_pixoffset(fibnum-1,camera)


def top_peak_wavelengths(pixels,spectra,coefs):
    top_pixels = top_peak_pixels(pixels, spectra)
    max_flux_wavelengths = pix_to_wave(top_pixels,coefs)
    return max_flux_wavelengths


def top_peak_pixels(pixels,spectra):
    max_locs = argrelmax(spectra)[0]
    max_vals = spectra[max_locs]
    sorted_max_val_inds = np.argsort(max_vals).astype(int)
    top_max_val_inds = sorted_max_val_inds[-2:]
    top_max_locs = max_locs[top_max_val_inds]
    max_flux_pixels = pixels[top_max_locs]
    return np.sort(max_flux_pixels)


def get_highestflux_waves(complinelistdict):
    fms, wms = [], []
    for (cwm, cfm) in complinelistdict.values():
        fms.extend(cfm)
        wms.extend(cwm)
    fms, wms = np.asarray(fms), np.asarray(wms)
    flux_sorter = np.argsort(fms)
    top_inds = flux_sorter[int(0.75 * len(fms)):]
    fsorted_top_flux = fms[top_inds]
    fsorted_top_wave = wms[top_inds]
    wave_sorter = np.argsort(fsorted_top_wave)

    wsorted_top_flux = fsorted_top_flux[wave_sorter]
    wsorted_top_wave = fsorted_top_wave[wave_sorter]

    return wsorted_top_wave,wsorted_top_flux


def update_default_dict(default_dict,fiber_identifier,default_vals, history_vals, \
                        pixels, comp_spec,matched_peak_waves,\
                        do_history=False,first_iteration=True):
    ## Change offset of the basic default
    if default_vals is not None:
        adef,bdef,cdef,ddef,edef,fdef = default_vals[fiber_identifier]
        default_dict['default'] = (adef,bdef,cdef)
    else:
        adef, bdef, cdef, ddef, edef, fdef = 4523.4,1.0007,-1.6e-6,0.,0.,0.
    ## Update historical default
    if do_history:
        if fiber_identifier in history_vals.colnames:
            default_dict['from history'] = history_vals[fiber_identifier]
        else:
            default_dict['from history'] = default_dict['default']

    ## Guess next position from the previous one and predictive offset function
    apred, bpred, cpred = default_dict['predicted from prev spec']
    #expected_difference = aperature_pixoffset_between2(fiber_identifier)
    #default_dict['predicted from prev spec'] = (apred+expected_difference, bpred, cpred)

    ## Use largest peaks to guess the constant and linear terms
    if not first_iteration:
        top_pixel_peaks = top_peak_pixels(pixels, comp_spec)
        ## Fit to line but including the predicted quadratic term
        dpix = top_pixel_peaks[1] - top_pixel_peaks[0]
        dlam = matched_peak_waves[1] - matched_peak_waves[0]
        mean_pix = np.mean(top_pixel_peaks)
        bcor = (dlam / dpix) - 2 * cpred * mean_pix
        mean_wave = np.mean(matched_peak_waves)
        acor = mean_wave - (bcor * mean_pix) - (cpred * mean_pix * mean_pix)
        prev_acor, prev_bcor, prev_ccor = default_dict['cross correlation']
        if np.abs(prev_acor - acor) < 50 and np.abs(prev_bcor - bcor) < 0.2:
            default_dict['cross correlation'] = (acor, bcor, cpred)
            default_dict['predicted from prev spec'] = (apred + (acor-prev_acor), bpred, cpred)
        else:
            default_dict['cross correlation'] = (adef, bpred, cpred)
            default_dict['predicted from prev spec'] = (adef, bpred, cpred)
    return default_dict


def interactive_plot(pixels,spectra,linelistdict,gal_identifier,\
                     default_dict, steps, default_key):
    fig = plt.figure()
    axsrc = plt.subplot2grid((1, 3), (0, 0), colspan=2)
    axzoom = plt.subplot2grid((1, 3), (0, 2))

    bigwin_cal_lines, zoom_cal_lines = [], []
    colors = ['r', 'g', 'k', 'orange', 'm','y','m'][:len(linelistdict)]
    visibility = [True]*len(linelistdict)
    min_wavelengths,max_wavelengths = [],[]
    min_flux,max_flux = np.min(spectra),np.max(spectra)
    labels = []
    ## unpack lines
    for (linelabel,(wave,flux)),color in zip(linelistdict.items(),colors):
        ## signal
        #sl, = axsrc.plot(wave, flux, lw=2, color=color, label=linelabel)
        linewidths = 4*(flux/np.max(flux))
        linewidths = 3*(np.log(linewidths-np.min(linewidths)+1.))
        sl = axsrc.vlines(wave, ymin=min_flux,ymax=max_flux,lw=linewidths, color=color, label=linelabel)
        bigwin_cal_lines.append(sl)
        ## zoomed
        #zl, = axzoom.plot(wave, flux, lw=2, color=color, label=linelabel)
        zl = axzoom.vlines(wave,  ymin=min_flux,ymax=max_flux,lw=linewidths, color=color, label=linelabel)
        zoom_cal_lines.append(zl)
        min_wavelengths.append(wave[0])
        max_wavelengths.append(wave[-1])
        labels.append(linelabel)

    # from utility_funcs import VertSlider
    max_wavelength = max(max_wavelengths)+400
    min_wavelength = min(min_wavelengths)-400
    wavelength_halfwidth = max_wavelength//50

    spectra_is_good = True
    axcolor = 'lightgoldenrodyellow'

    coefs = {}
    coefs['a'], coefs['b'], coefs['c'] = default_dict[default_key]
    #coefs['d'], coefs['e'], coefs['f'] = 0, 0, 0

    waves = pix_to_wave(pixels, coefs)
    tog1 = spectra
    tog2 = medfilt(tog1, 3)

    smooth_noise_dict = {'Original': tog1, 'Smooth': tog2}

    meanwave = (max_wavelength   + min_wavelength) // 2

    # fig, (axsrc, axzoom) = plt.subplots(nrows=1,ncols=2)


    ## Signal
    stogl, = axsrc.plot(waves, tog1, lw=2, color='b', label='Original')
    ztogl, = axzoom.plot(waves, tog1, lw=2, color='b', label='Original')


    ## Button and slider funcs
    def showunshow_lines(label):
        index = labels.index(label)
        bigwin_cal_lines[index].set_visible(not bigwin_cal_lines[index].get_visible())
        zoom_cal_lines[index].set_visible(not zoom_cal_lines[index].get_visible())
        plt.draw()


    def zoom_adjust(event):
        if event.button != 1:
            return
        elif (event.inaxes == axsrc or event.inaxes == axzoom):
            x, y = event.xdata, event.ydata
            axzoom.set_xlim(x - wavelength_halfwidth, x + wavelength_halfwidth)
            fig.canvas.draw()


    def smooth_noise_flip(label):
        ydata = smooth_noise_dict[label]
        stogl.set_ydata(ydata)
        ztogl.set_ydata(ydata)
        plt.draw()


    def change_default_sliderset(label):
        def_off, def_off_fine, def_lin, def_lin_fine, def_quad_fine = split_params(*(default_dict[label]), \
                                                                                   *steps)

        off_slide.valinit, lin_slide.valinit = def_off, def_lin
        off_slide.vline.set_xdata([def_off] * len(off_slide.vline.get_xdata()))
        lin_slide.vline.set_xdata([def_lin] * len(lin_slide.vline.get_xdata()))

        off_slide_fine.valinit, lin_slide_fine.valinit = def_off_fine, def_lin_fine
        off_slide_fine.vline.set_xdata([def_off_fine] * len(off_slide.vline.get_xdata()))
        lin_slide_fine.vline.set_xdata([def_lin_fine] * len(lin_slide.vline.get_xdata()))

        quad_slide_fine.valinit = def_quad_fine
        quad_slide_fine.vline.set_xdata([def_quad_fine]*len(lin_slide.vline.get_xdata()))

        plt.draw()


    def slider_spec_update(val):
        coefs['a'] = off_slide.val + off_slide_fine.val
        coefs['b'] = lin_slide.val + lin_slide_fine.val
        coefs['c'] = quad_slide_fine.val  # quad_slide.val+quad_slide_fine.val
        waves = pix_to_wave(pixels, coefs)
        stogl.set_xdata(waves)
        ztogl.set_xdata(waves)
        fig.canvas.draw_idle()


    def reset_sliders(event):
        lin_slide.reset()
        off_slide.reset()
        # quad_slide.reset()
        lin_slide_fine.reset()
        off_slide_fine.reset()
        quad_slide_fine.reset()
        slider_spec_update(None)


    def flag_spec(event):
        print("\n\tSpec flagged as bad\n")
        spectra_is_good = False
        plt.close()


    def save_and_close(event):
        print("\n\tSpec closed with the following params: a={} b={} c={}\n".format(coefs['a'], coefs['b'], \
                                                                             coefs['c']))
        plt.close()


    def print_to_screen(event):
        print("\ta={} b={} c={}".format(coefs['a'], coefs['b'], coefs['c']))


    ## Make checkbuttons with all plotted lines with correct visibility
    ## [x,y,width,height]
    plot_ystart = 0.36

    slider_ystart = 0.03

    slider_xstart = 0.04
    boxes_xstart_row1 = 0.7
    boxes_xstart_row2 = 0.8
    boxes_xstart_row3 = 0.92

    box_width = 0.08
    box_width2 = 0.14
    slider_width = 0.62

    height2 = 0.1
    height3 = 0.15
    height_slider = 0.03
    height_button = 0.04

    ## Move subplot over to make room for checkboxes
    plt.subplots_adjust(left=slider_xstart, right=1 - slider_xstart, \
                        bottom=plot_ystart, top=1 - slider_xstart)

    ## Change the name and limits of each axis
    axsrc.set(xlim=(min_wavelength, max_wavelength), ylim=(min_flux,max_flux), autoscale_on=False,
              title='Click to zoom')
    axzoom.set(xlim=(meanwave - wavelength_halfwidth, meanwave + wavelength_halfwidth), \
               ylim=(min_flux,max_flux), autoscale_on=False, title='Zoom window')

    ## Setup button locations
    # slider1_rax = plt.axes([slider_xstart, slider_ystart+10*height_slider, slider_width, height_slider], facecolor=axcolor)
    slider2_rax = plt.axes([slider_xstart, slider_ystart + 8 * height_slider, slider_width, height_slider],
                           facecolor=axcolor)
    slider3_rax = plt.axes([slider_xstart, slider_ystart + 6 * height_slider, slider_width, height_slider],
                           facecolor=axcolor)
    slider4_rax = plt.axes([slider_xstart, slider_ystart + 4 * height_slider, slider_width, height_slider],
                           facecolor=axcolor)
    slider5_rax = plt.axes([slider_xstart, slider_ystart + 2 * height_slider, slider_width, height_slider],
                           facecolor=axcolor)
    slider6_rax = plt.axes([slider_xstart, slider_ystart, slider_width, height_slider], facecolor=axcolor)

    linelists_rax = plt.axes([boxes_xstart_row1, slider_ystart, box_width, height3], facecolor=axcolor)

    spec_radio_rax = plt.axes([boxes_xstart_row1, 0.20, box_width, height2], facecolor=axcolor)
    def_radio_rax = plt.axes([boxes_xstart_row2, 0.15, box_width2, height3], facecolor=axcolor)

    close_rax = plt.axes([boxes_xstart_row3, slider_ystart, 0.05, height_button])
    reset_rax = plt.axes([boxes_xstart_row3, slider_ystart + 0.06, 0.05, height_button])
    flag_rax = plt.axes([boxes_xstart_row2, slider_ystart, 0.1, height_button])
    print_rax = plt.axes([boxes_xstart_row2, slider_ystart + 0.06, 0.1, height_button])

    ## Checkboxes
    linelist_check = CheckButtons(linelists_rax, labels, visibility)
    ## Radio boxes
    spec_radio = RadioButtons(spec_radio_rax, ['Original', 'Smooth'],active=[0])
    def_keys = list(default_dict.keys())
    def_key_index = def_keys.index(default_key)
    def_radio = RadioButtons(def_radio_rax, def_keys,active=def_key_index)
    ## Sliders

    def_off,def_off_fine, def_lin, def_lin_fine, def_quad_fine = split_params(*(default_dict[default_key]),\
                                                                              *steps)
    off_slide = Slider(slider2_rax, 'offset', -2000., 10000.0, valinit=def_off, valstep=steps[0])
    lin_slide = Slider(slider3_rax, 'stretch', 0.4,2.5, valinit=def_lin, valstep=steps[1])
    # quad_slide = Slider(slider3_rax, 'quad', -10.0, 10.0, valinit=default_dict[default_key][2], valstep=steps[2])
    off_slide_fine = Slider(slider4_rax, 'fine offset', -200, 200, valinit=def_off_fine, valstep=steps[0] / 100)
    lin_slide_fine = Slider(slider5_rax, 'fine stretch', -0.05,0.05, valinit=def_lin_fine, valstep=steps[1] / 100, \
                        valfmt='%1.4f')
    quad_slide_fine = Slider(slider6_rax, 'fine quad',-4e-5,4e-5, valinit=def_quad_fine, valstep=steps[2] / 100,\
                        valfmt='%1.6f')

    ## Buttons
    reset_button = Button(reset_rax, 'Reset', color=axcolor, hovercolor='0.975')
    flag_button = Button(flag_rax, 'Flag as Bad', color=axcolor, hovercolor='0.975')
    close_button = Button(close_rax, 'Close', color=axcolor, hovercolor='0.975')
    print_button = Button(print_rax, 'Print to Terminal', color=axcolor, hovercolor='0.975')

    ## Run the interactive buttons
    fig.canvas.mpl_connect('button_press_event', zoom_adjust)
    linelist_check.on_clicked(showunshow_lines)
    spec_radio.on_clicked(smooth_noise_flip)
    def_radio.on_clicked(change_default_sliderset)

    lin_slide.on_changed(slider_spec_update)
    off_slide.on_changed(slider_spec_update)
    # quad_slide.on_changed(slider_spec_update)
    lin_slide_fine.on_changed(slider_spec_update)
    off_slide_fine.on_changed(slider_spec_update)
    quad_slide_fine.on_changed(slider_spec_update)

    reset_button.on_clicked(reset_sliders)
    flag_button.on_clicked(flag_spec)
    close_button.on_clicked(save_and_close)
    print_button.on_clicked(print_to_screen)
    multi = MultiCursor(fig.canvas, (axsrc, axzoom), color='r', lw=1)

    ## plot the final canvas in a pop-up window
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()

    return spectra_is_good, coefs



def pix_to_wave(xs, coefs):
    return coefs['a'] + coefs['b'] * xs + coefs['c'] * np.power(xs, 2) #+ \
           #coefs['d'] * np.power(xs, 3) + coefs['e'] * np.power(xs, 4) + \
           #coefs['f'] * np.power(xs, 5)

def pix_to_wave_fifthorder(xs, coefs):
    return coefs[0] + coefs[1] * xs + coefs[2] * np.power(xs, 2) + \
           coefs[3] * np.power(xs, 3) + coefs[4] * np.power(xs, 4) + \
           coefs[5] * np.power(xs, 5)


def iterate_fib(fib,cam):
    tetn = int(fib[1])
    fibn = int(fib[2:])
    if tetn == 8 and fibn >= 8:
        fibn -= 1
    elif tetn == 4 and fibn >= 8:
        fibn -= 1
    else:
        fibn += 1
        if fibn > 16:
            tetn += 1
            fibn = 1
    outfib = '{}{}{:02d}'.format(cam, tetn, fibn)
    return outfib


def ensure_match(fib, allfibs, subset, cam):
    print(fib)
    outfib = fib
    if outfib not in allfibs:
        outfib = iterate_fib(outfib,cam)
        outfib = ensure_match(outfib, allfibs, subset, cam)
    if outfib in subset:
        outfib = iterate_fib(outfib,cam)
        outfib = ensure_match(outfib, allfibs, subset, cam)
    return outfib

def find_devs(table1,table2):
    xs = np.arange(2000).astype(np.float64)
    overlaps = list(set(list(table1.colnames)).intersection(set(list(table2.colnames))))
    devs = []
    for fib in overlaps:
        coef_dev = np.asarray(table1[fib])-np.asarray(table2[fib])
        full_devs = np.polyval(coef_dev[::-1],xs)
        dev = np.std(full_devs)
        devs.append(dev)
    return np.mean(devs)

def split_params(off,lin,quad,offstep,linstep,quadstep):
    def_quad_fine = quad
    off_fine = off % offstep
    if off_fine < (offstep/1000) or off_fine > (offstep*(999/1000.)):
        def_off = off
        def_off_fine = 0.
    else:
        def_off = off - off_fine
        def_off_fine = off_fine
    lin_fine = lin % linstep
    if lin_fine < (linstep/1000) or lin_fine > (linstep*(999/1000.)):
        def_lin = lin
        def_lin_fine = 0.
    else:
        def_lin = lin - lin_fine
        def_lin_fine = lin_fine
    return def_off, def_off_fine, def_lin, def_lin_fine, def_quad_fine

def air_to_vacuum(airwl, nouvconv=True):
    """
    Returns vacuum wavelength of the provided air wavelength array or scalar.
    Good to ~ .0005 angstroms.

    If nouvconv is True, does nothing for air wavelength < 2000 angstroms.

    Input must be in angstroms.

    Adapted from idlutils airtovac.pro, based on the IAU standard
    for conversion in Morton (1991 Ap.J. Suppl. 77, 119)
    """
    airwl = np.array(airwl, copy=False, dtype=float, ndmin=1)
    isscal = airwl.shape == tuple()
    if isscal:
        airwl = airwl.ravel()

    # wavenumber squared
    sig2 = (1e4 / airwl) ** 2

    convfact = 1. + 6.4328e-5 + 2.94981e-2 / (146. - sig2) + 2.5540e-4 / (41. - sig2)
    newwl = airwl.copy()
    if nouvconv:
        convmask = newwl >= 2000
        newwl[convmask] *= convfact[convmask]
    else:
        newwl[:] *= convfact
    return newwl[0] if isscal else newwl

if __name__ == '__main__':
    #cal_lamp = ['Hg', 'He','Ar', 'Ne', 'Xe']  ## crc
    #from calibrations import load_calibration_lines_crc_dict as load_calibration
    #cal_lamp = ['Hg', 'Ar', 'Ne','Xe'] ## nist
    #from calibrations import load_calibration_lines_nist_dict as load_calibration
    cal_lamp = ['HgAr', 'NeAr', 'Ar', 'Xe'] ## salt
    from calibrations import load_calibration_lines_salt_dict as load_calibration


    linelistdict, cal_states = load_calibration(cal_lamp,wavemincut=4500,wavemaxcut=6600)
    default = (4521,1.,0.)
    default_dict = {
        'default': default,
        'predicted from prev spec': (4521, 1., 0.),
        'predicted from history': (5625, 1., 0.),
        'from history': (5625, 1., 0.)
    }

    steps = (1, 0.01, 0.00001)
    default_key = 'predicted from prev spec'

    apcut_data = None
    import pickle as pkl
    with open('pkldump.pkl', 'rb') as pdump:
        apcut_data = pkl.load(pdump)

    specs = apcut_data['comp']['r'][628]
    fiber_identifier = 'r101'
    spectra = specs[fiber_identifier]
    interactive_plot(pixels=np.arange(len(spectra)), spectra=spectra,\
                     linelistdict=linelistdict, gal_identifier=fiber_identifier, \
                     default_dict=default_dict, steps=steps, default_key=default_key)

    cal_lamp = ['ThAr']  ## salt
    spectra = apcut_data['thar']['r'][627][fiber_identifier]
    linelistdict, cal_states = load_calibration(cal_lamp, \
                                                wavemincut=4500, \
                                                wavemaxcut=6600)
    interactive_plot(pixels=np.arange(len(spectra)), spectra=spectra,\
                     linelistdict=linelistdict, gal_identifier=fiber_identifier, \
                     default_dict=default_dict, steps=steps, default_key=default_key)


