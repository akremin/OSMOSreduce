
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
import datetime

import numpy as np
from scipy.signal import argrelmax
from collections import OrderedDict
deltat = np.datetime64('now' ,'m').astype(int) -np.datetime64('2018-06-01T00:00' ,'m').astype(int)
print(deltat)
import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table, hstack

from scipy.signal import medfilt
from scipy.signal import find_peaks





def coarse_calib_configure_tables(dict_of_dicts):
    maxlines = np.max([len(val['clines']) for val in dict_of_dicts.values()])
    coeftab,metrictab,linestab,pixtab = Table(),Table(),Table(),Table()

    for fib,fibdict in dict_of_dicts.items():
        coefcolvals = np.array(fibdict['coefs'])
        coefcol = Table.Column(name=fib,data=coefcolvals)
        coeftab.add_column(coefcol)

        metcol = Table.Column(name=fib,data=np.array([fibdict['metric']]))
        metrictab.add_column(metcol)

        lines = np.append(fibdict['clines'],np.zeros(maxlines-len(fibdict['clines'])))
        pixels = np.append(fibdict['pixels'],np.zeros(maxlines-len(fibdict['clines'])))

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
    fib1s = set(table1.colnames)
    fib2s = set(table2.colnames)
    matches = fib1s.intersection(fib2s)

    for match in matches:
        pixels = np.arange(len(raw_data[match])).astype(np.float64)
        waves1 = pix_to_wave(pixels, table1[match])
        waves2 = pix_to_wave(pixels, table2[match])
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




def pix_to_wave(xs, coefs):
    if type(coefs) is dict:
        if 'd' in coefs.keys():
            coefs = np.array([coefs['a'], coefs['b'], coefs['c'], coefs['d'], coefs['e'], coefs['f']])
        else:
            coefs = np.array([coefs['a'], coefs['b'], coefs['c']])
    if coefs[0] > 10.:
        return np.polyval(coefs[::-1], xs)
    else:
        return np.polyval(coefs, xs)

def pix_to_wave_explicit_coefs5(xs, a,b,c,d,e,f):
    return np.polyval([f,e,d,c,b,a], xs)

def pix_to_wave_explicit_coefs2(xs, a,b,c):
    return np.polyval([c, b, a], xs)

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
    if type(table1) is dict:
        table1 = Table(table1)
    if type(table2) is dict:
        table2 = Table(table2)
    overlaps = list(set(list(table1.colnames)).intersection(set(list(table2.colnames))))
    devs = []
    for fib in overlaps:
        coef_dev = np.asarray(table1[fib])-np.asarray(table2[fib])
        full_devs = pix_to_wave(xs,coef_dev)
        dev = np.std(full_devs)
        devs.append(dev)
    return np.mean(devs)


def get_meantime_and_date(header):
    jan12015 = datetime.datetime.timestamp(datetime.datetime(year=2015, month=1, day=1))
    fmt = '%Y-%m-%d %H:%M:%S'
    date = header['UT-DATE']
    stime,etime = header['UT-TIME'], header['UT-END']
    dt1 = datetime.datetime.timestamp(datetime.datetime.strptime(date + ' ' + stime,fmt))
    dt2 = datetime.datetime.timestamp(datetime.datetime.strptime(date+' '+etime,fmt))
    if int(stime[:2]) > int(etime[:2]):
        dt2 += (24*60*60.)
    mean_time = (dt1 + dt2) / 2.
    meantime_dt = datetime.datetime.fromtimestamp(mean_time)
    night = header['NIGHT']
    return (mean_time-jan12015), meantime_dt, night


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

def vacuum_to_air(vac_wave,resolution=1e-5):
    if np.isscalar(vac_wave):
        test_airs = np.arange(vac_wave-10.,vac_wave+10,resolution).astype(np.float64)
        test_vacs = air_to_vacuum(test_airs)
        index = np.argmin(np.abs(test_vacs-vac_wave))
        return np.round(test_airs[index],decimals=int(-1*np.log10(resolution)))
    else:
        outwaves = []
        for wave in vac_wave:
            outwaves.append(vacuum_to_air(wave, resolution))
        return np.array(outwaves)

def get_psf(current_flux, step=1., prom_quantile = 0.68, npeaks=6):
    for ii in np.arange(1,11, 1):
        peaks, props = find_peaks(current_flux, height=(np.mean(current_flux)/float(ii), 1e9), width=(2.355 / step, 6 * 2.355 / step))
        if len(peaks) < 6:
            continue
        prom_thresh = np.quantile(props['prominences'], prom_quantile)
        if np.sum(props['prominences'] > prom_thresh) < npeaks:
            continue
        sigma = step * np.mean(np.sort(props['widths'][props['prominences'] > prom_thresh])[:npeaks]) / 2.355
        break
    return sigma


def create_simple_line_spectra(elements, linelistdict, wave_low, wave_high, clab_step=0.01,\
                               return_individual=False,atm_weights={'Ne':1.,'Hg':0.2,'Ar':0.8}):
    if type(elements) is str:
        elements = [elements]
    def generate_line_spectra(wl, fl, wave_low, wave_high, calib_step):
        if type(fl) is Table.MaskedColumn:
            fl = fl.filled().data
        fl[np.isnan(fl)] = 0.
        fl[fl < 0.] = 0.
        cut = ((wl > wave_low) & (wl < wave_high))
        cut_wl, cut_fl = wl[cut], fl[cut]
        cut_waves = np.arange(wave_low, wave_high, calib_step).astype(np.float64)
        cut_fluxes = np.zeros(len(cut_waves))
        cal_line_inds = np.round(cut_wl / calib_step, 0).astype(int) - np.round(wave_low / calib_step, 0).astype(int)
        cut_fluxes[cal_line_inds] = cut_fl

        return cut_waves, cut_fluxes

    cut_fluxes = {}
    cut_flux = np.zeros(int((wave_high - wave_low) / clab_step)).astype(np.float64)
    for el in elements:
        if el in linelistdict.keys():
            wl,fl = linelistdict[el]
            cut_waves, cut_fluxes[el] = generate_line_spectra(wl,fl,wave_low,wave_high,clab_step)
            cut_fluxes[el][cut_waves > 6700] = 0.16 * cut_fluxes[el][cut_waves > 6700]
            if el in atm_weights.keys():
                cut_fluxes[el] *= atm_weights[el]
            cut_flux += cut_fluxes[el]
    # cut_flux = cut_fluxes['Ne'] + cut_fluxes['Hg'] + cut_fluxes['Ar']
    # cut_flux[cut_waves<5500] = 0.4*cut_flux[cut_waves<5500]
    if return_individual:
        return cut_waves, cut_flux, cut_fluxes
    else:
        return cut_waves,cut_flux

def update_coeficients_deviations(fiber,coef_table,completed_coefs):
    navg = 2
    if type(completed_coefs) is dict:
        allkeys = np.array(list(completed_coefs.keys()))
    else:
        allkeys = np.array(completed_coefs.colnames)
    fibn = get_fiber_number(fibername=fiber)
    all_fibns = np.array(get_fiber_number(fibername=allkeys))
    if type(all_fibns) not in [np.array,np.ndarray]:
        all_fibns = np.array([all_fibns])
    sortd_locs = np.argsort(np.abs(fibn-all_fibns))
    if len(sortd_locs) == 1:
        keys = allkeys
        diffs2 = np.asarray([(all_fibns - fibn) * (all_fibns - fibn)])
    else:
        if len(sortd_locs)>navg:
            locs = sortd_locs[:navg]
        else:
            locs = sortd_locs
        keys = allkeys[locs]
        diffs2 = (all_fibns[locs] - fibn) * (all_fibns[locs] - fibn)

    total = np.sum(diffs2)
    invw2 = {}
    for key,diff in zip(keys,diffs2):
        invw2[key] = diff
    # coef_devs = np.asarray([np.power(np.asarray(completed_coefs[key]) - np.asarray(coef_table[key]),2)*invw2[key] for key in keys])
    # coef_dev_med = np.sqrt(np.sum(coef_devs,axis=0)/total)
    coef_devs = np.asarray([(np.asarray(completed_coefs[key]) - np.asarray(coef_table[key]))*invw2[key] for key in keys])
    coef_dev_med = np.sum(coef_devs,axis=0)/total
    return coef_dev_med


def get_fiber_number(fibername='r101', cam=None):
    if type(fibername) not in [list,np.array,np.ndarray,Table.Column]:
        fibername = [fibername]
    if cam is None:
        cam = fibername[0][0]
    if cam not in ['b','r']:
        return None
    if cam == 'b':
        fibern = [(16 * (9 - int(fibnm[1]))) + int(fibnm[2:]) for fibnm in fibername]
    else:
        fibern = [(16 * int(fibnm[1])) + int(fibnm[2:]) for fibnm in fibername]

    if len(fibern) == 1:
        return fibern[0]
    else:
        return np.array(fibern)






###################################################################
############       Start Deprecated Section         ###############
###################################################################

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


def run_interactive_slider_calibration(self,coarse_comp, complinelistdict, default_vals=None,history_vals=None,\
                               steps = None, default_key = None, trust_initial = False):

    init_default = (4523.4,1.0007,-1.6e-6)

    default_dict = {    'default': init_default,
                        'predicted from prev spec': init_default,
                        'cross correlation': init_default           }

    do_history = False
    if history_vals is not None:
        default_dict['from history'] = init_default
        do_history = True

    if steps is None:
        steps = (1, 0.01, 0.00001)

    if default_key is None:
        default_key = 'cross correlation'

    ## Find the highest flux wavelengths in the calibrations
    wsorted_top_wave, wsorted_top_flux = get_highestflux_waves(complinelistdict)
    ## Make sure the information is in astropy table format
    coarse_comp = Table(coarse_comp)
    ## Define loop params
    counter = 0
    first_iteration = True

    ## Initiate arrays/dicts for later appending inside loop (for keeping in scope)
    matched_peak_waves, matched_peak_flux = [], []
    matched_peak_index = []
    all_coefs = {}
    all_flags = {}

    ## Loop over fiber names (strings e.g. 'r101')
    for fiber_identifier in coarse_comp.colnames:
        counter += 1
        print(fiber_identifier)

        ## Get the spectra (column with fiber name as column name)
        comp_spec = np.asarray(coarse_comp[fiber_identifier])

        ## create pixel array for mapping to wavelength
        pixels = np.arange(len(comp_spec))

        ## Update the defaults using history or cross correlation if available,
        ## and also update with a fitted function for the offsets
        default_dict = update_default_dict(default_dict,fiber_identifier,default_vals, history_vals, \
                                           pixels, comp_spec,matched_peak_waves,\
                                           do_history,first_iteration)

        ## Do an interactive second order fit to the spectra
        if trust_initial and counter != 1:
            good_spec = True
            out_coef = {}
            out_coef['a'],out_coef['b'],out_coef['c'] = default_dict[default_key]
            print("\t\tYou trusted {} which gave: a={} b={} c={}".format(default_key,*default_dict[default_key]))
        else:
            good_spec,out_coef = interactive_plot(pixels=pixels, spectra=comp_spec,\
                             linelistdict=complinelistdict, gal_identifier=fiber_identifier,\
                             default_dict=default_dict,steps=steps,default_key=default_key)

        ## If it's the first iteration, use the results to compute the largest
        ## flux lines and their true wavelength values
        ## these are used in all future iterations of this loop in the cross cor
        if first_iteration and good_spec:
            top_peak_waves = top_peak_wavelengths(pixels, comp_spec, out_coef)

            for peak in top_peak_waves:
                index = np.argmin(np.abs(wsorted_top_wave-peak))
                matched_peak_waves.append(wsorted_top_wave[index])
                matched_peak_flux.append(wsorted_top_flux[index])
                matched_peak_index.append(index)

            matched_peak_waves = np.asarray(matched_peak_waves)
            matched_peak_flux = np.asarray(matched_peak_flux)
            matched_peak_index = np.asarray(matched_peak_index)
            print("Returned waves: {}\nMatched_waves:{}\n".format(top_peak_waves,matched_peak_waves))

        ## Save the flag
        all_flags[fiber_identifier] = good_spec

        ## Save the coefficients if it's good
        if good_spec:
            default_dict['predicted from prev spec'] = (out_coef['a'],out_coef['b'],out_coef['c'])
            all_coefs[fiber_identifier] = [out_coef['a'],out_coef['b'],out_coef['c'],0.,0.,0.]
            first_iteration = False
        else:
            all_coefs[fiber_identifier] = [0.,0.,0.,0.,0.,0.]

        if counter == 999:
            counter = 0
            with open('_temp_wavecalib.pkl','wb') as temp_pkl:
                pkl.dump([all_coefs,all_flags],temp_pkl)
            print("Saving an incremental backup to _temp_wavecalib.pkl")
            cont = str(input("\n\n\tDo you want to continue? (y or n)\t\t"))
            if cont.lower() == 'n':
                break

    return Table(all_coefs)


###################################################################
############       End Deprecated Section         #################
###################################################################









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

