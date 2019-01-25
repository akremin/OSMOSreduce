##  a zoom in window
##  mutlicursor
## And checkboxes
## and radio buttons
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import MultiCursor
from matplotlib.widgets import CheckButtons
from matplotlib.widgets import RadioButtons
from matplotlib.widgets import Slider, Button
from scipy.signal import medfilt

import numpy as np

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

