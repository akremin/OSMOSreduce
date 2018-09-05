


def gaussian_lines(line_x,line_a,xgrid,width=2.0):
    '''
    Creates ideal Xenon spectrum
    '''
    #print 'Creating ideal calibration spectrum'
    temp = np.zeros(xgrid.size)
    for i in range(line_a.size):
        gauss = line_a[i]*np.exp(-(xgrid-line_x[i])**2/(2*width**2))
        temp += gauss
    return temp

#def polyfour(x,a,b,c,d,e,f):
#    if np.isscalar(x):
#        return float(a + b*x + c*x**2.0 + d*x**3.0 + e*x**4.0 + f*x**5.0)
#    else:
#        x = np.asarray(x)
#        return np.array(a + b * x + c * x ** 2.0 + d * x ** 3.0 + e * x ** 4.0 + f * x ** 5.0).astype(float)


def quad_to_linear(c):
    def outfunc(xs,a,b):
        return a + b*xs + c*np.power(xs,2)
    return outfunc

#def linear(xs,a,b):
#    return a + b*xs

def wavecalibrate(px, fx, slit_x, stretch_est=0.0, shift_est=0.0, quad_est=0.0, cube_est=0.0, fourth_est=0.0,
                  fifth_est=0.0):
    # flip and normalize flux
    fx = fx - np.min(fx)
    fx = fx / signal.medfilt(fx, 201)

    # prep calibration lines into 1d spectra
    wm, fm = np.loadtxt('osmos_Xenon.dat', usecols=(0, 2), unpack=True)
    wm = air_to_vacuum(wm)
    xgrid = np.arange(0.0, 6800.0, 0.01)
    lines_gauss = gaussian_lines(wm, fm, xgrid)
    interp = interp1d(xgrid, lines_gauss, bounds_error=False, fill_value=0)
    # interp = UnivariateSpline(xgrid,lines_gauss)

    wave_est = fifth_est * (px - slit_x) ** 5 + fourth_est * (px - slit_x) ** 4 + cube_est * (
                px - slit_x) ** 3 + quad_est * (px - slit_x) ** 2 + (
                           px - slit_x) * stretch_est + shift_est  # don't subtract the slit pos because interactive plot doesn't (easier)
    wm_in = wm[np.where((wm < wave_est.max()) & (wm > wave_est.min()))]
    # wm_in = wm[np.where((wm<5000.0)&(wm>wave_est.min()))]
    px_max = np.zeros(wm_in.size)
    for i in range(wm_in.size):
        px_in = px[np.where((wave_est < wm_in[i] + 5.0) & (wave_est > wm_in[i] - 5))]
        px_max[i] = px_in[fx[np.where((wave_est < wm_in[i] + 5.0) & (wave_est > wm_in[i] - 5))].argmax()]

    params, pcov = curve_fit(polyfour, (px_max - slit_x), wm_in,
                             p0=[shift_est, stretch_est, quad_est, cube_est, fourth_est, fifth_est])
    # return (wave_new,fx,max_fourth,max_cube,max_quad,max_stretch,max_shift)

    return (
    params[0] + params[1] * (px - slit_x) + params[2] * (px - slit_x) ** 2 + params[3] * (px - slit_x) ** 3.0 + params[
        4] * (px - slit_x) ** 4.0 + params[5] * (px - slit_x) ** 5.0, fx, params[5], params[4], params[3], params[2],
    params[1], params[0])
    # return (param0+param1*(px-slit_x)+param2*(px-slit_x)**2+param3*(px-slit_x)**3.0+param4*(px-slit_x)**4.0+param5*(px-slit_x)**5.0,fx,params[4],params[3],params[2],params[1],params[0])

