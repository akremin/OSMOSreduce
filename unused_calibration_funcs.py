


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




def load_calibration_lines_dict(self,cal_lamp):
    from calibrations import air_to_vacuum
    linelistdict = {}
    print(('Using calibration lamps: ', cal_lamp))
    possibilities = ['Xe', 'Ar', 'HgNe', 'Hg', 'Ne', 'ThAr', 'Th']
    for lamp in possibilities:
        if lamp in cal_lamp:
            wm, fm = np.loadtxt('./lamp_linelists/osmos_{}.dat'.format(lamp), usecols=(0, 2), unpack=True)
            wm_vac = air_to_vacuum(wm)
            ## sort lines by wavelength
            sortd = np.argsort(wm_vac)
            srt_wm_vac, srt_fm = wm[sortd], fm[sortd]
            linelistdict[lamp] = (srt_wm_vac, srt_fm)

    cal_states = {
        'Xe': ('Xe' in cal_lamp), 'Ar': ('Ar' in cal_lamp), \
        'HgNe': ('HgNe' in cal_lamp), 'Ne': ('Ne' in cal_lamp)
    }

    return linelistdict, cal_states


def load_calibration_lines_NIST_dict(cal_lamp,wavemincut=4000,wavemaxcut=10000):
    from calibrations import air_to_vacuum
    lineuncertainty = 0.002
    linelistdict = {}
    print(('Using calibration lamps: ', cal_lamp))
    possibilities = ['Ar','He','Hg','Ne','ThAr','Th','Xe']
    for lamp in cal_lamp:
        if lamp in possibilities:
            print(lamp)
            if lamp is 'ThAr':
                tab = Table.read('./lamp_linelists/NIST/{}.txt'.format(lamp),
                                 format='ascii.csv', header_start=8, data_start=9)
                ## Quality Control
                tab = tab[tab['Obs_Unc'] <= lineuncertainty]
                ## Remove non-ThAr lines
                #tab = tab[tab['Name'] != 'NoID']
                tab = tab[tab['Name'] != 'Unkwn']
                ## Get wavelength and frequency information
                fm = tab.filled()['Rel_Intensity'].data
                wm_vac = tab.filled()['Obs_Wave'].data
            else:
                tab = Table.read('./lamp_linelists/NIST/{}.txt'.format(lamp),\
                                 format='ascii.csv', header_start=5, data_start=6)
                ## Quality Control the lines
                #tab = tab[tab['Flag'].mask]
                names = np.unique(tab['Name'])
                if 'I' in names[0]:
                    selection = names[0].split('I')[0]
                else:
                    selection = names[0].split('I')[0]
                name = selection+'I'
                tab = tab[tab['Name']==name]
                tab = tab[np.bitwise_not(tab['Rel_Intensity'].mask)]
                tab = tab[np.bitwise_not(tab['Ritz_Wave'].mask)]
                tab['Obs_Unc'].fill_value = 999.
                tab['Ritz_Unc'].fill_value = 999.
                tab['Obs-Ritz'].fill_value = 999.

                for col in ['Obs_Unc','Ritz_Unc','Obs-Ritz']:
                    tab = tab[(tab[col].filled().data.astype(float)<=lineuncertainty)]

                if np.all(tab['Calibd_Intensity'].mask):
                    fm = tab['Rel_Intensity'].filled().data
                else:
                    tab['Calib_Conf'].fill_value = 'E-'
                    tab['Calibd_Intensity'].fill_value = -999
                    calibd = tab[np.bitwise_not(tab['Calibd_Intensity'].mask)].filled()
                    calibs, rel = [],[]
                    for grade in ['A','AA','A+','A-','B+','BB','B']:
                        boolean = np.where(calibd['Calib_Conf']==grade)[0]
                        calibs.extend(calibd['Calibd_Intensity'][boolean].data.tolist())
                        rel.extend(calibd['Rel_Intensity'][boolean].data.tolist())
                    if len(calibs)>0:
                        ratios = np.array(calibs).astype(float)/np.array(rel).astype(float)
                        fm = tab.filled()['Rel_Intensity'].data.astype(float)*np.median(ratios)
                    else:
                        fm = tab['Rel_Intensity'].filled().data
                ## NIST values from 2000A to 10000A are in air wavelenghts
                ## This function only converts wavelengths in that range to vacuum
                ## assumes the rest are already in vacuum
                waves_nm = tab.filled()['Obs_Wave'].data
                waves_ang = 10*waves_nm
                wm_vac = air_to_vacuum(waves_ang)

            ## sort lines by wavelength
            sortd = np.argsort(wm_vac)
            srt_wm_vac, srt_fm = wm_vac[sortd], fm[sortd]
            good_waves = np.where((srt_wm_vac>=wavemincut)&(srt_wm_vac<=wavemaxcut))[0]
            out_wm_vac,out_fm_vac = srt_wm_vac[good_waves], srt_fm[good_waves]
            linelistdict[lamp] = (out_wm_vac,out_fm_vac)

    return linelistdict