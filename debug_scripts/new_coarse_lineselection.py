
cut = 0.
nattempts = 100
niters = 100
elements = ['Hg','Ne']
euc_tolerance_perpix = 0.08  # 0.14
frac_reduction,stepfrac = 0.6,0.1

## 5063.36264  9.98328419e-01 -1.51722887e-06
abound_mean, abound_hw = 5000.0, 600.0  # 5063.36264,40.0
bbound_mean, bbound_hw = 1.0, 0.1  # 9.98328419e-01, 0.01
cbound_mean, cbound_hw = 0.0, 2.0e-5  # -1.51722887e-06,2.0e-6#


select_complinelistdict = {element:complinelistdict[element] for element in elements}
wm,fm = get_wm_fm(select_complinelistdict,cut=cut)
mat_wm = wm.reshape((wm.size, 1)).T

frac = 1.0 / frac_reduction
specs_to_run = np.array(coarse_comp.colnames)#[::20]:['r216','r301','r302','r309','r310','r311','r312']
calib_coefs = {}
for attempt in range(nattempts):
    last_good = {'euc': 1e8, 'a': abound_mean, 'b': bbound_mean, 'c': cbound_mean, 'nlines': 0}
    n_baddist_skip = int(attempt)
    bad_specs = []
    for specname in specs_to_run:
        specflux,pix,ph,ppix = get_peaks(coarse_comp,specname=specname)

        mat_ppix = ppix.reshape((ppix.size,1))
        best = last_good.copy()
        best['euc'],best['nlines'] = 1e8,0
        euc_tolerance = euc_tolerance_perpix * (len(ppix)-n_baddist_skip)

        nochange = 0
        for iter in np.arange(niters):
            frac = frac_reduction * frac
            aas = make_range(best['a'],abound_hw*frac,step_fraction= stepfrac)
            bs =  make_range(best['b'],bbound_hw*frac,step_fraction= stepfrac)
            cs =  make_range(best['c'],cbound_hw*frac,step_fraction= stepfrac)

            old_bests = np.array([best['a'], best['b'], best['c']],copy=True)
            for c in cs:
                cterm = (c * mat_ppix * mat_ppix)
                for b in bs:
                    bcterm = (b * mat_ppix) + cterm
                    for a in aas:
                        guess = a + bcterm
                        dists_locs = np.argmin(np.abs(mat_wm - guess), axis=1)
                        dists = np.abs(wm[dists_locs].flatten() - guess.flatten())
                        if n_baddist_skip > 0:
                            sorted_inds = np.argsort(dists)[:-n_baddist_skip]#[:int(0.5*len(ppix))]
                            cdists = dists[sorted_inds]
                        else:
                            cdists = dists
                        euc_dist = np.sqrt( np.dot(cdists, cdists) )

                        if euc_dist < best['euc']:
                            best['euc'] = euc_dist
                            best['nlines'] = len(cdists)
                            if n_baddist_skip > 0:
                                best['lines'] = wm[dists_locs].flatten()[sorted_inds]
                                best['pix'] = ppix[sorted_inds]
                            else:
                                best['lines'] = wm[dists_locs].flatten()
                                best['pix'] = ppix
                            best['a'] = a
                            best['b'] = b
                            best['c'] = c
            if np.all(np.abs(old_bests - np.array([best['a'],best['b'],best['c']]))/old_bests < 0.00001):
                # print(specname,(old_bests - np.array([best['a'], best['b'], best['c']])) / old_bests )
                nochange += 1
            else:
                nochange = 0
            if best['euc'] < euc_tolerance or nochange > 5:
                break
        calib_coefs[specname] = best.copy()
        print(specname, best)
        if best['euc'] < 2*euc_tolerance:
            last_good = best.copy()
        if best['euc'] > 4*euc_tolerance:
            if best['nlines'] > 3: # 3 coefs, reducing nlines by 1 each iteration
                bad_specs.append(specname)
            else:
                print("{} only has 3 lines remaining. A better fit cannot be made. Not continuing with that fiber".format(specname))
            plt.figure()
            plt.title(specname)
            plt.plot(get_waves_fromdict(pix,best), specflux, label='r816')
            plt.plot(get_waves_fromdict(ppix,best), ph, 'r*')
            best_guess = get_waves_fromdict(ppix,best)
            matched_wm_inds = np.argmin(np.abs(mat_wm - guess), axis=1)
            calibflux = fm[matched_wm_inds]
            spec_adjusted_calibflux = specflux.max()*calibflux/calibflux.max()
            plt.plot(wm[matched_wm_inds],spec_adjusted_calibflux,'b^')
            plt.show()
        del best
        ## Don't divide by frac, meaning all iterations after the first
        ## will start with smaller step sizes by that factor
        frac = 1.

    specs_to_run = np.array(bad_specs,copy=True)
    if len(bad_specs) == 0:
        break