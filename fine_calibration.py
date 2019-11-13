






def auto_wavelength_fitting_by_lines_wrapper(input_dict):
    return auto_wavelength_fitting_by_lines(**input_dict)


def auto_wavelength_fitting_by_lines(comp, fulllinelist, coef_table, linelistdict,all_coefs,user_input='some',filenum='',\
                                     bounds=None, save_plots = True,  savetemplate_funcs='{}{}{}{}{}'.format):
    if 'ThAr' in linelistdict.keys():
        wm, fm = linelistdict['ThAr']
        app_specific = False
    else:
        app_specific = True
    comp = Table(comp)

    variances = {}
    app_fit_pix = {}
    app_fit_lambs = {}
    outlinelist = {}
    hand_fit_subset = np.array(list(all_coefs.keys()))

    cam = comp.colnames[0][0]

    if cam =='b':
        numeric_hand_fit_names = np.asarray([ (16*(9-int(fiber[1])))+int(fiber[2:]) for fiber in hand_fit_subset])
    else:
        numeric_hand_fit_names = np.asarray([ (16*int(fiber[1]))+int(fiber[2:]) for fiber in hand_fit_subset])

    coef_table = Table(coef_table)

    if cam =='b':
        numerics = np.asarray([(16 * (9 - int(fiber[1]))) + int(fiber[2:]) for fiber in comp.colnames])
    else:
        numerics = np.asarray([(16 * int(fiber[1])) + int(fiber[2:]) for fiber in comp.colnames])

    sorted = np.argsort(numerics)
    all_fibers = np.array(comp.colnames)[sorted]
    del sorted,numerics

    # ## go from outside in
    # if cam == 'r' and int(all_fibers[0][1]) > 3:
    #     all_fibers = all_fibers[::-1]
    # elif cam =='b' and int(all_fibers[0][1]) < 6:
    #     all_fibers = all_fibers[::-1]

    ## go from inside out
    if cam == 'r' and int(all_fibers[0][1]) < 4:
        all_fibers = all_fibers[::-1]
    elif cam =='b' and int(all_fibers[0][1]) > 5:
        all_fibers = all_fibers[::-1]

    upper_limit_resid = 0.4
    for itter in range(20):
        badfits = []
        maxdevs = []
        upper_limit_resid += 0.01
        for fiber in all_fibers:
            if fiber in hand_fit_subset:
                continue
            if fiber not in coef_table.colnames:
                continue
            if app_specific:
                wm,fm = linelistdict[fiber]
            coefs = np.asarray(coef_table[fiber])
            f_x = comp[fiber].data

            fibern = get_fiber_number(fibername=fiber,cam=cam)

            if len(hand_fit_subset)==0:
                adjusted_coefs_guess = coefs
            elif len(hand_fit_subset)==1:
                nearest_fib = hand_fit_subset[0]
                diffs_fib1 = np.asarray(all_coefs[nearest_fib]) - np.asarray(coef_table[nearest_fib])
                diffs_mean = diffs_fib1
                adjusted_coefs_guess = coefs + diffs_mean
            else:
                # if len(hand_fit_subset)>1:
                #     dists = np.abs(fibern-numeric_hand_fit_names)
                #     closest = np.argsort(dists)[:2]
                #     nearest_fibs = hand_fit_subset[closest]
                #     # diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                #     # diffs_fib2 = np.asarray(all_coefs[nearest_fibs[1]]) - np.asarray(coef_table[nearest_fibs[1]])
                #     # d1,d2 = dists[closest[0]], dists[closest[1]]
                #     # diffs_hfib = (d2*diffs_fib1 + d1*diffs_fib2)/(d1+d2)
                #     diffs_hfib = closest
                # else:
                #     nearest_fibs = hand_fit_subset[np.argsort(np.abs(fibern-numeric_hand_fit_names))]
                #     diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                #     diffs_hfib = diffs_fib1
                #
                # if last_fiber is None:
                #     last_fiber = nearest_fibs[0]
                #
                # nearest_fib = np.asarray(all_coefs[last_fiber]) - np.asarray(coef_table[last_fiber])
                #
                # diffs_mean = (0.5*diffs_hfib)+(0.5*nearest_fib)
                #
                # adjusted_coefs_guess = coefs+diffs_mean
                nearest_fibs = hand_fit_subset[np.argsort(np.abs(fibern - numeric_hand_fit_names))]
                diffs_fib1 = np.asarray(all_coefs[nearest_fibs[0]]) - np.asarray(coef_table[nearest_fibs[0]])
                diffs_mean = diffs_fib1
                adjusted_coefs_guess = coefs + diffs_mean

            browser = LineBrowser(wm,fm, f_x, adjusted_coefs_guess, fulllinelist, bounds=None, edge_line_distance=(-20.0),initiate=False)

            params,covs, resid = browser.fit()

            print('\n\n',fiber,'{:.2f} {:.6e} {:.6e} {:.6e} {:.6e} {:.6e}'.format(*params))
            fitlamb = np.polyval(params[::-1],np.asarray(browser.line_matches['peaks_p']))
            dlamb = fitlamb - browser.line_matches['lines']
            print("  ----> mean={}, median={}, std={}, sqrt(resid)={}".format(np.mean(dlamb),np.median(dlamb),np.std(dlamb),np.sqrt(resid)))

            if np.sqrt(resid) < upper_limit_resid:
                if save_plots:
                    template = savetemplate_funcs(cam=str(filenum) + '_', ap=fiber, imtype='calib', step='finalfit',
                                                  comment='auto')
                    browser.initiate_browser()
                    browser.create_saveplot(params, covs, template)

                all_coefs[fiber] = params
                variances[fiber] = covs.diagonal()
                outlinelist[fiber] = (wm, fm)
                app_fit_pix[fiber] = browser.line_matches['peaks_p']
                app_fit_lambs[fiber] = browser.line_matches['lines']
                # if np.sqrt(resid) < upper_limit_resid:
                numeric_hand_fit_names = np.append(numeric_hand_fit_names,fibern)
                hand_fit_subset = np.append(hand_fit_subset,fiber)
            else:
                badfits.append(fiber)
                if np.sqrt(resid) < 3.0:
                    guessed_waves = np.polyval(np.asarray(adjusted_coefs_guess)[::-1], browser.line_matches['peaks_p'])
                    lines = browser.line_matches['lines']
                    dlines = np.array(lines) - np.array(guessed_waves)

                    sorted_args = np.argsort(np.abs(dlines))
                    sorted_dlamb = np.abs(dlines)[sorted_args]
                    if (sorted_dlamb[-1] > 10.0) and sorted_dlamb[-3] < 4.0:
                        maxdev_line = lines[sorted_args[-1]]

                        if app_specific and len(wm) > 11:
                            print(
                                "\n\n\nDetermined that line={} is causing bad fit for fiber={}. Dev:{}  vs 3rd:{}\n\n\n".format( \
                                    maxdev_line, fiber, sorted_dlamb[-1], sorted_dlamb[-3]))
                            wm_loc = np.where(wm == maxdev_line)[0][0]
                            wmlist, fmlist = wm.tolist(), fm.tolist()
                            wmlist.pop(wm_loc)
                            fmlist.pop(wm_loc)
                            linelistdict[fiber] = (np.asarray(wmlist), np.asarray(fmlist))
                        else:
                            maxdevs.append(maxdev_line)

            plt.close()
            del browser

        if (not app_specific) and (len(maxdevs) > (len(all_fibers)//4)) and (len(wm) > 11):
            count = Counter(maxdevs)
            line, num = count.most_common(1)[0]
            if (num >= (len(maxdevs)//3)):
                print(
                    "\n\n\nDetermined that line={} is causing bad fits, {} of {} had problems with it out of {} fibers\n\n\n".format( \
                        line, num, len(maxdevs), len(all_fibers)))

                wm_loc = np.where(wm == line)[0][0]
                wmlist,fmlist = wm.tolist(),fm.tolist()
                wmlist.pop(wm_loc)
                fmlist.pop(wm_loc)
                wm,fm = np.asarray(wmlist),np.asarray(fmlist)

        all_fibers = np.array(badfits)[::-1]

    return all_coefs, outlinelist, app_fit_lambs, app_fit_pix, variances, all_fibers