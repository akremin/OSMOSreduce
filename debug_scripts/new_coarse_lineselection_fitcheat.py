import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

def poly5(x,a,b,c,d,e,f):
    return a+ b*x + c*x*x + d*np.power(x,3) + \
           e * np.power(x, 4) + f*np.power(x,5)
def poly2(x,a,b,c):
    return a+ b*x + c*x*x

poly2_coefs = np.array([ 5.06336264e+03,  9.98328419e-01, -1.51722887e-06])
poly5_coefs = np.array([5063.444,1.0044487,-3.19859e-5,4.311885118e-8,-2.3264790e-11,4.2521e-15])

r816,pix,ph,ppix = get_peaks(coarse_comp,specname='r816')

poly2vals = poly2(pix,*poly2_coefs)
poly5vals = poly5(pix,*poly5_coefs)

cut = 0.
# wm,fm = get_wm_fm(complinelistdict,cut=cut)
for key in complinelistdict.keys():#['Ne','Xe']:
    chopped_complinelistdict = {key:complinelistdict[key]}
    wm,fm = get_wm_fm(chopped_complinelistdict,cut=cut)

    fm_nmd = r816.max()*fm/fm.max()
    mat_wm = wm.reshape((1,wm.size))

    graphing_wm = np.append(wm - 0.001, np.append(wm, wm + 0.001))
    graphing_fm = np.append(0 * fm_nmd, np.append(fm_nmd, 0 * fm_nmd))
    sortd = np.argsort(graphing_wm)
    graphing_wm = graphing_wm[sortd]
    graphing_fm = graphing_fm[sortd]
    # for vals,vname in zip([poly2vals,poly5vals],['Quad '+key,'Fifth '+key]):
    for vals, vname in zip([poly5vals], ['Fifth ' + key]):
        mat_peaks = vals[ppix].reshape((len(ppix),1))
        dists_locs = np.argmin(np.abs(mat_wm - mat_peaks), axis=1)
        dists = np.abs(wm[dists_locs] - vals[ppix])
        matchedw = wm[dists_locs[dists<0.5]]
        matchedf = fm[dists_locs[dists<0.5]]
        print(dists)

        plt.figure()
        plt.title(vname)
        plt.plot(vals, r816, 'r-',label='r816',alpha=0.6)
        plt.plot(vals[ppix], ph, 'r*')
        plt.plot(graphing_wm, graphing_fm,'b-',label='True',alpha=0.4)
        for line in matchedw:
            plt.axvline(line,0,np.max(r816),color='cyan',alpha=0.2)
        plt.axvline(vals.min(),0,0,color='cyan',alpha=0.6,label='Matched')
        plt.plot(matchedw,matchedf,'k^',label='Matched Peaks')
        plt.legend(loc='best')
        plt.xlim(vals.min(),vals.max())
        print(key,matchedw)
        print(key,matchedf)

plt.show()