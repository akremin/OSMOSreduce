from scipy.signal import find_peaks
from scipy.optimize import curve_fit

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