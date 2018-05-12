import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy.optimize import curve_fit

binnedx = 2070   # this is in binned pixels
binnedy = 1256    # this is in binned pixels
binxpix_mid = int(binnedx/2)
binypix_mid = int(binnedy/2)


def _gaus(x,a,x0):
    return a*np.exp(-(x-x0)**2/(2*5.0**2))

def _quadfit(x,a,b,c):
    '''define quadratic galaxy fitting function'''
    return a*(x-binxpix_mid)**2 + b*(x-binxpix_mid) + c

def gal_trace(cutout):
    means = []
    meansx = []
    #define 40 'pins' to fit quadratic function to
    for i in range(40):
        i += 1
        x = np.arange(0,len(cutout),1)
        
        #stack the flux values for the nearest 100 xpixels and take the median along 
        #the xpixels to identify galaxy light
        y = np.median(cutout[:,(100*i)-50:(100*i)+50],axis=1)

        ymg = 25.0 #approx value (guess) for galaxy
        
        try:
            popt,pcov = curve_fit(_gaus,x,y,p0=[1.0,ymg]) #fit gaussian to points
        except RuntimeError:
            popt = [np.nan,np.nan]
            pcov = [[np.inf,np.inf],[np.inf,np.inf]]

        #if not a good fit, fill with nan values
        if np.sqrt(np.diag(pcov))[1] < 5.0:
            means.append(popt[1])
        else:
            means.append(np.nan)
        meansx.append(100*i)

    means.insert(0,means[0])
    meansx.insert(0,0.0)
    means.append(means[-1])
    meansx.append(binnedx)

    means = np.array(means)
    meansx = np.array(meansx)
    
    if means[np.isfinite(means)].size >= 10: #otherwise don't trust the fit and return the entire cutout summed
        #fit quadratic equation to estimated galaxy position
        popt,pcov = curve_fit(_quadfit,meansx[np.isfinite(means)],means[np.isfinite(means)],p0=[-1e-6,1e-6,20])

        #plt.plot(meansx,means,'ko')
        #plt.plot(meansx,_quadfit(np.array(meansx),*popt))
        #plt.show()

        gal_pos = _quadfit(np.arange(0,cutout.shape[1],1),*popt)

        gal_posr = np.round(np.array(gal_pos))

        gal_sum = []
        for i in range(cutout.shape[1]):
            gal_sum.append(np.sum(cutout[gal_posr[i]-5:gal_posr[i]+5,i]))

        gal_sum = np.array(gal_sum)
    else:
        gal_sum = np.sum(cutout,axis=0)
    return gal_sum


if __name__ == '__main__':
    from astropy.io import fits as pyfits
    fits = pyfits.open('C4_0199/science/C4_0199_science.reduced.fits')
    img = fits[0].data
    plt.imshow(img,vmin=-4,vmax=14)
    plt.show()
    mini = np.round(1478.0)
    maxi = np.round(1525.0)
    cutout = img[mini:maxi]
    plt.imshow(cutout,vmin=-4,vmax=14,aspect='auto')
    plt.show()
    
    sigma = 5.0
    def gaus(x,a,x0):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))
    means = []
    meansx = []
    for i in range(40):
        i += 1
        x = np.arange(0,len(cutout),1)
        y = np.median(cutout[:,(100*i)-50:(100*i)+50],axis=1)
        ymg = 25.0
        popt,pcov = curve_fit(gaus,x,y,p0=[1.0,ymg])
        print(np.sqrt(np.diag(pcov)))
        if i % 1.0 == 0:
            plt.plot(np.arange(0,len(cutout),1),np.median(cutout[:,100*i-50:100*i+50],axis=1),'ko',markersize=4)
            plt.plot(x,gaus(x,*popt),'r',ls='--',alpha=0.6)
            plt.show()
        if np.sqrt(np.diag(pcov))[1] < 5.0:
            means.append(popt[1])
        else:
            means.append(np.nan)
        meansx.append(100*i)

    means.insert(0,means[0])
    meansx.insert(0,0.0)
    means.append(means[-1])
    meansx.append(binnedx)
    
    #define quadratic galaxy fitting function
    def quadfit(x,a,b,c):
        return a*(x-binxpix_mid)**2 + b*(x-binxpix_mid) + c
    popt,pcov = curve_fit(quadfit,np.array(meansx)[np.isfinite(means)],np.array(means)[np.isfinite(means)],p0=[-1e-6,1e-6,35])
    
    #gal_spec = gal_trace(cutout)
