'''
Next to do: Figure out how to keep consistant slits in each slice
'''


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits as pyfits
from scipy.optimize import curve_fit
from matplotlib.widgets import  RectangleSelector
import warnings
import pdb

#warnings.filterwarnings("ignore")
binnedx = 2070    # 4064    # this is in binned pixels
binnedy = 1256    # this is in binned pixels
binxpix_mid = int(binnedx/2)
binypix_mid = int(binnedy/2)


def _quadfit(x,a,b):
    '''define quadratic galaxy fitting function'''
    return a*(x-binxpix_mid)**2 + b

def _gaus(x,a,x0,c):
    if a <= 0: a = np.inf
    return a*np.exp(-(x-x0)**2/(2*4.0**2)) + c

def chip_background(pixels,flux):
    """
    Params:
    ------
    pixels (array-like): a vertical slice through the chip. Values likely range between 0 - 4064

    flux (array-like): the corresponding flux values to pixels in the vertical slice

    Returns:
    -------
    background (array-like): An array with background values with the same length as the input arrays.
                                values are binned and interpolated.
    """
    '''
    binsize = 10
    bins = np.arange(0,len(pixels)+binsize,binsize)
    
    #shifting median-filter
    medvals = []
    for i in range(len(bins)):
        if bins[i] >= 80: lower = binsize*i - 80
        else: lower = 0
        upper = binsize*i+80
        medvals.append(np.median(np.sort(flux[lower:upper])[:40]))
    I = interp1d(bins,medvals)
    return I(pixels)
    '''
    return np.median(np.sort(flux)[:10])

def identify_slits(pixels,flux,slit_y,slitsize = 40,n_emptypixs = 5,good_detect=True):
    """
    """
    half_nepixs = (n_emptypixs-1.)/2.
    diff = flux[n_emptypixs:] - flux[:-n_emptypixs]
    diffpix = pixels[half_nepixs:][:diff.size]
    maxdiff = np.max(diff[diffpix<flux.shape[0]/2.0])
    rmaxdiff = np.min(diff[diffpix>flux.shape[0]/2.0])
    start = []
    end = []
    for i in range(len(pixels)-n_emptypixs):
        j = i+1
        if diff[i] > maxdiff*0.80:
            if len(start) > 0:
                if pixels[j]+half_nepixs > ((2*n_emptypixs)+start[-1]):
                    start.append(pixels[j]+half_nepixs)
                else: pass
            else: start.append(pixels[j]+half_nepixs)
        elif diff[i] < rmaxdiff*0.80:
            if len(end) > 0:
                if pixels[j]+half_nepixs > ((2*n_emptypixs)+end[-1]):
                    end.append(pixels[j]+half_nepixs)
                else: pass
            else: end.append(pixels[j]+half_nepixs)

    #pdb.set_trace()
    start = np.array(start)[np.array(start) < len(pixels) - slitsize]
    end = np.array(end)[np.array(end)> start[0] + slitsize - n_emptypixs]
    if len(start) > len(end):
        if slit_y < binypix_mid:
            startf = start[:1]
        else:
            startf = start[1:]
        #else:
        #    startf = start[1:]
        endf = end
    elif len(end) > len(start):
        startf = start
        if end[0] < startf[0]:
            endf = np.array(end)[end>startf[0] + slitsize - n_emptypixs]
        else:
            endf = end[:1]
    else:
        startf = start
        endf = end
    try:
        assert len(startf) == 1 and len(endf) == 1, 'Bad slit bounds'
    except:
        if len(startf) > len(endf) and len(endf) == 1:
            diff = np.abs(slitsize - n_emptypixs - (endf[0] - np.array(startf)))
            return np.array(startf)[diff == np.min(diff)],endf
        elif len(endf) > len(startf) and len(startf) == 1:
            diff = np.abs(slitsize - n_emptypixs - (np.array(endf) - startf[0]))
            return startf,np.array(endf)[diff == np.min(diff)]
        else:
            return [0],[0]
    if startf[0] > endf[0]:
        endf = [startf[0] + slitsize - n_emptypixs]

    return startf,endf


def slit_find(flux,science_flux,arc_flux,lower_lim,upper_lim,slitsize = 40,n_emptypixs = 5,slit_yloc = 300):
    ##
    #Idenfity slit position as function of x
    ##
    slicesize = 10
    startingcol = 500
    endingcol = flux.shape[1]-np.mod(flux.shape[1],slicesize)-slicesize
    nslices = int((endingcol - startingcol)/slicesize)-1
    first = []
    last = []
    pixels = np.arange(flux.shape[1])
    flux = np.log(flux)
    fig,ax = plt.subplots(1)
    ax.imshow(flux - chip_background(pixels,flux),aspect=25)
    for i in range(nslices):
        flux2 = np.sum(flux[:,(startingcol+i*slicesize):(startingcol+slicesize+i*slicesize)],axis=1)
        pixels2 = np.arange(len(flux2))
        start,end = identify_slits(pixels2,flux2-chip_background(pixels2,flux2),slit_yloc,slitsize,n_emptypixs)
        first.extend(start)
        last.extend(end)
    xpix = np.arange(startingcol,endingcol-slicesize,slicesize)
    pdb.set_trace()
    last = np.array(last)
    last = np.ma.masked_where((last<(slitsize))|(last>=flux.shape[0]),last)
    first = np.array(first)
    first = np.ma.masked_where((first<=0)|(first>=flux.shape[0]-slitsize),first)
    #pdb.set_trace()
    ax.plot(xpix,first,'b')
    ax.plot(xpix,last,'r')
    
    class FitQuad:
        def __init__(self,ax,xpix,first,last):
            self.startx,self.endx=0,0
            self.upper, = ax.plot(xpix,np.zeros(xpix.size),'g',lw=2)
            self.lower, = ax.plot(xpix,np.zeros(xpix.size),'g',lw=2)
            self.first = first
            self.last = last
            self.xpix = xpix
        
        def fitting(self,lower_lim,upper_lim):
            self.lower_lim=lower_lim
            self.upper_lim = upper_lim
            for i in range(3):
                #pdb.set_trace()
                mask = np.ma.getmask(self.first[self.lower_lim:self.upper_lim])
                if np.sum(~mask) ==0:
                    continue
                xmask = np.ma.array(self.xpix[self.lower_lim:self.upper_lim],mask=mask)
                popt,pcov = curve_fit(_quadfit,xmask.compressed(),self.first[self.lower_lim:self.upper_lim].compressed(),p0=[1e-4,50])
                self.first = np.ma.masked_where(np.abs(self.first - (popt[0]*(self.xpix-binxpix_mid)**2 + popt[1])) >= 2*n_emptypixs,self.first)
            for i in range(3):
                #pdb.set_trace()
                mask = np.ma.getmask(self.last[self.lower_lim:self.upper_lim])
                if np.sum(~mask) ==0:
                    continue
                xmask = np.ma.array(self.xpix[self.lower_lim:self.upper_lim],mask=mask)
                popt2,pcov2 = curve_fit(_quadfit,xmask.compressed(),self.last[self.lower_lim:self.upper_lim].compressed(),p0=[1e-4,50])
                self.last = np.ma.masked_where(np.abs(self.last - (popt2[0]*(self.xpix-binxpix_mid)**2 + popt2[1])) >= 2*n_emptypixs,self.last)

            self.popt_avg = [np.average([popt[0],popt2[0]]),popt[1]]
            self.slitwidth = popt2[1]-popt[1]
            self.plot_fit()
            return self.popt_avg

        def onselect(self,eclick, erelease):
            'eclick and erelease are matplotlib events at press and release'
            print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
            print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
            print ' used button   : ', eclick.button
            self.startx,self.endx=eclick.xdata,erelease.xdata
            self.startpx = np.where(xpix>self.startx)[0][0]
            self.endpx = np.where(xpix<self.endx)[0][-1]
            self.fitting(lower_lim=self.startpx,upper_lim=self.endpx)

        def plot_fit(self):
            self.upper.set_ydata(_quadfit(self.xpix,*self.popt_avg))
            self.lower.set_ydata(self.popt_avg[0]*(self.xpix-binxpix_mid)**2 + self.popt_avg[1]+self.slitwidth)
            plt.draw()
    
    
    ##
    ## Fit quadratic
    ##
    Sel = FitQuad(ax,xpix,first,last)
    Sel.fitting(lower_lim,upper_lim)
    xdat = RectangleSelector(ax, Sel.onselect, drawtype='box')
    plt.show()
    popt_avg = Sel.popt_avg
    lower_lim = Sel.lower_lim
    
    
    ##
    ## cut out slit
    ##
    d2_spectra_s = np.zeros((science_flux.shape[1],slitsize))
    d2_spectra_a = np.zeros((arc_flux.shape[1],slitsize))
    for i in range(science_flux.shape[1]):
        yvals = np.arange(0,science_flux.shape[0],1)
        d2_spectra_s[i] = science_flux[:,i][np.where((yvals>=popt_avg[0]*(i-binypix_mid)**2 + popt_avg[1])&(yvals<=popt_avg[0]*(i-binypix_mid)**2 + popt_avg[1]+slitsize+n_emptypixs))][:slitsize]
        d2_spectra_a[i] = arc_flux[:,i][np.where((yvals>=popt_avg[0]*(i-binypix_mid)**2 + popt_avg[1])&(yvals<=popt_avg[0]*(i-binypix_mid)**2 + popt_avg[1]+slitsize+n_emptypixs))][:slitsize]

    ##
    ## Identify and cut out galaxy light
    ##
    #pdb.set_trace()
    gal_guess = np.arange(0,slitsize,1)[np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1)== \
                                        np.max(np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1))][0]
    popt_g,pcov_g = curve_fit(_gaus,np.arange(0,slitsize,1),np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1),p0=[1,gal_guess,0])
    gal_pos = popt_g[1]
    gal_wid = 4.0
    #if gal_wid > 5: gal_wid=5
    
    upper_gal = gal_pos + gal_wid*2.0
    lower_gal = gal_pos - gal_wid*2.0
    if upper_gal >= slitsize: upper_gal = (slitsize-1)
    if lower_gal <= 0: lower_gal = 0
    raw_gal = d2_spectra_s.T[lower_gal:upper_gal,:]
    sky = np.append(d2_spectra_s.T[:lower_gal,:],d2_spectra_s.T[upper_gal:,:],axis=0)
    sky_sub = np.zeros(raw_gal.shape) + np.median(sky,axis=0)
    sky_sub_tot = np.zeros(d2_spectra_s.T.shape) + np.median(sky,axis=0)
    
    plt.imshow(np.log(d2_spectra_s.T),aspect=35,origin='lower')
    plt.axhline(lower_gal,color='k',ls='--')
    plt.axhline(upper_gal,color='k',ls='--')
    plt.xlim(0,binnedx)
    plt.show()

    plt.plot(np.arange(0,slitsize,1),_gaus(np.arange(0,slitsize,1),*popt_g))
    plt.plot(np.arange(0,slitsize,1),np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1))
    plt.show()
    
    print 'gal dim:',raw_gal.shape
    print 'sky dim:',sky.shape

    plt.imshow(np.log(d2_spectra_s.T-sky_sub_tot),aspect=35,origin='lower')
    plt.show()

    plt.plot(np.arange(raw_gal.shape[1]),np.sum(raw_gal-sky_sub,axis=0)[::-1])
    plt.show()

    return d2_spectra_s.T,d2_spectra_a.T,raw_gal-sky_sub,[lower_gal,upper_gal],lower_lim,upper_lim

if __name__ == '__main__':
    '''
    for i in range(2):
        hdu = pyfits.open('C4_0199/flats/flat590813.000'+str(i+1)+'b.fits')
        hdu2 = pyfits.open('C4_0199/science/C4_0199_science.000'+str(i+1)+'b.fits')
        hdu3 = pyfits.open('C4_0199/arcs/arc590813.000'+str(i+1)+'b.fits')
        if i == 0:
            X = slit_find(hdu[0].data[1470:1540,:],hdu2[0].data[1470:1540,:])
        else:
            X += slit_find(hdu[0].data[1470:1540,:],hdu2[0].data[1470:1540,:])
    plt.imshow(np.log(X),aspect=35)
    plt.show()
    '''

    hdu = pyfits.open('C4_0199/flats/C4_0199_flat.cr.fits')
    hdu2 = pyfits.open('C4_0199/science/C4_0199_science.cr.fits')
    hdu3 = pyfits.open('C4_0199/arcs/C4_0199_arc.cr.fits')
    X,gal,gal_bounds = slit_find(hdu[0].data[1470:1540,:],hdu2[0].data[1470:1540,:],hdu3[0].data[1470:1540,:])
    #plt.imshow(gal,aspect=35)
    #plt.show()
    #plt.plot(np.arange(gal.shape[1]),np.sum(gal,axis=0)[::-1])
    #plt.show()
