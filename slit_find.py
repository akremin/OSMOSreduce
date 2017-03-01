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

def _fullquadfit(dx,a,b,c):
    '''define quadratic galaxy fitting function'''
    return a*dx*dx + b*dx + c

def _ellipsoid(dx,a,b,h,k):
    '''define quadratic galaxy fitting function'''
    #return k + b*np.sqrt(1.-(((dx-h)*(dx-h))/(a*a)))
    psuedox2 = ((dx-h)*(dx-h))/(a*a)
    sqrt_expansion = 1 + 0.5*psuedox2 - (psuedox2*psuedox2)/8. + (5/16.)*(psuedox2**3) - (5/128.)*(psuedox2**4)
    return (k + b*sqrt_expansion)

def _gaus(x,amp,sigma,x0,background):
    if amp <= 0: amp = np.inf
    # sig = 4.0
    return amp*np.exp(-(x-x0)**2/(2*sigma**2)) + background

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
    #pdb.set_trace()
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
            endf = np.array(end)[end>(startf[0] + slitsize - n_emptypixs)]
        else:
            endf = end[:1]
    else:
        startf = start
        endf = end
    if len(startf) == 1 and len(endf) == 1:
        pass
    else:
        print('Bad slit bounds')
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
    #pdb.set_trace()
    slicesize = 20
    startingcol = 540
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
    #pdb.set_trace()
    last = np.array(last)
    last = np.ma.masked_where((last<(slitsize))|(last>=flux.shape[0]),last)
    first = np.array(first)
    first = np.ma.masked_where((first<=0)|(first>=flux.shape[0]-slitsize),first)
    #pdb.set_trace()
    ax.plot(xpix,first,'b')
    ax.plot(xpix,last,'r')
    
    class FitQuad:
        def __init__(self,ax,xpix,first,last,fitting_function,initial_conds):
            self.startx,self.endx=0,0
            self.upper, = ax.plot(xpix,np.zeros(xpix.size),'g',lw=2)
            self.lower, = ax.plot(xpix,np.zeros(xpix.size),'g',lw=2)
            self.first = first
            self.last = last
            self.xpix = xpix - binxpix_mid
            self.user_offset = 0.
            self.fitting_function = fitting_function
            self.slitwidth = 10
            self.pof = np.asarray(initial_conds)
            self.pol = self.pof.copy()
            self.pol[-1] = self.pol[-1]+self.slitwidth


        def fitting(self,lower_lim,upper_lim):
            self.lower_lim=lower_lim
            self.upper_lim = upper_lim
            for i in range(3):
                #pdb.set_trace()
                mask = np.ma.getmask(self.first[self.lower_lim:self.upper_lim])
                if np.sum(~mask) ==0:
                    continue
                xmask = np.ma.array(self.xpix[self.lower_lim:self.upper_lim],mask=mask)
                popt,pcov = curve_fit(self.fitting_function,xmask.compressed(),self.first[self.lower_lim:self.upper_lim].compressed(),p0=self.pof,maxfev = 100000)
                self.first = np.ma.masked_where(np.abs(self.first - self.fitting_function(self.xpix,*popt)) >= 2*n_emptypixs,self.first)
            for i in range(3):
                #pdb.set_trace()
                mask2 = np.ma.getmask(self.last[self.lower_lim:self.upper_lim])
                if np.sum(~mask) ==0:
                    continue
                xmask2 = np.ma.array(self.xpix[self.lower_lim:self.upper_lim],mask=mask2)
                popt2,pcov2 = curve_fit(self.fitting_function,xmask2.compressed(),self.last[self.lower_lim:self.upper_lim].compressed(),p0=self.pol,maxfev = 100000)
                self.last = np.ma.masked_where(np.abs(self.last - self.fitting_function(self.xpix,*popt2)) >= 2*n_emptypixs,self.last)
            self.popt_avg = 0.5*(popt+popt2)
            self.popt_avg[-1] = popt[-1]-self.user_offset
            self.slitwidth = popt2[-1]-popt[-1]+self.user_offset
            self.plot_fit()
            return self.popt_avg

        def onselect(self,eclick, erelease):
            'eclick and erelease are matplotlib events at press and release'
            print ' startposition : (%f, %f)' % (eclick.xdata, eclick.ydata)
            print ' endposition   : (%f, %f)' % (erelease.xdata, erelease.ydata)
            print ' used button   : ', eclick.button
            #if eclick.button >1:
            self.startx,self.endx=eclick.xdata,erelease.xdata
            startpx = np.where(xpix>self.startx)[0][0]
            endpx = np.where(xpix<self.endx)[0][-1]
            self.fitting(lower_lim=startpx,upper_lim=endpx)
            #else: 
            #    self.starty,self.endy=eclick.ydata,erelease.ydata
            #    self.user_offset = self.endy-self.starty
            #    #self.popt_avg[2] = self.popt_avg[2]-self.user_offset
            #    self.fitting(lower_lim=self.lower_lim,upper_lim=self.upper_lim)
                
        def plot_fit(self):
            self.upper.set_ydata(self.fitting_function(self.xpix,*self.popt_avg))
            cop = self.popt_avg.copy()
            cop[-1] = cop[-1]+self.slitwidth
            self.lower.set_ydata(self.fitting_function(self.xpix,*cop))
            plt.draw()
    
    
    ##
    ## Fit quadratic
    ##
    fitting_function = _fullquadfit
    init_conds = np.asarray([1e-4,1e-4,1e-4])
    #fitting_function = _ellipsoid
    #init_conds = np.asarray([10.,10.,0.,0.])

    
    Sel = FitQuad(ax,xpix,first,last,fitting_function,init_conds)
    Sel.fitting(lower_lim,upper_lim)
    xdat = RectangleSelector(ax, Sel.onselect, drawtype='box')
    plt.show()
    popt_avg = Sel.popt_avg
    lower_lim = Sel.lower_lim
    slit_width = np.round(Sel.slitwidth).astype(int)
    
    
    ##
    ## cut out slit
    ##
    d2_spectra_s = np.zeros((science_flux.shape[1],slit_width))
    d2_spectra_a = np.zeros((arc_flux.shape[1],slit_width))
    xvals = np.arange(0,arc_flux.shape[1],1)
    lower_yvals = fitting_function(xvals-binxpix_mid,*popt_avg)
    lower_yinds = np.round(lower_yvals).astype(int)
    higher_yinds = lower_yinds + slit_width
    for i in range(science_flux.shape[1]):
        d2_spectra_s[i] = science_flux[lower_yinds[i]:higher_yinds[i],i]
        d2_spectra_a[i] = arc_flux[lower_yinds[i]:higher_yinds[i],i]
        
    #    yvals = np.arange(0,science_flux.shape[0],1)
    #    fullquadfitvals = fitting_function(i-binxpix_mid,*popt_avg)
    #    yvalmask = np.where( (yvals >= fullquadfitvals) & (yvals <= (fullquadfitvals + slit_width)) ) #  + n_emptypixs
    #    d2_spectra_s[i] = science_flux[:,i][yvalmask][:slit_width]
    #    d2_spectra_a[i] = arc_flux[:,i][yvalmask][:slit_width]

    ##
    ## Identify and cut out galaxy light
    ##
    pdb.set_trace()
    gal_guess = np.arange(0,slit_width,1)[np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1)== \
                                        np.max(np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1))][0]
    #_gaus(x,amp,sigma,x0,background)
    popt_g,pcov_g = curve_fit(_gaus,np.arange(0,slit_width,1),np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1),p0=[1,4.0,gal_guess,0],maxfev = 10000)
    gal_amp,gal_pos,gal_wid,sky_val = popt_g
    #gal_wid = popt_g[1]#4.0
    #if gal_wid > 5: gal_wid=5
    
    upper_gal = gal_pos + gal_wid*2.0
    lower_gal = gal_pos - gal_wid*2.0
    if upper_gal >= slit_width: upper_gal = (slit_width-1)
    if lower_gal <= 0: lower_gal = 0
    raw_gal = d2_spectra_s.T[lower_gal:upper_gal,:]
    sky = np.append(d2_spectra_s.T[:lower_gal,:],d2_spectra_s.T[upper_gal:,:],axis=0)
    sky_sub = np.zeros(raw_gal.shape) + np.median(sky,axis=0)
    sky_sub_tot = np.zeros(d2_spectra_s.T.shape) + np.median(sky,axis=0)
    plt.imshow(np.log(d2_spectra_s.T),aspect=35,origin='lower')#
    plt.axhline(lower_gal,color='k',ls='--')
    plt.axhline(upper_gal,color='k',ls='--')
    plt.xlim(0,binnedx)
    plt.show()

    plt.plot(np.arange(0,slit_width,1),_gaus(np.arange(0,slit_width,1),*popt_g))
    plt.plot(np.arange(0,slit_width,1),np.median(d2_spectra_s.T/np.max(d2_spectra_s),axis=1))
    plt.show()
    
    print 'gal dim:',raw_gal.shape
    print 'sky dim:',sky.shape

    plt.imshow(np.log(d2_spectra_s.T-sky_sub_tot),aspect=35,origin='lower')#aspect=35,
    plt.show()

    plt.plot(np.arange(raw_gal.shape[1]),np.sum(raw_gal-sky_sub,axis=0),'b-')
    plt.plot(np.arange(raw_gal.shape[1]),np.median(sky,axis=0),'r-')
    plt.show()
    
    plt.figure(); 
    plt.subplot(311); 
    plt.imshow(np.log(d2_spectra_s.T),aspect=35,origin='lower');
    plt.axhline(lower_gal,color='k',ls='--'); 
    plt.axhline(upper_gal,color='k',ls='--'); 
    plt.subplot(312); 
    plt.imshow(np.log(sky_sub_tot),aspect=35,origin='lower'); 
    plt.subplot(313); 
    plt.imshow(np.log(d2_spectra_s.T-sky_sub_tot),aspect=35,origin='lower'); 
    plt.axhline(lower_gal,color='k',ls='--'); 
    plt.axhline(upper_gal,color='k',ls='--'); 
    plt.show()
    
    pdb.set_trace()
    return d2_spectra_s.T,d2_spectra_a.T,raw_gal-sky_sub,[lower_gal,upper_gal],slit_width

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
