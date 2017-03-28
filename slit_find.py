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
import cv2
from makegif import make_gif
import os
#warnings.filterwarnings("ignore")
#binnedx = 2070    # 4064    # this is in binned pixels
#binnedy = 1256    # this is in binned pixels
#binxpix_mid = int(binnedx/2)
#binypix_mid = int(binnedy/2)



def normalized_Canny(imgdata,low = 10,high = 240):
    imgdata_0 = imgdata-np.min(imgdata)
    imgo = (255.*(imgdata_0/np.max(imgdata_0))).astype(np.uint8)
    img = cv2.medianBlur(imgo,5)
    return cv2.Canny(img,low,high)

def get_template(imgdata,xlow,xhigh,ylow,yhigh):
    imgdata_0 = imgdata-np.min(imgdata)
    imgo = (255.*(imgdata_0/np.max(imgdata_0))).astype(np.uint8)
    img = cv2.medianBlur(imgo,5)
    #templateo = imgo[xlow:xhigh,ylow:yhigh]
    template = img[xlow:xhigh,ylow:yhigh]
    return template

def match_template(imgdata,template):
     img = imgdata.copy()
     w, h = template.shape[::-1]
     res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
     threshold = 0.8
     loc = np.where( res >= threshold)
     for pt in zip(*loc[::-1]):
         cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 4)
     cv2.imwrite('res.png',img)
     plt.imshow(res)
     plt.show()
    
def bad_locs(dgalwids):
    abs_slopes = np.abs(np.gradient(dgalwids))
    return np.where(abs_slopes > 5*np.median(abs_slopes))[0] + 1   

#def _quadfit(x,a,b):
#    '''define quadratic galaxy fitting function'''
#    return a*(x-binxpix_mid)**2 + b

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

def _constrained_gaus(dx_ov_sig,amp,background):
    if amp <= 0: amp = np.inf
    # sig = 4.0
    return amp*np.exp(-0.5*dx_ov_sig*dx_ov_sig) + background



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

def identify_slits(pixels,flux,yundermid = True,slitsize = 40,n_emptypixs = 5,good_detect=True):
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
        if yundermid:
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





def slit_find(flux,science_flux,arc_flux,edges,lower_lim,upper_lim,slitsize = 40,n_emptypixs = 5,slit_yloc = 300,figure_save_loc='./',binypix_mid=1000):
    ##
    #Idenfity slit position as function of x
    ##
    first = []
    last = []
    xpix = []
    binnedx = flux.shape[1]
    binxpix_mid = int(binnedx/2)
    pixels = np.arange(flux.shape[1])
    fig,ax = plt.subplots(1)
    ax.imshow(flux - chip_background(pixels,flux),aspect=25)

    for column in np.where(np.abs(np.sum(edges,axis=0) - 2*np.max(edges)) < 511)[0]:
        edge = np.where(edges[:,column] > 0)[0]
        if type(edges) is not type(np.array([])):
            continue
        elif len(edge) == 2:
            xpix.append(column)
            first.append(edge[0]+5)#3
            last.append(edge[1])#0
         
    last = np.ma.array(last)
    xpix = np.ma.array(xpix)
    first = np.ma.array(first)
    ax.plot(xpix,first,'b')
    ax.plot(xpix,last,'r')
    
    class FitQuad:
        def __init__(self,ax,xpix,first,last,fitting_function,initial_conds,savename=''):
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
            self.savename = savename

        def fitting(self,lower_lim,upper_lim):
            self.lower_lim=lower_lim
            self.upper_lim = upper_lim
            for i in range(3):
                #pdb.set_trace()
                mask = np.ma.getmask(self.first[self.lower_lim:self.upper_lim])
                if np.sum(~mask) ==0:
                    continue
                xmask = np.ma.array(self.xpix[self.lower_lim:self.upper_lim],mask=mask)
                popt,pcov = curve_fit(self.fitting_function,xmask.compressed(),self.first[self.lower_lim:self.upper_lim].compressed(),p0=self.pof,maxfev = 1000000)
                self.first = np.ma.masked_where(np.abs(self.first - self.fitting_function(self.xpix,*popt)) >= 2*n_emptypixs,self.first)
            for i in range(3):
                #pdb.set_trace()
                mask2 = np.ma.getmask(self.last[self.lower_lim:self.upper_lim])
                if np.sum(~mask) ==0:
                    continue
                xmask2 = np.ma.array(self.xpix[self.lower_lim:self.upper_lim],mask=mask2)
                popt2,pcov2 = curve_fit(self.fitting_function,xmask2.compressed(),self.last[self.lower_lim:self.upper_lim].compressed(),p0=self.pol,maxfev = 1000000)
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
                
        def plot_fit(self):
            self.upper.set_ydata(self.fitting_function(self.xpix,*self.popt_avg))
            cop = self.popt_avg.copy()
            cop[-1] = cop[-1]+self.slitwidth
            self.lower.set_ydata(self.fitting_function(self.xpix,*cop))
            plt.draw()
            plt.savefig(self.savename)
    
    
    ##
    ## Fit quadratic
    ##
    fitting_function = _fullquadfit
    init_conds = np.asarray([1e-4,1e-4,1e-4])
    #fitting_function = _ellipsoid
    #init_conds = np.asarray([10.,10.,0.,0.])
    savename = figure_save_loc+'_2dslit_curvature_fitting.png'
    Sel = FitQuad(ax,xpix,first,last,fitting_function,init_conds,savename)
    Sel.fitting(lower_lim,upper_lim)
    xdat = RectangleSelector(ax, Sel.onselect, drawtype='box')
    plt.show()
    del xdat
    popt_avg = Sel.popt_avg
    lower_lim = Sel.lower_lim
    slit_width = np.round(Sel.slitwidth).astype(int)
    
    ##
    ## cut out slit
    ##
    d2_spectra_s = np.zeros((slit_width,science_flux.shape[1]))
    d2_spectra_a = np.zeros((slit_width,arc_flux.shape[1]))
    xvals = np.arange(0,arc_flux.shape[1],1)
    lower_yvals = fitting_function(xvals-binxpix_mid,*popt_avg)
    lower_yinds = np.round(lower_yvals).astype(int)
    higher_yinds = lower_yinds + slit_width
    for i in range(science_flux.shape[1]):
        d2_spectra_s[:,i] = science_flux[lower_yinds[i]:higher_yinds[i],i]
        d2_spectra_a[:,i] = arc_flux[lower_yinds[i]:higher_yinds[i],i]
        
    #    yvals = np.arange(0,science_flux.shape[0],1)
    #    fullquadfitvals = fitting_function(i-binxpix_mid,*popt_avg)
    #    yvalmask = np.where( (yvals >= fullquadfitvals) & (yvals <= (fullquadfitvals + slit_width)) ) #  + n_emptypixs
    #    d2_spectra_s[i] = science_flux[:,i][yvalmask][:slit_width]
    #    d2_spectra_a[i] = arc_flux[:,i][yvalmask][:slit_width]

    ##
    ## Identify and cut out galaxy light
    ##
    #pdb.set_trace()
    gal_guess = np.argmax(np.median(d2_spectra_s/np.max(d2_spectra_s),axis=1))
    #_gaus(x,amp,sigma,x0,background)
    popt_g,pcov_g = curve_fit(_gaus,np.arange(0,slit_width,1),np.median(d2_spectra_s/np.max(d2_spectra_s),axis=1),p0=[1,4.0,gal_guess,0],maxfev = 1000000)
    gal_amp,gal_wid,gal_pos,sky_val = popt_g
    #gal_wid = popt_g[1]#4.0
    #if gal_wid > 5: gal_wid=5
    
    upper_gal = gal_pos + gal_wid*2.0
    lower_gal = gal_pos - gal_wid*2.0
    if upper_gal >= slit_width: upper_gal = (slit_width-2)
    if lower_gal <= 0: lower_gal = 2
    raw_gal = d2_spectra_s[lower_gal:upper_gal,:]
    sky = np.append(d2_spectra_s[:lower_gal,:],d2_spectra_s[upper_gal:,:],axis=0)
    sky_sub = np.zeros(raw_gal.shape) + np.median(sky,axis=0)
    sky_sub_tot = np.zeros(d2_spectra_s.shape) + np.median(sky,axis=0)
    #plt.figure()
    #plt.imshow(np.log(d2_spectra_s),aspect=35,origin='lower')#
    #plt.axhline(lower_gal,color='k',ls='--')
    #plt.axhline(upper_gal,color='k',ls='--')
    #plt.xlim(0,binnedx)
    #plt.savefig(figure_save_loc+'_dansgalaxyloc.png')
    #plt.show()

    plt.figure()
    plt.plot(np.arange(0,slit_width,1),_gaus(np.arange(0,slit_width,1),*popt_g))
    plt.plot(np.arange(0,slit_width,1),np.median(d2_spectra_s/np.max(d2_spectra_s),axis=1))
    plt.savefig(figure_save_loc+'_xsumd_gausfit.png')
    #plt.show()
    
    print 'gal dim:',raw_gal.shape
    print 'sky dim:',sky.shape

    skysubd = d2_spectra_s-sky_sub_tot
    skysubd[skysubd <=0] = 1e-6
    #plt.figure()
    #plt.imshow(np.log(skysubd),aspect=35,origin='lower')#aspect=35,
    #plt.title('Gal Spectra with Median Sky Subtracted')
    #plt.savefig(figure_save_loc+'_medskysubd_2dgalspec.png')

    #plt.plot(np.arange(raw_gal.shape[1]),np.sum(raw_gal-sky_sub,axis=0),'b-')
    #plt.plot(np.arange(raw_gal.shape[1]),np.median(sky,axis=0),'r-')
    
    plt.figure()
    plt.subplot(311)
    plt.title('Spectra, Median Sky, Med Sky Subd Gal Spec')
    plt.imshow(np.log(d2_spectra_s),aspect=35,origin='lower')
    plt.axhline(lower_gal,color='k',ls='--')
    plt.axhline(upper_gal,color='k',ls='--')
    plt.subplot(312)
    plt.imshow(np.log(sky_sub_tot),aspect=35,origin='lower')
    plt.subplot(313)
    plt.imshow(np.log(skysubd),aspect=35,origin='lower')
    plt.axhline(lower_gal,color='k',ls='--')
    plt.axhline(upper_gal,color='k',ls='--')
    plt.savefig(figure_save_loc+'_2dgalspec_withandwo_sky.png')
    #plt.show()
    
    
    #############################################
    # My mods
    #############################################
    #pdb.set_trace()
    ncols = d2_spectra_s.shape[1]
    
    gal_guess = np.argmax(np.median(d2_spectra_s/np.max(d2_spectra_s),axis=1))
    gal_amp,gal_pos,gal_wid,sky_val = 1,4.0,gal_guess,np.min(d2_spectra_s[0,:])
    yvals = np.arange(0,slit_width,1)
    lowerpix = xpix[0]
    upperpix = xpix[-1]
    cut_xvals = np.arange(lowerpix,upperpix)
    cutncols = len(cut_xvals)
    galposs = np.zeros(cutncols)
    galwids = np.zeros(cutncols)
    #pdb.set_trace()
    for i,col in enumerate(cut_xvals):
        try:
            popt_g,pcov_g = curve_fit(_gaus,yvals,d2_spectra_s[:,col],p0=[1,4.0,gal_guess,np.min(d2_spectra_s[:,col])],maxfev = 1000000)
            if np.abs(popt_g[2]-slit_width)<= slit_width and np.abs(popt_g[2])<=slit_width:
                gal_pos = popt_g[2]
                gal_wid = np.abs(popt_g[1])
            # else use value from previous index
        except:
            # if something breaks, implicitly use the previous iterations fit values for this index
            print i
        #print popt_g
        galwids[i],galposs[i] = gal_wid,gal_pos
    
    cutxmask = np.ones(len(cut_xvals)).astype(bool)

    for i in range(3):
        galcut = galwids[cutxmask]; 
        badds = bad_locs(galcut)
        if badds == [] or badds == np.array([]) or len(badds)<2:
            continue
        temp = cutxmask[cutxmask] 
        if badds[-1] >= temp.size:
            badds = badds[:-1]
        temp[badds] = False
        cutxmask[cutxmask]=temp
        del temp
        
    galwids_fitparams,pcov = curve_fit(_fullquadfit,cut_xvals[cutxmask],galwids[cutxmask],p0=[1e-4,1e-4,1e-4],maxfev = 1000000)  
    change_in_trues = True
    size_fin = cutxmask.sum()
    
    while change_in_trues:
        size_init = size_fin
        tempfitd_galwids = _fullquadfit(cut_xvals[cutxmask],*galwids_fitparams)
        dgalwids = tempfitd_galwids-galwids[cutxmask]
        deviants_mask = np.where(np.abs(dgalwids)>5*np.std(dgalwids))
        temp = cutxmask[cutxmask]
        temp[deviants_mask] = False
        cutxmask[cutxmask] = temp
        del temp
        galwids_fitparams,pcov = curve_fit(_fullquadfit,cut_xvals[cutxmask],galwids[cutxmask],p0=[1e-4,1e-4,1e-4],maxfev = 1000000)  
        size_fin = cutxmask.sum()
        if size_init == size_fin:
            change_in_trues = False
            
    galposs_fitparams,pcov = curve_fit(_fullquadfit,cut_xvals[cutxmask],galposs[cutxmask],p0=[1e-4,1e-4,1e-4],maxfev = 1000000)
    xvals = np.arange(ncols)
    fitd_galwids = _fullquadfit(xvals,*galwids_fitparams)
    fitd_galposs = _fullquadfit(xvals,*galposs_fitparams)    
    naivegalflux = np.zeros(ncols)
    naiveskyflux = np.zeros(ncols)
    fitgalflux = np.zeros(ncols)
    fitskyflux = np.zeros(ncols)
    fitgalamps = np.zeros(ncols)
    fitskyamps = np.zeros(ncols)
    fitgalamperrs = np.zeros(ncols)
    fitskyamperrs = np.zeros(ncols)
    totalflux = np.sum(d2_spectra_s,axis=0)
    #bad_xvals = cut_xvals[~cutxmask]ncols
    #pltnames = []
    errs = [1e6,1e6]
    #pdb.set_trace()
    for i in xvals:
        try:
            dy_over_sigmas = (yvals-fitd_galposs[i])/fitd_galwids[i]
            popt_cg,pcov_cg = curve_fit(_constrained_gaus,dy_over_sigmas,d2_spectra_s[:,i],p0=[1,np.min(d2_spectra_s[:,i])],maxfev = 1000000)
            gal_amp,sky_val = popt_cg
            errs = np.sqrt(np.diag(pcov_cg))
        except:
            # if something breaks, implicitly use the previous iterations fit values for this index
            print i
        fitgalamps[i] = gal_amp
        fitskyamps[i] = sky_val
        fitgalamperrs[i] = errs[0]
        fitskyamperrs[i] = errs[1]
        naiveskyflux[i] = slit_width*sky_val
        naivegalflux[i] = totalflux[i] - naiveskyflux[i]
        constraind_guasfit = _constrained_gaus(dy_over_sigmas,gal_amp,0.)
        fitgalflux[i] = np.sum(constraind_guasfit)
        fitskyflux[i] = totalflux[i] - fitgalflux[i]
        #plt.close()
        #if i in np.arange(400,1800,50):
        #    plt.figure()
        #    plt.plot(yvals,constraind_guasfit,'b-')
        #    plt.plot(yvals,constraind_guasfit+sky_val,'y-')
        #    plt.plot(yvals,d2_spectra_s[:,i],'g-')
        #plt.plot(yvals,constraind_guasfit,'b-')
        #plt.plot(yvals,constraind_guasfit+sky_val,'y-')
        #plt.plot(yvals,d2_spectra_s[i,:],'g-')
        #if i in bad_xvals:
        #    plt.title('Column i='+str(i)+' *Flagged Bad* fit')
        #else:
        #    plt.title('Column i='+str(i)+' fit')
        #plt_name = '_temp'+str(i)+'.png'
        #plt.savefig(plt_name)
        #pltnames.append(plt_name)
        #if i in bad_xvals:
        #    plt.show()
    plt.show()
    #make_gif(pltnames,figure_save_loc+'_gaussianfit.gif',delay=5)
    #os.system('rm ./_temp*.png')
    tempfitd_galwids = _fullquadfit(cut_xvals[cutxmask],*galwids_fitparams)
    dgalwids = tempfitd_galwids-galwids[cutxmask]
    deviants_mask = np.where(np.abs(dgalwids)>3*np.std(dgalwids))
    cutxmask[deviants_mask] = False 
    good_xvals = cut_xvals[cutxmask]
  
    maskthebaderrs = np.ones(fitgalamps.size).astype(bool)
    maskthebaderrs[fitgalamperrs == np.inf] = False
    change_in_trues = True
    size_fin = maskthebaderrs.sum()
    while change_in_trues:
        size_init = size_fin     
        dfiterrs = fitgalamperrs[maskthebaderrs]-np.median(fitgalamperrs[maskthebaderrs])
        temp = maskthebaderrs[maskthebaderrs]
        temp[dfiterrs > 5*np.std(dfiterrs)] = False
        maskthebaderrs[maskthebaderrs] = temp
        del temp
        size_fin = maskthebaderrs.sum()
        if size_init == size_fin:
            change_in_trues = False        

    plt.figure()
    plt.plot(xvals[maskthebaderrs],dfiterrs,'b-')
    plt.axhline(5*np.std(dfiterrs),color='r',linestyle='dashed')
    plt.axhline(-5*np.std(dfiterrs),color='r',linestyle='dashed')
    plt.ylabel('Fitting Error in Gal Amp')
    plt.xlabel('Pixel Loc on ccd')
    plt.title('Sigma Clip Masking for Galaxy Spectrum')
    plt.savefig(figure_save_loc+'_sigclipped_galamperr.png')
    #plt.show()
    #for i in xvals[~maskthebaderrs]:
    #    plt.figure()
    #    dy_over_sigmas = (yvals-fitd_galposs[i])/fitd_galwids[i]
    #    constraind_guasfit = _constrained_gaus(dy_over_sigmas,fitgalamps[i],0.)
    #    plt.plot(yvals,constraind_guasfit,'b-')
    #    plt.plot(yvals,constraind_guasfit+fitskyamps[i],'y-')
    #    plt.plot(yvals,d2_spectra_s[i,:],'g-')
    #    plt.title('Column i='+str(i)+' bad fit')
    #    plt.show()  


    #fullxmask = good_xvals # since xvals is just 0 to ncols, values are same as index
    plt.figure()
    plt.subplot(211)
    plt.title('Sky  Masked b = fitted  r = naive')
    plt.plot(xvals,fitskyflux,'b-')
    plt.plot(xvals,naiveskyflux,'r-')
    plt.subplot(212)
    plt.title('Galaxies  Masked b = fitted  r = naive')
    plt.plot(xvals[maskthebaderrs],fitgalflux[maskthebaderrs],'b-')
    plt.plot(xvals[maskthebaderrs],naivegalflux[maskthebaderrs],'r-')
    plt.savefig(figure_save_loc+'_skyandgalspectra_allfits.png')
    #plt.show()
    
    plt.figure()
    plt.plot(xvals[maskthebaderrs],fitgalflux[maskthebaderrs],'b-',alpha=0.4,label='Fitted')
    plt.plot(xvals[maskthebaderrs],naivegalflux[maskthebaderrs],'r-',alpha=0.4,label='Naive')
    plt.plot(xvals,np.sum(raw_gal-sky_sub,axis=0),'g-',alpha=0.4,label='Dans')
    plt.legend(loc='best')
    plt.savefig(figure_save_loc+'_galspectra_dancompare.png')
    #plt.show()
    
    plt.figure()
    plt.plot(xvals,fitskyflux,'b-.',alpha=0.4,label='Fitted')
    plt.plot(xvals,naiveskyflux,'r-.',alpha=0.4,label='Naive')
    plt.plot(xvals,np.median(sky,axis=0)*slit_width,'g-.',alpha=0.4,label='Dans')
    plt.legend(loc='best')
    plt.savefig(figure_save_loc+'_skyspectra_dancompare.png')

    plt.figure()
    plt.plot(xvals,fitd_galposs,label='fittedvals');
    plt.plot(good_xvals,galposs[cutxmask],label='maskedfromwidth')
    plt.plot(cut_xvals,galposs,label='allusedvals')
    ignore_infs = fitd_galposs[fitd_galposs != np.inf]
    fitmean = np.mean(ignore_infs)
    fivesig = 5*np.std(ignore_infs-fitmean)
    plt.ylim(fitmean-fivesig, fitmean+fivesig)
    del fitmean, fivesig, ignore_infs
    #plt.plot(xvals[maskthebaderrs],galposs[maskthebaderrs],label='maskedfromerr')
    plt.legend(loc='best')
    plt.savefig(figure_save_loc+'_galcenter_fit.png')  
  
    plt.figure()
    plt.plot(xvals,fitd_galwids,label='fittedvals')
    plt.plot(good_xvals,galwids[cutxmask],label='maskedfromwidth')
    plt.plot(cut_xvals,galwids,label='allusedvals')
    ignore_infs = fitd_galwids[fitd_galwids != np.inf]
    fitmean = np.mean(ignore_infs)
    fivesig = 5*np.std(ignore_infs-fitmean)
    plt.ylim(fitmean-fivesig, fitmean+fivesig)
    #plt.plot(xvals[maskthebaderrs],galposs[maskthebaderrs],label='maskedfromerr')
    plt.legend(loc='best')
    plt.savefig(figure_save_loc+'_galwidth_fit.png')
    plt.show()    
    
    #pdb.set_trace()
    dorf = str((raw_input("Which should we save, Dans or the Fitted? (d or f)"))).lower()
    if dorf != 'd':
        return d2_spectra_s,d2_spectra_a,fitgalflux,maskthebaderrs,[lower_gal,upper_gal],slit_width
    else:
        return d2_spectra_s,d2_spectra_a,raw_gal-sky_sub,maskthebaderrs,[lower_gal,upper_gal],slit_width




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
