
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2

'''
Bias 1829-1853
ThAr 1928,1934,1939
NeHgArXe 1929,1930,1935,1936,1940,1941
Science 1931,1932,1933,1937,1938
Fibermaps 1612-1620
'''

biass = np.arange(1829,1854)
thar_lamps = [1928,1934,1939]
comp_lamps = [1929,1930,1935,1936,1940,1941]
sciences = [1931,1932,1933,1937,1938]
fibermaps = np.arange(1612,1621)

masterbias = '{}_masterbias_A11_c.fits'#.format(camera))

thar = '{}_ThAr_A11_cb.fits'#.format(camera))

nehg = '{}_NeHgArXe_A11_cb.fits'#.format(camera))
science = '{}_science_A11_cb.fits'#.format(camera))
fibermap = '{}_fibermap_A11_cb.fits'#.format(camera))




import pdb
from scipy.signal import argrelextrema


def plotcanny(img,minis,maxis):
    mini,mini2,minstep = minis
    maxi,maxi2,maxstep = maxis
    xplots = int((mini2-mini)/minstep)
    yplots = int((maxi2-maxi)/maxstep)
    if xplots > 3:
        xplots = 3
    if yplots > 3:
        yplots = 3
    for i in range(xplots):
        for j in range(yplots):
            thismin = mini+i*minstep
            thismax = maxi+j*maxstep
            if thismin > mini2:
                thismin = mini2
            if thismax > maxi2:
                thismax = maxi2
            edges = cv2.Canny(img,thismin,thismax)
            subval = xplots*100+yplots*10+i*xplots+j+1
            print subval
            plt.subplot(subval),plt.imshow(edges,cmap = 'gray')
            plt.title('Edge Image'+str(thismin)+' '+str(thismax)), plt.xticks([]), plt.yticks([])
    plt.show()

def thresh(img,abscut,adapt1,adapt2):
    ret,th1 = cv2.threshold(img,abscut,255,cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY,adapt1,adapt2)
    th3 = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,adapt1,adapt2)
    titles = ['Original Image', 'Global Thresholding (v = 127)',\
        'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding']
    images = [img, th1, th2, th3]
    for i in xrange(4):
        plt.subplot(2,2,i+1),plt.imshow(images[i],'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()

def plotcannysingle(img,minis,maxis,window):
    mini,mini2,minstep = minis
    maxi,maxi2,maxstep = maxis
    xplots = int((mini2-mini)/minstep)
    yplots = int((maxi2-maxi)/maxstep)
    for i in range(xplots):
        for j in range(yplots):
            thismin = mini+i*minstep
            thismax = maxi+j*maxstep
            if thismin > thismax:
                continue
            if thismin > mini2:
                thismin = mini2
            if thismax > maxi2:
                thismax = maxi2
            edges = cv2.Canny(img,thismin,thismax,window)
            #subval = xplots*100+yplots*10+i*xplots+j+1
            #print subval
            plt.subplot(111),plt.imshow(edges,cmap = 'gray')
            plt.title('Edge Image'+str(thismin)+' '+str(thismax)), plt.xticks([]), plt.yticks([])
            plt.show()

def sobel(img):
    # Output dtype = cv2.CV_8U
    sobelx8u = cv2.Sobel(img,cv2.CV_8U,1,0,ksize=5)
    # Output dtype = cv2.CV_64F. Then take its absolute and convert to cv2.CV_8U
    sobelx64f = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    abs_sobel64f = np.absolute(sobelx64f)
    sobel_8u = np.uint8(abs_sobel64f)
    plt.subplot(1,3,1),plt.imshow(img,cmap = 'gray')
    plt.title('Original'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,2),plt.imshow(sobelx8u,cmap = 'gray')
    plt.title('Sobel CV_8U'), plt.xticks([]), plt.yticks([])
    plt.subplot(1,3,3),plt.imshow(sobel_8u,cmap = 'gray')
    plt.title('Sobel abs(CV_64F)'), plt.xticks([]), plt.yticks([])
    plt.show()


def edges(img):
    color = 'hot' #'gray'
    canny = cv2.Canny(img,80,180,5)
    laplacian = cv2.Laplacian(img,cv2.CV_64F)
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=5)
    plt.subplot(2,2,1),plt.imshow(canny,cmap = color)
    plt.title('Canny'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,2),plt.imshow(laplacian,cmap = color)
    plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,3),plt.imshow(sobelx,cmap = color)
    plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
    plt.subplot(2,2,4),plt.imshow(sobely,cmap = color)
    plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
    plt.show()
    return canny,laplacian,sobelx,sobely

def load_image(typeoffile):
    if 'HOSTNAME' in os.environ.keys() and os.environ['HOSTNAME'] == 'umdes7.physics.lsa.umich.edu':
        data_dir =  'goodman_jan17'#
    else:
        data_dir = 'SOAR_data'

    if typeoffile[:3].lower() == 'sci':
        filename = '../../'+data_dir+'/Kremin10/mask1/data_products/science/Kremin10_science.cr.fits'
    if typeoffile[:3].lower() == 'arc':
        filename = '../../'+data_dir+'/Kremin10/mask1/data_products/comp/Kremin10_arc.cr.fits'
    if typeoffile[:3].lower() == 'com':
         filename = '../../'+data_dir+'/Kremin10/mask1/data_products/comp/Kremin10_arc.cr.fits'
    if typeoffile[:3].lower() == 'fla':
        filename ='../../'+data_dir+'/Kremin10/mask1/data_products/flat/Kremin10_flat.cr.fits'
    ft = fits.open(filename)
    imgdata = ft[0].data
    ft.close()
    imgdata_0 = imgdata-np.min(imgdata)
    imgo = (255.*(imgdata_0/np.max(imgdata_0))).astype(np.uint8)
    img = cv2.medianBlur(imgo,5)
    templateo = imgo[572:578,1200:-400]
    template = img[572:578,1200:-400]
    return imgo,img,templateo,template


def load_file(filename):
    ft = fits.open(filename)
    imgdata = ft[0].data
    if len(imgdata.shape)==3:
        imgdata = imgdata[0]
    #pdb.set_trace()
    ft.close()
    for i in range(100):
        imgdata = cv2.medianBlur(imgdata,5)
    imgdata_0 = imgdata-np.min(imgdata)
    imgo = (255.*(imgdata_0/np.max(imgdata_0))).astype(np.uint8)
    img = cv2.medianBlur(imgo,5)
    templateo = imgo[572:578,1200:-400]
    template = img[572:578,1200:-400]
    return imgo,img,templateo,template


def matched_filter(imgg,template):
     img = imgg.copy()
     w, h = template.shape[::-1]
     res = cv2.matchTemplate(img,template,cv2.TM_CCOEFF_NORMED)
     threshold = 0.8
     loc = np.where( res >= threshold)
     for pt in zip(*loc[::-1]):
         cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), 255, 4)
     cv2.imwrite('res.png',img)
     plt.imshow(res)
     plt.show()


for camera in ['b']:#,'r']:
    #fibermap.format(camera)
    imag = fits.getdata(fibermap.format(camera))
    imgdata_0 = imag-np.min(imag)
    imgo = (255.*(imgdata_0/np.max(imgdata_0))).astype(np.uint8)
    img = cv2.medianBlur(imgo,5)
    #plt.figure()
    #plt.imshow(img)
    #plt.show()
    #plt.close()
    #sobel(imgo)
    #canny, laplacian, sobelx, sobely = edges(imgo)
    #plotcanny(img,(20,40,5),(200,400,20))
    #thresh(imgo, 100, 80,120)
    #sobx_abs = np.abs(sobelx)
    #med_sobx = np.median(np.median(sobx_abs,axis=1),axis=0)
    #bool_x = (sobx_abs > med_sobx)
    #imgo[bool_x] = img[bool_x]
    #canny, laplacian, sobelx, sobely = edges(imgo)
    print(imag.shape)
    stepsize = 10
    all_starts,all_ends = [], []
    ncol_start,ncol_end = [],[]
    for i in range(0,imag.shape[1],stepsize):
        cut = imag[:,i:i+stepsize]
        cut2d = np.median(cut,axis=1)
        cut2d[cut2d < 3400] = 0
        cut2d[cut2d > 6500] = 6500#.clip(3400,6500)
        boolarr = (cut2d > 0.001)
        #print(sum(boolarr))
        starts, ends = [], []
        for j in range(1,len(boolarr)-1):
            if boolarr[j] and not boolarr[j-1]:
                starts.append(j)
            elif boolarr[j] and not boolarr[j+1]:
                ends.append(j)
        all_starts.append(starts)
        all_ends.append(ends)
        ncol_start.append(len(starts))
        ncol_end.append(len(ends))

    sncols = np.max(ncol_start)
    encols = np.max(ncol_end)
    starts_np = np.ndarray(shape=(len(all_starts),sncols))
    ends_np = np.ndarray(shape=(len(all_ends), encols))
    plt.figure(0)
    for row in range(len(all_starts)):
        rowstarts = all_starts[row]
        starts_np[row,:len(rowstarts)] = rowstarts
        rowends = all_ends[row]
        ends_np[row,:len(rowends)] = rowends
        plt.plot(range(len(rowstarts)),rowstarts,'b-')
        plt.plot(range(len(rowends)),rowends,'r-')
    plt.show()
    plt.close('all')

    print(starts_np.shape,ends_np.shape)
    #print(all_starts)
    #imag.clip(0.0001,44000)
    #img[img < 15] = 0
    #plt.hist(img.ravel(),bins=100)
    #plt.hist(imag.ravel(), bins=100)
    #plt.imshow(imag)
    #plt.show()




