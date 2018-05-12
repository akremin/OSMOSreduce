

import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2



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
            print(subval)
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
    for i in range(4):
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
    if 'HOSTNAME' in list(os.environ.keys()) and os.environ['HOSTNAME'] == 'umdes7.physics.lsa.umich.edu':
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