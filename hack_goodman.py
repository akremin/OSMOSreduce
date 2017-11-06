
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
import cv2
from hack_funcs import *
import scipy.signal as sci

example_flat = './example_goodman/0023.Flat_7-1.fits'


import pdb
from scipy.signal import argrelextrema





#fibermap.format(camera)
imag = fits.getdata(example_flat)[0,10:,:].astype(np.float64)
print imag.shape
imgdata_0 = imag-np.min(imag)
imgdata_0[imgdata_0 > 30000] = 30000.0
imgo = (255.*(imgdata_0/np.max(imgdata_0))).astype(np.uint8)

# plt.figure()
# plt.imshow(imag)
# plt.show()
img = cv2.medianBlur(imgo,5)
# plt.figure()
# plt.imshow(img)
# plt.show()
# plt.close()
# plt.figure()
# plt.hist(img.ravel(),bins=100)
# plt.show()
# sobel(imgo)
canny, laplacian, sobelx, sobely = edges(imgo)
print(sobely.shape,type(sobely),type(sobely[0]))
# plotcanny(img,(20,40,5),(200,400,20))
#thresh(imgo, 101, 80,120)
# sobx_abs = np.abs(sobelx)
# med_sobx = np.median(np.median(sobx_abs,axis=1),axis=0)
# bool_x = (sobx_abs > med_sobx)
# imgo[bool_x] = img[bool_x]
# canny, laplacian, sobelx, sobely_new = edges(imgo)
# plt.figure()
# plt.imshow(sobely_new-sobely)
# plt.show()


stepsize = 10
row_starts,row_ends = [], []
col_starts,col_ends = [],[]
for i in range(200,imag.shape[1]-20,stepsize):
    # take all rows and only stepsize worth of columns
    cut = sobely[:,i:i+stepsize]
    # find the median of the 10 columns for each row
    cut2d = np.median(cut,axis=1)
    cut2d[np.abs(cut2d)<800] = 0
    # if the median is still large, say it's a detection of an edge
    maxlocs = sci.argrelextrema(cut2d, np.greater)[0].tolist()
    minlocs = sci.argrelextrema(cut2d, np.less)[0].tolist()
    row_starts.extend(minlocs)
    col_starts.extend([i]*len(minlocs))
    row_ends.extend(maxlocs)
    col_ends.extend([i]*len(maxlocs))
    #boolarrtop = (cut2d > 800)
    #boolarrbot = (cut2d < -800)
    #for j in range(1,len(boolarrtop)-1):
    #    if boolarrbot[j] and not boolarrbot[j-1]:
    #        row_starts.append(j)
    #        col_starts.append(i)
    #    elif boolarrtop[j] and not boolarrtop[j+1]:
    #        row_ends.append(j)
    #        col_ends.append(i)



#sncols = np.max(ncol_start)
#encols = np.max(ncol_end)
#starts_np = np.ndarray(shape=(len(all_starts),sncols))
#ends_np = np.ndarray(shape=(len(all_ends), encols))
# plt.figure()
# for row in range(len(all_starts)):
#     rowstarts = all_starts[row]
#     starts_np[row,:len(rowstarts)] = rowstarts
#     rowends = all_ends[row]
#     ends_np[row,:len(rowends)] = rowends
#     plt.plot(range(len(rowstarts)),rowstarts,'b-')
#     plt.plot(range(len(rowends)),rowends,'r-')
#     plt.xlabel("spec number")
#     plt.ylabel("row starts/ends (b/r)")
# plt.show()
# plt.close('all')

row_starts = np.asarray(row_starts)
col_starts = np.asarray(col_starts)
row_ends = np.asarray(row_ends)
col_ends = np.asarray(col_ends)


starts_srtd_inds = np.argsort(row_starts)
ends_srtd_inds = np.argsort(row_ends)

row_starts = row_starts[starts_srtd_inds]
col_starts = col_starts[starts_srtd_inds]
row_ends = row_ends[ends_srtd_inds]
col_ends = col_ends[ends_srtd_inds]

start_diffs = np.diff(np.append(row_starts,2000))
end_diffs = np.diff(np.append(row_ends,2000))

scut_locs = np.where(start_diffs > 10)[0]
ecut_locs = np.where(end_diffs > 10)[0]


# print(row_ends[ecut_locs],len(ecut_locs))
plt.figure()
plt.imshow(imgo,origin='lower')
# plt.plot([1000]*len(scut_locs),row_starts[scut_locs],'r^')
# plt.plot([1000]*len(ecut_locs),row_ends[ecut_locs],'b^')
# plt.plot(col_starts,row_starts,'c.')
# plt.plot(col_ends,row_ends,'b.')
# plt.show()

onecut = scut_locs[0]
print(row_starts[onecut-1],row_starts[onecut],row_starts[onecut+1])

rstarts = []
cstarts = []
rstarts.append(row_starts[:scut_locs[0]+1])
cstarts.append(col_starts[:scut_locs[0]+1])
for i in range(len(scut_locs)-1):
    rstarts.append(row_starts[scut_locs[i]+1:scut_locs[i+1]+1])
    cstarts.append(col_starts[scut_locs[i]+1:scut_locs[i+1]+1])
print(np.shape(rstarts))


rends = []
cends = []
rends.append(row_ends[:ecut_locs[0]+1])
cends.append(col_ends[:ecut_locs[0]+1])
for i in range(len(ecut_locs)-1):
    rends.append(row_ends[ecut_locs[i]+1:ecut_locs[i+1]+1])
    cends.append(col_ends[ecut_locs[i]+1:ecut_locs[i+1]+1])

xs = np.arange(10,imag.shape[1]-20)
power = 2
for srowlist,scollist, erowlist, ecollist in zip(rstarts,cstarts,rends,cends):
    sparams = np.polyfit(scollist,srowlist,power,full=False,cov=False)
    eparams = np.polyfit(ecollist,erowlist,power,full=False,cov=False)
    params = (sparams+eparams)*0.5
    ys = np.ones(len(xs))*params[-1]
    for itter,param in enumerate(params[:-1][::-1]):
        yitter = param*(xs**(itter+1))
        ys += yitter
    plt.plot(xs,ys,'r-')
    plt.plot(xs,ys+sparams[-1]-params[-1],'c-')
    plt.plot(xs,ys+eparams[-1]-params[-1],'c-')
# for rowlist,collist in zip(rends,cends):
#     a,b,c,d = np.polyfit(collist,rowlist,3,full=False,cov=False)
#     ys = ((a*xs + b)*xs + c)*xs + d
#     plt.plot(xs,ys,'r-')

plt.show()