
import os
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

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

for camera in ['b','r']:
    all_biass = []
    for bias in biass:
        cur_bias = fits.getdata('{}{}_c.fits'.format(camera,bias))
        all_biass.append(cur_bias)
    #master_bias = np.zeros(shape=(cur_bias.shape[0],cur_bias.shape[1],len(all_biass)))
    all_biass_np = np.asarray(all_biass)
    master_bias = np.median(all_biass_np,axis=0)
    print(master_bias.shape,all_biass_np.shape)
    del all_biass_np
    plt.figure()
    plt.imshow(master_bias)
    plt.colorbar()
    plt.show()
    plt.close()
    outhead = fits.getheader('{}{}_c.fits'.format(camera,biass[0]))
    outhead.add_history("Median Master bias done by hack_A11 101517")
    outhdu = fits.PrimaryHDU(data=master_bias,header=outhead)
    outhdu.writeto('{}_masterbias_A11_c.fits'.format(camera))

    fibms = []
    for fib in fibermaps:
        cur_fib = fits.getdata('{}{}_c.fits'.format(camera,fib))
        fibms.append(cur_fib-master_bias)
    fibm_np = np.asarray(fibms)
    out_fibm = fibm_np.sum(axis=0)
    outhead = fits.getheader('{}{}_c.fits'.format(camera,fibermaps[0]))
    outhead.add_history("Median Bias Sudb by hack_A11 101517")
    outhdu = fits.HDUList(hdus=[fits.PrimaryHDU(data=out_fibm,header=outhead)])
    outhdu.writeto('{}_fibermap_A11_cb.fits'.format(camera))

    thars = []
    for thar in thar_lamps:
        cur_thar = fits.getdata('{}{}_c.fits'.format(camera,thar))
        thars.append(cur_thar-master_bias)
    thars_np = np.asarray(thars)
    out_thar = thars_np.sum(axis=0)
    outhead = fits.getheader('{}{}_c.fits'.format(camera,thar_lamps[0]))
    outhead.add_history("Median Bias Sudb by hack_A11 101517")
    outhdu = fits.HDUList(hdus=[fits.PrimaryHDU(data=out_thar,header=outhead)])
    outhdu.writeto('{}_ThAr_A11_cb.fits'.format(camera))

    comps = []
    for comp in comp_lamps:
        cur_comp = fits.getdata('{}{}_c.fits'.format(camera,comp))
        comps.append(cur_comp-master_bias)
    comps_np = np.asarray(comps)
    out_comp = comps_np.sum(axis=0)
    outhead = fits.getheader('{}{}_c.fits'.format(camera,comp_lamps[0]))
    outhead.add_history("Median Bias Sudb by hack_A11 101517")
    outhdu = fits.HDUList(hdus=[fits.PrimaryHDU(data=out_comp,header=outhead)])
    outhdu.writeto('{}_NeHgArXe_A11_cb.fits'.format(camera))

    science_stack = []
    for science in sciences:
        cur_science = fits.getdata('{}{}_c.fits'.format(camera, science))
        science_stack.append(cur_science - master_bias)
    sciences_np = np.asarray(science_stack)
    out_science = sciences_np.sum(axis=0)
    outhead = fits.getheader('{}{}_c.fits'.format(camera, sciences[0]))
    outhead.add_history("Median Bias Sudb by hack_A11 101517")
    outhdu = fits.HDUList(hdus=[fits.PrimaryHDU(data=out_science, header = outhead)])
    outhdu.writeto('{}_science_A11_cb.fits'.format(camera))