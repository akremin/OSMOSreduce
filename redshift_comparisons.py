

from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt






outtab = Table.read("../data/catalogs/merged_target_lists/full_dataset_table.csv",format='ascii.csv')
mask = (outtab['SDSS_zsp'] < 1.0)
overlap = outtab[mask]



zmask = np.abs((overlap['z']-overlap['SDSS_zsp'])*(3.e5) )< 1.0e3
sdss,diffvel = overlap['SDSS_zsp'][zmask],(3.e5)*(overlap['z']-overlap['SDSS_zsp'])[zmask]
zsort = np.argsort(sdss)
sdss = sdss[zsort]
diffvel = diffvel[zsort]
med = np.median(diffvel)
mean = np.mean(diffvel)
stdev = np.std(diffvel)
mns,stds,zmids = [],[],[]
zwidth = 0.1
for zmid in np.arange(0.15,0.55,0.01):
    cut = diffvel[((sdss<(zmid+zwidth))&(sdss>=(zmid-zwidth)))]
    mns.append(np.mean(cut))
    stds.append(np.std(cut))
    zmids.append(zmid)
mns,stds,zmids = np.array(mns),np.array(stds),np.array(zmids)
smoother = np.array([1,3,7,3,1])
smoother = smoother/np.sum(smoother)
mns = np.convolve(mns,smoother,mode='same')
stds = np.convolve(stds,smoother,mode='same')
mns,stds,zmids = mns[2:-5],stds[2:-5],zmids[2:-5]

plt.figure()
plt.plot(sdss,diffvel,'.')
plt.plot(zmids,mns,color='gray',alpha=0.4,label='Mean, Global: {:.01f} km/s'.format(mean))
plt.plot(zmids,mns+stds,color='red',alpha=0.4,label=r'1$\sigma$, Global: {:.01f} km/s'.format(stdev))
plt.plot(zmids,mns-stds,color='red',alpha=0.4)
plt.axhline(0.,color='k',linestyle='--',alpha=0.4)
plt.title("Redshift Comparison",fontsize=18)
plt.xlabel("SDSS Redshift [z]",fontsize=16)
plt.ylabel("M2FS - SDSS [km/s]",fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best')



plt.figure()
plt.hist(diffvel,bins=30)
plt.axvline(0.,color='k',linestyle='--')
plt.axvline(mean,color='gray',label='Mean, ({:.01f} km/s)'.format(mean))
plt.axvline(mean+stdev,color='red')
plt.axvline(mean-stdev,color='red',label=r'1$\sigma$, ({:.01f} km/s)'.format(stdev))
plt.title("Redshift Comparison",fontsize=18)
plt.ylabel("Number",fontsize=16)
plt.xlabel("M2FS - SDSS [km/s]",fontsize=16)
plt.legend(loc='best')






zmask = np.abs((overlap['z']-overlap['SDSS_zsp'])*(3.e5) )< 200
sdss,diffvel = overlap['SDSS_zsp'][zmask],(3.e5)*(overlap['z']-overlap['SDSS_zsp'])[zmask]
zsort = np.argsort(sdss)
sdss = sdss[zsort]
diffvel = diffvel[zsort]
med = np.median(diffvel)
mean = np.mean(diffvel)
stdev = np.std(diffvel)
mns,stds,zmids = [],[],[]
zwidth = 0.1
for zmid in np.arange(0.15,0.55,0.01):
    cut = diffvel[((sdss<(zmid+zwidth))&(sdss>=(zmid-zwidth)))]
    mns.append(np.mean(cut))
    stds.append(np.std(cut))
    zmids.append(zmid)
mns,stds,zmids = np.array(mns),np.array(stds),np.array(zmids)
smoother = np.array([1,3,7,3,1])
smoother = smoother/np.sum(smoother)
mns = np.convolve(mns,smoother,mode='same')
stds = np.convolve(stds,smoother,mode='same')
mns,stds,zmids = mns[2:-5],stds[2:-5],zmids[2:-5]


plt.figure()
plt.plot(sdss,diffvel,'.')
plt.plot(zmids,mns,color='gray',alpha=0.4,label='Mean, Global: {:.01f} km/s'.format(mean))
plt.plot(zmids,mns+stds,color='red',alpha=0.4,label=r'1$\sigma$, Global: {:.01f} km/s'.format(stdev))
plt.plot(zmids,mns-stds,color='red',alpha=0.4)
plt.axhline(0.,color='k',linestyle='--',alpha=0.4)
plt.title("Redshift Comparison (Outliers Cut)",fontsize=18)
plt.xlabel("SDSS Redshift [z]",fontsize=16)
plt.ylabel("M2FS - SDSS [km/s]",fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(loc='best')


plt.figure()
plt.hist(diffvel,bins=30)
plt.axvline(0.,color='k',linestyle='--')
plt.axvline(mean,color='gray',label='Mean, ({:.01f} km/s)'.format(mean))
plt.axvline(mean+stdev,color='red')
plt.axvline(mean-stdev,color='red',label=r'1$\sigma$, ({:.01f} km/s)'.format(stdev))
plt.title("Redshift Comparison (Outliers Cut)",fontsize=18)
plt.ylabel("Number",fontsize=16)
plt.xlabel("M2FS - SDSS [km/s]",fontsize=16)
plt.legend(loc='best')


plt.show()