import numpy as np
import matplotlib.pyplot as plt

from testopt import air_to_vacuum


def gauss(span=5,width=1):
    xs = np.arange(-span,span)
    return np.exp(-xs*xs/(2*width*width))

wm_HgNe, fm_HgNe = np.loadtxt('osmos_HgNe.dat', usecols=(0, 2), unpack=True)
wm_HgNe = air_to_vacuum(wm_HgNe)

scale_factor = 10
fake_spectra_waves = np.arange(3000*scale_factor,8000*scale_factor)
fake_spectra_fluxs = np.zeros(len(fake_spectra_waves))

span=5
width=1
generic_gauss = gauss(span=span*scale_factor,width=width)

scaledwave_to_ind = fake_spectra_waves[0]

for lamb,flux in zip(wm_HgNe,fm_HgNe):
    lambda_ind = int(lamb*scale_factor)-scaledwave_to_ind
    ind_range = np.arange(lambda_ind-scale_factor*span,lambda_ind+scale_factor*span).astype(int)
    fake_spectra_fluxs[ind_range] += flux* generic_gauss

plt.plot(fake_spectra_waves/scale_factor,fake_spectra_fluxs,'b-')
plt.plot(wm_HgNe,fm_HgNe,'ro')
plt.show()

