
import os
os_slash = os.path.sep

from testopt import *
#rom zestipy.zpy import *
from slit_find import *

def getch():
    import tty, termios
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd,termios.TCSADRAIN,old_settings)
    return ch

def filter_image(img):
    img_sm = signal.medfilt(img,5)
    sigma = 2.0
    bad = np.abs(img-img_sm) / sigma > 8.0
    img_cr = img.copy()
    img_cr[bad] = img_sm[bad]
    return img_cr

#create reduced files if they don't exist
def reduce_files(filetype,cluster_dir):
    print("Skipping bias reduction... assuming it was already done")
    #full_path = cluster_dir + filetype + os_slash
    #for fil in os.listdir(full_path):
    #    if fnmatch.fnmatch(fil, '*.fits'):
    #        if not os.path.isfile(full_path + fil[:-5]+'b.fits'):
    #            print 'Creating '+  + fil[:-5]+'b.fits'
    #            p = subprocess.Popen(code_dir + 'python procM2FS.py ' + full_path + fil,shell=True)
    #            p.wait()
    #        else:
    #            print 'Reduced '+filetype+' files exist'


import matplotlib.transforms as mtransforms

fig, ax = plt.subplots()
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
plt.title('Staight Plots' + skyfib)
plt.plot(gallams, gal_contsubd, label='Galaxy')
plt.plot(gallams, outsky, label='sky')
plt.plot(gallams, master_interp, label='master')
for fib, flux in zip(skyfibs, master_skies):
    plt.plot(skyllams, flux, label=fib, alpha=0.5)
ax.fill_between(gallams, 0, 1, where=corlog > upper, facecolor='green', alpha=0.5, transform=trans)
ax.fill_between(gallams, 0, 1, where=corlog < lower, facecolor='red', alpha=0.5, transform=trans)
figManager = plt.get_current_fig_manager()
figManager.window.showMaximized()
plt.tight_layout()
plt.legend(loc='best')
# raise()







x = np.arange(0.0, 2, 0.01)
y1 = np.sin(2*np.pi*x)
y2 = 1.2*np.sin(4*np.pi*x)


# show how to use transforms to create axes spans where a certain condition is satisfied
fig, ax = plt.subplots()
y = np.sin(4*np.pi*x)
ax.plot(x, y, color='black')

# use the data coordinates for the x-axis and the axes coordinates for the y-axis
import matplotlib.transforms as mtransforms
trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
theta = 0.9
ax.axhline(theta, color='green', lw=2, alpha=0.5)
ax.axhline(-theta, color='red', lw=2, alpha=0.5)
ax.fill_between(x, -1, 1, where=y > theta, facecolor='green', alpha=0.5, transform=trans)
ax.fill_between(x, -1, 1, where=y < -theta, facecolor='red', alpha=0.5, transform=trans)


plt.show()