
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