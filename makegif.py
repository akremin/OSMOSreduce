# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 12:15:05 2017

@author: kremin
"""

import matplotlib.pyplot as plt
import os


def make_gif(files,output,delay=100, repeat=True,**kwargs):
    """
    Uses imageMagick to produce an animated .gif from a list of
    picture files.
    """
     
    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              %(delay,loop," ".join(files),output))
    pltnames = []
    for i,z in enumerate(np.arange(0.05,1.2,0.005)):
        plt_name = os.path.join(os.getcwd(),'fit_sim','redEst_simulation_%d.png' % i)
        summary_plot(masked_wave,Flux_Science,early_type_wave[0,:],early_type_flux[0,:],z,temp[2],temp[3],plt_name,filframe,mock_photoz) 
        pltnames.append(plt_name)
        if np.abs(z-redshift_est)<0.0025:
            for j in np.arange(10):
                pltnames.append(plt_name)

            
    
    make_gif(pltnames,'fit_sim.gif',delay=5)