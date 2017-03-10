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
    print "You are now creating the video: %s" % output
    loop = -1 if repeat else 0
    os.system('convert -delay %d -loop %d %s %s'
              %(delay,loop," ".join(files),output))
              
