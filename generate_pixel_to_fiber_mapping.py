# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 18:01:01 2016

@author: kremin
"""


import numpy as np


bfibermap = {}
bfibername_array = []
itterb = 0
for j in np.arange(8)+1:
   for i in np.arange(16,0,-1):
       itterb += 1
       key = "FIBER%1d%02d" % (j,i)
       bfibermap[key] = itterb
       bfibername_array.append(key)
       
       
rfibermap = {}
rfibername_array = []
itterr = 0
for j in np.arange(8,0,-1):
   for i in np.arange(16,0,-1):
       itterr += 1
       key = "FIBER%1d%02d" % (j,i)
       rfibermap[key] = itterr
       rfibername_array.append(key)