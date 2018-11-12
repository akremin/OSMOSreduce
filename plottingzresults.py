#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:42:13 2016

@author: kremin
"""

from astropy.table import Table
import matplotlib.pyplot as plt
tab = Table.read('./bothsides_A07.txt',format='ascii.tab')
highercor = tab[tab['correlation']>3.1]
hhcor = highercor[highercor['zbest']>0.25]
hmhcor = hhcor[hhcor['zbest']<0.35]
hmhhcor = hmhcor[hmhcor['correlation']>3.3]
hmmhcor = hmhcor[hmhcor['correlation']>3.2]
plt.hist(hmmhcor['zbest'],bins=20)
plt.show()
