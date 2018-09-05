# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 11:10:59 2015

@author: kremin
"""


import numpy as np

 
    
def ask_to_break(itt):
    '''
    Ask to continue or to quit (results saved and file closed if quitting)
    '''
    if ((itt%10)==0 and itt!=0):
        print "\n\n\tWould you like to Quit?"
        YorN = 'b'
        while (YorN!='Y' and  YorN!='N' and YorN!='n' and YorN!='y'):
            YorN = raw_input('\tYes (y) or No (n): ')
        if (YorN=='Y' or YorN=='y'):
            return True
        else:
            return False


def combine_brz(bflux,rflux,zflux, bwave,rwave,zwave):
    brcut = 5800
    rzcut = 7600
    bgood = np.where(bwave <= brcut)[0]
    rgood = np.where((rwave>brcut)&(rwave<rzcut))[0]
    zgood = np.where(zwave >= rzcut)[0]
    outflux = np.concatenate([bflux[bgood],rflux[rgood],zflux[zgood]])
    outwave = np.concatenate([bwave[bgood],rwave[rgood],zwave[zgood]])
    return outflux,outwave
