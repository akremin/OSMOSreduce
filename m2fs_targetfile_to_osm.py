# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:33:56 2016

@author: kremin
"""

import numpy as np
import os
from astropy.table import Table
from astropy.coordinates import SkyCoord
import re


plate_file = {}
with open('Kremin_02_07_08_11_Plate2.plate','r') as plate:
    for line in plate:
        if line[0] == '[':
            if ":" not in line:
                current_field = line.strip(" \t\[\]\n")
                current_key = 'header'
                plate_file[current_field] = dict(header={})
            else:
                current_key = line.strip(" \t\[\]\n").split(':')[1]
                plate_file[current_field][current_key] = []
        elif current_key == 'header':
            stripped_line = line.strip(" \t\n")
            stripped_split = [x.strip(' \t\n') for x in stripped_line.split("=")]
            if len(stripped_split)>1:
                plate_file[current_field][current_key][stripped_split[0]] = stripped_split[1] 
            else:
                plate_file[current_field][current_key][stripped_split[0]] = ''
        else:
            stripped_line = line.strip(" \t\n")
            stripped_split = re.split(r'[\s\t]*',stripped_line)
            plate_file[current_field][current_key].append(stripped_split)
        
            
for field,fielddict in list(plate_file.items()):
    sectionkeys = list(fielddict.keys())
    if 'header' in sectionkeys:
            if 'Standards' in sectionkeys and len(fielddict['Standards'])>1:
                if 'Drilled' in sectionkeys:
                    for row in fielddict['Standards'][1:]:
                        fielddict['Drilled'].append(row)
                else:
                    fielddict['Drilled'] = fielddict['Standards']
            else:
                pass
            if 'Drilled' in list(fielddict.keys()):
                current_table = Table(rows=fielddict['Drilled'][1:],names = fielddict['Drilled'][0])
                #for cur_row in fielddict['Drilled'][1:]:
                #    current_table.add_row(Table.Row(cur_row))
                outtab = current_table['id','ra','dec','epoch','type','x','y','z']
                outname = fielddict['header']['name']+'_targeted_id_ra_decs'
                if os.path.isfile('./'+outname+'.fits'):
                    yorn = eval(input("Overwrite file?"))
                    if yorn.lower() == 'y':
                        os.remove('./'+outname+'.fits')
                        os.remove('./'+outname+'.csv')
                        outtab.write(outname+'.fits',format='fits')
                        outtab.write(outname+'.csv',format='ascii.csv')
                    else:
                        pass
                else:
                    outtab.write(outname+'.fits',format='fits')
                    outtab.write(outname+'.csv',format='ascii.csv')
    else:
        continue
        
        