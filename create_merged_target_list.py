# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 12:33:56 2016

@author: kremin
"""

import numpy as np
import os
from astropy import table
import astropy.units as u
#from astropy.table import Table
from astropy.coordinates import SkyCoord
import re
#from astroquery.sdss import SDSS
from astroquery.vizier import Vizier
import time
from astropy.io import fits

## User defined variables

vizier_catalogs = ['sdss12'] #'panstarrs','sdss'
field_prefix,semester,fieldnum = 'M2FS16','A','02'
plate_name = 'Kremin_2017A_A02_07_08_11.plate'
fieldtarget_name = 'C4DR1200'+fieldnum+'_field.txt'
data_filenum = '0614'  # as a 0 offset string  ex '0614'

field = semester+fieldnum
catalog_path = os.path.abspath('../../OneDrive - umich.edu/Research/M2FSReductions/catalogs')
data_path =  os.path.abspath('../../OneDrive - umich.edu/Research/M2FSReductions/'+field+'/raw_data')


## Assuming standard format outputs, nothing below this point should need to be changed

field_name = field_prefix+field
dataname = '{cam}'+data_filenum+'c1.fits'
plate_path = os.path.join(catalog_path,'plates')

drilled_field_name_suffix = '_field_drilled'
drilled_field_path = os.path.join(catalog_path,'fields')

fieldtarget_path = os.path.join(catalog_path,'fields')

targeting_name = None
targeting_path = os.path.join(catalog_path,'labelmapping')

redshifts_name = field_prefix + '_' + field + '_redshifts_' + vizier_catalogs[0] + '.csv'
redshifts_path = os.path.join(catalog_path,'redshifts')

mtlz_name = 'mtlz_' + field_prefix + '_' + field
mtlz_path = os.path.join(catalog_path,'merged_target_lists')


def create_drilled_field_file(plate_name,drilled_field_name_suffix,plate_path = os.path.abspath('./'),\
                              drilled_field_path = os.path.abspath('./')):
    plate_file = {}
    plate_pathname = os.path.join(plate_path,plate_name)
    with open(plate_pathname,'r') as plate:
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
                    current_table = table.Table(rows=fielddict['Drilled'][1:],names = fielddict['Drilled'][0])
                    #for cur_row in fielddict['Drilled'][1:]:
                    #    current_table.add_row(Table.Row(cur_row))
                    outtab = current_table['id','ra','dec','epoch','type','x','y','z']
                    for col in ['id','epoch','type']:
                        outtab.rename_column(col,col.upper())
                    skycoords = SkyCoord(outtab['ra'], outtab['dec'],
                                         unit=(u.hourangle, u.deg))
                    outtab.remove_column('ra')
                    outtab.remove_column('dec')
                    racol = table.Column(data=skycoords.ra.deg, name='RA_drilled')
                    deccol = table.Column(data=skycoords.dec.deg, name='DEC_drilled')
                    outtab.add_columns([racol, deccol])
                    outname = fielddict['header']['name']+drilled_field_name_suffix
                    drill_field_pathname = os.path.join(drilled_field_path,outname)
                    if os.path.isfile(drill_field_pathname+'.csv'):
                        yorn = eval(input("Overwrite file?"))
                        if yorn.lower() == 'y':
                            outtab.write(drill_field_pathname+'.csv',format='ascii.csv',overwrite=True)
                        else:
                            pass
                    else:
                        outtab.write(drill_field_pathname+'.csv',format='ascii.csv')
        else:
            continue

def create_m2fs_fiber_info_table(datapath,dataname,cams=['r','b']):
    datapathname = os.path.join(datapath,dataname)
    hrs = {}
    for cam in cams:
        hrs[cam] = fits.open(datapathname.format(cam=cam))[0].header

    od = {'ID': [], 'FIBNAME': [], 'm2fs_fiberID': [], 'm2fs_fab': [], 'm2fs_CH': []}
    fibs = [key for key in hrs[cams[0]].keys() if 'FIBER' in key]
    for cam in cams:
        hr = hrs[cam]
        for fib in fibs:
            id = hr[fib]
            if id != 'unplugged':
                comm = hr.comments[fib]
                fid, fab, ch = comm.split(' ')
                od['m2fs_fiberID'].append(fid.split('=')[1])
                od['m2fs_fab'].append(fab.split('=')[1])
                od['m2fs_CH'].append(ch.split('=')[1])
                od['ID'].append(id)
                od['FIBNAME'].append(fib.replace('FIBER',cam))

    otab = table.Table(od)
    return otab

def get_vizier_matches(short_mtl,vizier_catalogs=['sdss12']):
    skycoords = SkyCoord(short_mtl['RA'] * u.deg, short_mtl['DEC'] * u.deg)
    matches = None
    for i in range(len(skycoords)):
        pos = skycoords[i]
        ra, dec, name = short_mtl['RA'][i], short_mtl['DEC'][i], short_mtl['ID'][i]
        if (np.abs(ra * u.deg - pos.ra) > 1 * u.arcsec) or (np.abs(dec * u.deg - pos.dec) > 1 * u.arcsec):
            print("RA and DECS didn't match!")
            print(ra, pos.ra.deg, dec, pos.dec.deg)
            continue

        result = Vizier.query_region(pos, radius=1 * u.arcsec, catalog=vizier_catalogs)
        res_tab = result[0]

        if vizier_catalogs[0] == 'sdss12':
            if np.all(res_tab['zsp'].mask):
                if np.all(res_tab['zph'].mask):
                    out_tab = res_tab
                else:
                    cut_tab = res_tab[np.where(~res_tab['zph'].mask)]
                    out_tab = cut_tab[np.where(cut_tab['zph'] > -99)]
            else:
                out_tab = res_tab[np.where(~res_tab['zsp'].mask)]
        else:
            out_tab = res_tab

        if len(out_tab) > 0:
            if len(out_tab) > 1:
                out_tab = out_tab[0:1]
            for col in out_tab.colnames:
                out_tab.rename_column(col, 'sdss_' + col)
            out_tab.add_column(table.Table.Column(data=[name], name='ID'))
            if matches is None:
                matches = out_tab
            else:
                matches = table.vstack([matches, out_tab], join_type='outer', metadata_conflicts='silent')
        time.sleep(0.2)
    return matches

def load_merged_target_list(field_prefix='M2FS16',field='A02',catalog_path=os.path.abspath('./')):
    from astropy.table import Table
    mtlz_name = 'mtlz_' + field_prefix + '_' + field + '_full.csv'
    mtlz_path = os.path.join(catalog_path,'merged_target_lists')
    try:
        mtlz = Table.read(os.path.join(mtlz_path,mtlz_name),format='ascii.csv', \
                          include_names=['ID','TARGETNAME','FIBNAME','sdss_SDSS12','RA','DEC','sdss_zsp','sdss_zph'])
        return mtlz
    except:
        print("Failed to load the matched target list file")
        return None


if __name__ == '__main__':
    ## Housekeeping to make sure all the specified things are there to run
    paths = [plate_path, drilled_field_path, fieldtarget_path, targeting_path, redshifts_path, mtlz_path]
    names = [plate_name, drilled_field_name_suffix, fieldtarget_name, targeting_name, redshifts_name, mtlz_name]
    for path, filename in zip(paths, names):
        if filename is not None:
            if not os.path.exists(path):
                os.makedirs(path)
    del paths, names

    if drilled_field_name_suffix is None and fieldtarget_name is None:
        raise(IOError,"Either a drilled file or a target file must be supplied")

    ## Get fiber info
    fiber_table = create_m2fs_fiber_info_table(data_path, dataname, cams=['r', 'b'])

    ## Mine plate file for drilling info
    if drilled_field_name_suffix is None or plate_name is None:
        observed_field_table = fiber_table
    else:
        drilled_field_pathname = os.path.join(drilled_field_path,field_name+drilled_field_name_suffix+'.csv')
        plate_pathname = os.path.join(plate_path,plate_name)
        if not os.path.exists(drilled_field_pathname):
            create_drilled_field_file(plate_name, drilled_field_name_suffix, plate_path, drilled_field_path)
        drilled_field_table = table.Table.read(drilled_field_pathname)
        ## Merge fiber and drill info
        observed_field_table = table.join(fiber_table,drilled_field_table,keys='ID',join_type='left')

    ## Merge in the rest of the submitted field data
    if fieldtarget_name is not None:
        field_pathname = os.path.join(fieldtarget_path, fieldtarget_name)
        all_field_table = table.Table.read(field_pathname,format='ascii.basic',header_start=1)
        joined_field_table = table.join(left=observed_field_table,right=all_field_table,
                                    keys='ID',join_type='left',table_names= ['1','2'])
        if 'EPOCH' in observed_field_table.colnames:
            for row in joined_field_table:
                if row['EPOCH_1'] != row['EPOCH_2']:
                    raise(TypeError,"Row EPOCH didn't match! breaking")
                if row['TYPE_1'] != row['TYPE_2']:
                    raise(TypeError,"Row TYPE didn't match! breaking")
            joined_field_table.remove_column('EPOCH_2')
            joined_field_table.remove_column('TYPE_2')
            joined_field_table.rename_column('EPOCH_1','EPOCH')
            joined_field_table.rename_column('TYPE_1','TYPE')
    else:
        joined_field_table = observed_field_table


    ## If there is targeting information, merge that in as well
    if targeting_name is not None:
        full_pathname = os.path.join(targeting_path,targeting_name)
        if not os.path.exists(full_pathname):
            raise(IOError,"The targeting file doesn't exist")
        else:
            targeting = table.Table.read(full_pathname,format='ascii.csv')
            mtl = table.join(joined_field_table,targeting,keys='ID',join_type='left')
    else:
        mtl = joined_field_table


    ## If there is a separate redshifts file, merge that in
    ## Else query vizier (either sdss or panstarrs) to get redshifts and merge that in
    if redshifts_name is not None and os.path.exists(os.path.join(redshifts_path, redshifts_name)):
        full_pathname = os.path.join(redshifts_path, redshifts_name)
        redshifts = table.Table.read(full_pathname, format='ascii.csv')
        mtlz = table.join(mtl, redshifts, keys='ID', join_type='left')
    else:
        short_mtl = mtl[~mtl['RA'].mask]
        matches = get_vizier_matches(short_mtl,vizier_catalogs)
        if matches is not None:
            full_pathname = os.path.join(redshifts_path, field_name+'redshifts_'+vizier_catalogs[0]+'.csv')
            matches.write(full_pathname, format='ascii.csv',overwrite='True')
            mtlz = table.join(mtl,matches,keys='ID',join_type='left')
        else:
            mtlz = mtl

    all_most_interesting = ['ID','TARGETNAME','FIBNAME','sdss_SDSS12','RA','DEC','sdss_zsp','sdss_zph','sdss_rmag','MAG']
    all_cols = mtlz.colnames
    final_order = []
    for name in all_most_interesting:
        if name in all_cols:
            final_order.append(name)
            all_cols.remove(name)


    ## Save the completed merged target list as a csv
    outname = os.path.join(mtlz_path,mtlz_name)

    subtable = mtlz[final_order]
    subtable.write(outname+'_selected.csv',format='ascii.csv',overwrite=True)

    final_order.extend(all_cols)
    fulltable = mtlz[final_order]
    fulltable.write(outname + '_full.csv', format='ascii.csv', overwrite=True)