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
from astropy import units as u
from astropy.time import Time
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.constants as consts
from astropy.cosmology import Planck13
from astroquery.vizier import Vizier
from astropy.table import Table, vstack, join


## User defined variables

def create_drilled_field_file(plate_pathname,drilled_field_name_template,
                              drilled_field_path = os.path.abspath('./'),overwrite_file=False):
    plate_file = {}
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
                    outtab = current_table['id','ra','dec','epoch','type','x','y','z','mag']
                    for col in ['id','epoch','type']:
                        outtab.rename_column(col,col.upper())
                    skycoords = SkyCoord(outtab['ra'], outtab['dec'],
                                         unit=(u.hourangle, u.deg))
                    outtab.remove_column('ra')
                    outtab.remove_column('dec')
                    racol = table.Column(data=skycoords.ra.deg, name='RA_targeted')
                    deccol = table.Column(data=skycoords.dec.deg, name='DEC_targeted')
                    outtab.add_columns([racol, deccol])
                    outname = drilled_field_name_template.format(fielddict['header']['name'])
                    drill_field_pathname = os.path.join(drilled_field_path,outname)
                    if not os.path.isfile(drill_field_pathname) or overwrite_file:
                        outtab.write(drill_field_pathname,format='ascii.csv',overwrite=True)

        else:
            continue


def create_m2fs_fiber_info_table(datapath,dataname,cams=['r','b']):
    datapathname = os.path.join(datapath,dataname)
    od = {'ID': [], 'FIBNAME': [], 'm2fs_fiberID': [], 'm2fs_fab': [], 'm2fs_CH': []}
    for cam in cams:
        fulldatapathname = datapathname.format(cam=cam)

        if not os.path.exists(fulldatapathname):
            print("Data not found at path: {}".format(fulldatapathname))
            continue

        hdr = fits.getheader(fulldatapathname,1)
        fibs = [key for key in hdr.keys() if 'FIBER' in key]
        for fib in fibs:
            id = hdr[fib]
            if id != 'unplugged' and len(hdr.comments[fib])>0:
                comm = hdr.comments[fib]
                fid, fab, ch = comm.split(' ')
                od['m2fs_fiberID'].append(fid.split('=')[1])
                od['m2fs_fab'].append(fab.split('=')[1])
                od['m2fs_CH'].append(ch.split('=')[1])
            else:
                od['m2fs_fiberID'].append('N/A')
                od['m2fs_fab'].append('N/A')
                od['m2fs_CH'].append('N/A')
            od['ID'].append(id)
            od['FIBNAME'].append(fib.replace('FIBER',cam))

    if np.all(np.array(od['m2fs_fiberID']) == 'N/A'):
        od.pop('m2fs_fiberID')
        od.pop('m2fs_fab')
        od.pop('m2fs_CH')
    otab = table.Table(od)
    return otab

def get_vizier_matches(mtl,vizier_catalogs=['sdss12']):
    if 'RA' in mtl.colnames:
        if type(mtl['RA']) is table.Table.MaskedColumn:
            short_mtl = mtl[~mtl['RA'].mask]
        else:
            short_mtl = mtl.copy()
    else:
        mtl.add_column(table.Table.Column(data=mtl['RA_targeted'].data,name='RA'))
        mtl.add_column(table.Table.Column(data=mtl['DEC_targeted'].data,name='DEC'))
        short_mtl = mtl.copy()

    skies = [('SKY' in name) for name in short_mtl['ID']]
    short_mtl = short_mtl[np.bitwise_not(skies)]

    skycoords = SkyCoord(short_mtl['RA'] * u.deg, short_mtl['DEC'] * u.deg)
    matches = None
    for i in range(len(skycoords)):
        pos = skycoords[i]
        ra, dec, name = short_mtl['RA'][i], short_mtl['DEC'][i], short_mtl['ID'][i]

        if (np.abs(ra * u.deg - pos.ra) > 0.1 * u.arcsec) or (np.abs(dec * u.deg - pos.dec) > 0.1 * u.arcsec):
            print("RA and DECS didn't match!")
            print(ra, pos.ra.deg, dec, pos.dec.deg)
            continue

        result = Vizier.query_region(pos, radius=1 * u.arcsec, catalog=vizier_catalogs)
        if len(result)>0:
            res_tab = result[0]

            if vizier_catalogs[0] == 'sdss12':
                if np.all(res_tab['zsp'].mask):
                    if np.all(res_tab['zph'].mask):
                        out_tab = res_tab
                    else:
                        cut_tab = res_tab[np.where(~res_tab['zph'].mask)]
                        if np.any(cut_tab['zph'] > -99):
                            out_tab = cut_tab[np.where(cut_tab['zph'] > -99)]
                        else:
                            cut_tab = res_tab[np.where(~res_tab['__zph_'].mask)]
                            if np.any(cut_tab['__zph_'] > -99):
                                out_tab = cut_tab[np.where(cut_tab['__zph_'] > -99)]
                            else:
                                out_tab = res_tab[np.where(~res_tab['zph'].mask)]
                else:
                    out_tab = res_tab[np.where(~res_tab['zsp'].mask)]
            else:
                out_tab = res_tab
        else:
            print("Couldn't find match for: ",name,ra,dec)
            out_tab = []

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

def load_additional_field_data(field_pathname,format='ascii.basic',header_start=1):
        return table.Table.read(field_pathname,format=format,header_start=header_start)

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


def make_mtl(io_config,science_filenum,vizier_catalogs,overwrite_field,overwrite_redshifts):
    catalog_loc = os.path.abspath(io_config['PATHS']['catalog_loc'])
    data_path = os.path.abspath(os.path.join( io_config['PATHS']['data_product_loc'] , io_config['DIRS']['oneD']))
    dataname = io_config['FILETEMPLATES']['oneds'].format(cam='{cam}',filenum=science_filenum,imtype='science')
    dataname = dataname + io_config['FILETAGS']['crrmvd'] + '.fits'

    plate_path = os.path.join(catalog_loc,io_config['DIRS']['plate'])
    plate_name = io_config['SPECIALFILES']['plate']
    print("In mask: {}".format(io_config['GENERAL']['mask_name']))
    if plate_name == 'None':
        print("No platename given")
        plate_name = None

    field_path = os.path.join(catalog_loc,io_config['DIRS']['field'])
    field_name = io_config['SPECIALFILES']['field']
    if field_name == 'None':
        print("No fieldname given")
        field_name = None

    redshifts_path = os.path.join(catalog_loc,io_config['DIRS']['redshifts'])
    redshifts_name = io_config['SPECIALFILES']['redshifts']
    if redshifts_name == 'None':
        print("No redshift filename given")
        redshifts_name == None

    targeting_path, targeting_name = None, None

    mtl_path = os.path.join(catalog_loc,io_config['DIRS']['mtl'])
    mtl_name = io_config['SPECIALFILES']['mtl']


    ## Housekeeping to make sure all the specified things are there to run
    paths = [plate_path, field_path, targeting_path, redshifts_path, mtl_path]
    names = [plate_name, field_name, targeting_name, redshifts_name, mtl_name]
    for path, filename in zip(paths, names):
        if filename is not None:
            if not os.path.exists(path):
                print("Creating {}".format(path))
                os.makedirs(path)
    del paths, names

    ## Get fiber info
    fiber_table = create_m2fs_fiber_info_table(data_path, dataname, cams=['r', 'b'])
    if fiber_table is None or len(fiber_table)== 0:
        print("No fiber table created!")

    ## Mine plate file for drilling info
    field_pathname = os.path.join(field_path, field_name)
    if plate_name is not None:
        if not os.path.exists(field_pathname) or overwrite_field:
            plate_pathname = os.path.join(plate_path, plate_name)
            create_drilled_field_file(plate_pathname, drilled_field_name_template=field_name.replace(io_config['GENERAL']['mask_name'],'{}'),
                                      drilled_field_path=field_path, overwrite_file=overwrite_field)

        try:
            field_table = table.Table.read(field_pathname,format='ascii.tab')
        except:
            field_table = table.Table.read(field_pathname, format='ascii.basic')
        if len(field_table.colnames) == 1:
            field_table = table.Table.read(field_pathname, format='ascii.basic')

        if 'RA_targeted' in field_table.colnames:
            field_table.rename_column('RA_targeted','RA')
            field_table.rename_column('DEC_targeted','DEC')
        if 'RA_drilled' in field_table.colnames:
            field_table.rename_column('RA_drilled','RA')
            field_table.rename_column('RA_drilled','DEC')
    elif field_name is not None and os.path.exists(field_pathname):
        field_table = table.Table.read(field_pathname, format='ascii.basic')#header_start=2,
        if 'RA_targeted' in field_table.colnames:
            field_table.rename_column('RA_targeted','RA')
            field_table.rename_column('DEC_targeted','DEC')
    else:
        print("Couldn't manage to open a field file through any of the available means, returning None")
        field_table = None

    ## Merge fiber and drill info
    if len(fiber_table)==0:
        print("No field file found and no plate file available for conversion.")
        print("Continuing with just the fiber data")
        observed_field_table = field_table
    else:
        try:
            observed_field_table = table.join(fiber_table, field_table, keys='ID', join_type='left')
        except:
            print("Something went wrong combining fiber table and field table.")
            if type(fiber_table) is table.Table:
                print("Fiber table: ",fiber_table.colnames,len(fiber_table))
            if field_table is not None and type(field_table) is table.Table:
                print("Field table: ",field_table.colnames,len(field_table))
            raise()

    ## If there is targeting information, merge that in as well
    if targeting_name is not None:
        full_pathname = os.path.join(targeting_path, targeting_name)
        if not os.path.exists(full_pathname):
            raise (IOError, "The targeting file doesn't exist")
        else:
            targeting = table.Table.read(full_pathname, format='ascii.csv')
            ml = table.join(observed_field_table, targeting, keys='ID', join_type='left')
    else:
        ml = observed_field_table

    ## If there is a separate redshifts file, merge that in
    ## Else query vizier (either sdss or panstarrs) to get redshifts and merge that in
    if redshifts_name is not None and os.path.exists(
            os.path.join(redshifts_path, redshifts_name)) and not overwrite_redshifts:
        full_pathname = os.path.join(redshifts_path, redshifts_name)
        redshifts = table.Table.read(full_pathname, format='ascii.csv')
        mtl = table.join(ml, redshifts, keys='ID', join_type='left')
    else:
        if 'DEC' in ml.colnames:
            dec_name = 'DEC'
        else:
            dec_name = 'DEC_targeted'
        if len(vizier_catalogs) == 1 and vizier_catalogs[0] == 'sdss12' and ml[dec_name][0] < -20:
            matches = None
        else:
            matches = get_vizier_matches(ml, vizier_catalogs)

        # print(len(fiber_table),len(drilled_field_table),len(observed_field_table),len(joined_field_table),len(mtl),len(matches))
        if matches is not None:
            full_pathname = os.path.join(redshifts_path,redshifts_name.format(zsource=vizier_catalogs[0]))
            matches.write(full_pathname, format='ascii.csv', overwrite='True')
            mtl = table.join(ml, matches, keys='ID', join_type='left')
        else:
            mtl = ml

    if 'sdss_SDSS12' not in mtl.colnames:
        mtl.add_column(Table.Column(data=['']*len(mtl),name='sdss_SDSS12'))
    all_most_interesting = ['ID', 'TARGETNAME', 'FIBNAME', 'sdss_SDSS12', 'RA', 'DEC', 'sdss_zsp', 'sdss_zph',
                            'sdss_rmag', 'MAG']
    all_cols = mtl.colnames
    final_order = []
    for name in all_most_interesting:
        if name in all_cols:
            final_order.append(name)
            all_cols.remove(name)

    ## Save the completed merged target list as a csv
    outname = os.path.join(mtl_path, mtl_name)

    mtl.meta['comments'] = []
    subtable = mtl[final_order]
    subtable.write(outname + '_selected.csv', format='ascii.csv', overwrite=True)

    final_order.extend(all_cols)
    fulltable = mtl[final_order]
    fulltable.write(outname + '_full.csv', format='ascii.csv', overwrite=True)


def make_mtlz(mtl_table,hdus, find_more_redshifts = False, outfile = 'mtlz.csv', \
                                                            vizier_catalogs = ['sdss12']):

    if len(hdus)==2:
        hdu1, hdu2 = hdus
        if len(Table(hdu1.data)) == 0 and len(Table(hdu2.data))==0:
            print("No data found!")
            print(hdus)
            raise (IOError)
        elif len(Table(hdu1.data)) == 0:
            hdu1 = hdu2.copy()
            hdu2 = None
        elif len(Table(hdu2.data)) == 0:
            hdu2 = None
    elif len(hdus)==1:
        hdu1 = hdus[0]
        hdu2 = None
    else:
        print("No data found!")
        print(hdus)
        raise(IOError)

    # apperature/FIBNUM, redshift_est, cor, template
    # ID,FIBNAME,sdss_SDSS12,RA,DEC,sdss_zsp,sdss_zph,sdss_rmag,MAG

    table1 = Table(hdu1.data)
    header1 = hdu1.header

    cam1 = str(header1['SHOE']).lower()
    if 'apperature' in table1.colnames:
        if len(table1) > 0 and str(table1['apperature'][0])[0].lower() != cam1:
            print("I couldn't match the camera between the header and data table for hdu1!")
        table1.rename_column('apperature', 'FIBNAME')

    if hdu2 is not None:
        table2 = Table(hdu2.data)
        header2 = hdu2.header
        cam2 = str(header2['SHOE']).lower()
        if 'apperature' in table2.colnames:
            if len(table2) > 0 and str(table2['apperature'][0])[0].lower() != cam2:
                print("I couldn't match the camera between the header and data table for hdu2!")
            table2.rename_column('apperature', 'FIBNAME')


    mtl = Table(mtl_table)


    ra_clust,dec_clust = float(header1['RA_TARG']),float(header1['DEC_TARG'])
    cluster = SkyCoord(ra=ra_clust * u.deg, dec=dec_clust * u.deg)
    z_clust = float(header1['Z_TARG'])
    kpc_p_amin = Planck13.kpc_comoving_per_arcmin(z_clust)


    fibermap = {}
    for key, val in dict(header1).items():
        if key[:5] == 'FIBER':
            fibermap['{}{}'.format(cam1, key[5:])] = val.strip(' \t')

    for t in range(1, 9):
        for f in range(1, 17):
            testkey = 'FIBER{:d}{:02d}'.format(t, f)
            replacekey = '{}{:d}{:02d}'.format(cam1,t, f)
            if testkey in table1.meta.keys():
                table1.meta[replacekey] = table1.meta[testkey]
                table1.meta.pop(testkey)

    if hdu2 is not None:
        for key, val in dict(header2).items():
            if key[:5] == 'FIBER':
                fibermap['{}{}'.format(cam2, key[5:])] = val.strip(' \t')
        for t in range(1, 9):
            for f in range(1, 17):
                testkey = 'FIBER{:d}{:02d}'.format(t, f)
                replacekey = '{}{:d}{:02d}'.format(cam2,t, f)
                if testkey in table2.meta.keys():
                    table2.meta[replacekey] = table2.meta[testkey]
                    table2.meta.pop(testkey)

        for ii in range(len(mtl)):
            id = mtl['ID'][ii]
            fbnm = mtl['FIBNAME'][ii]
            if fbnm not in fibermap.keys():
                print("{} not in fibermap!".format(fbnm))
            elif fibermap[fbnm].upper().strip(' \t\r\n') != id.upper().strip(' \t\r\n'):
                print(ii, fbnm, fibermap[fbnm], id)

        combined_table = vstack([table1, table2])
    else:
        combined_table = table1

    full_table = join(combined_table, mtl, 'FIBNAME', join_type='left')

    ## Add additional information
    if int(header1['UT-DATE'][:4]) > 2014:
        time = Time(header1['MJD'], format='mjd')
    else:
        time = Time(header1['UT-DATE'] + ' ' + header1['UT-TIME'])
    location = EarthLocation(lon=header1['SITELONG'] * u.deg, lat=header1['SITELAT'] * u.deg, \
                             height=header1['SITEALT'] * u.meter)
    bc_cor = cluster.radial_velocity_correction(kind='barycentric', obstime=time, location=location)
    dzb = bc_cor / consts.c

    hc_cor = cluster.radial_velocity_correction(kind='heliocentric', obstime=time, location=location)
    dzh = hc_cor / consts.c

    full_table.add_column(Table.Column(data=full_table['redshift_est']/(1+dzb), name='z_est_bary'))
    full_table.add_column(Table.Column(data=full_table['redshift_est'] / (1 + dzh), name='z_est_helio'))
    full_table.add_column(Table.Column(data=np.ones(len(full_table))*z_clust,name='z_clust_lit'))

    if type(full_table['RA'][1]) is str and ':' in full_table['RA'][1]:
        all_coords = SkyCoord(ra=full_table['RA'], dec=full_table['DEC'], unit=(u.hour, u.deg))
        newras = Table.Column(data=all_coords.icrs.ra.deg,name='RA')
        newdecs = Table.Column(data=all_coords.icrs.dec.deg,name='DEC')
        full_table.replace_column('RA',newras)
        full_table.replace_column('DEC',newdecs)
    else:
        all_coords = SkyCoord(ra=full_table['RA'], dec=full_table['DEC'], unit=(u.deg, u.deg))

    seps = cluster.separation(all_coords)
    full_table.add_column(Table.Column(data=seps.to(u.arcsec).value, name='Proj_R_asec'))
    full_table.add_column(Table.Column(data=(kpc_p_amin * seps).to(u.Mpc).value, name='Proj_R_Comoving_Mpc'))

    dvs = consts.c.to(u.km / u.s).value * (z_clust - full_table['z_est_bary']) / (1. + z_clust)
    full_table.add_column(Table.Column(data=dvs, name='velocity'))

    if find_more_redshifts:
        radius = 5 * u.Mpc / kpc_p_amin
        Vizier.ROW_LIMIT = -1
        result = Vizier.query_region(cluster, radius=radius, catalog=vizier_catalogs)
        if len(result)>0 and type(result) is not table.Table:
            res_tab = result[0]

            if np.all(res_tab['zsp'].mask):
                if np.all(res_tab['zph'].mask):
                    sdss_archive_table = res_tab
                else:
                    cut_tab = res_tab[np.where(~res_tab['zph'].mask)]
                    sdss_archive_table = cut_tab[np.where(cut_tab['zph'] > -99)]
            else:
                sdss_archive_table = res_tab[np.where(~res_tab['zsp'].mask)]

            for col in sdss_archive_table.colnames:
                sdss_archive_table.rename_column(col, 'sdss_' + col)

            for ii in np.arange(len(sdss_archive_table))[::-1]:
                if sdss_archive_table['sdss_SDSS12'][ii] in full_table['sdss_SDSS12']:
                    print("Removing: {}".format(sdss_archive_table['sdss_SDSS12'][ii]))
                    sdss_archive_table.remove_row(ii)

            sdss_archive_table.add_column(Table.Column(data=sdss_archive_table['sdss_RA_ICRS'], name='RA'))
            sdss_archive_table.add_column(Table.Column(data=sdss_archive_table['sdss_DE_ICRS'], name='DEC'))
            sdss_archive_table.add_column(Table.Column(data=['T'] * len(sdss_archive_table), name='TYPE'))
            sdss_archive_table.add_column(Table.Column(data=[2000.0] * len(sdss_archive_table), name='EPOCH'))

            all_sdss_coords = SkyCoord(ra=sdss_archive_table['sdss_RA_ICRS'], dec=sdss_archive_table['sdss_DE_ICRS'])
            seps = cluster.separation(all_sdss_coords)
            sdss_archive_table.add_column(Table.Column(data=seps.to(u.arcsec).value, name='Proj_R_asec'))
            sdss_archive_table.add_column(Table.Column(data=(kpc_p_amin * seps).to(u.Mpc).value, name='Proj_R_Comoving_Mpc'))
            sdss_archive_table.add_column(Table.Column(data=sdss_archive_table['sdss_zsp'], name='z_est_helio'))
            dvs = consts.c.to(u.km / u.s).value * (z_clust - sdss_archive_table['z_est_helio']) / (1. + z_clust)
            sdss_archive_table.add_column(Table.Column(data=dvs, name='velocity'))
            #
            #
            full_table.add_column(Table.Column(data=[False] * len(full_table), name='SDSS_only'))
            sdss_archive_table.add_column(Table.Column(data=[True] * len(sdss_archive_table), name='SDSS_only'))

            sdss_archive_table.convert_bytestring_to_unicode()
            convert = []
            for row in sdss_archive_table['sdss_q_mode']:
                convert.append(float(row.strip(' ') == '+'))

            new_sdssq_col = Table.Column(data=convert, name='sdss_q_mode')
            sdss_archive_table.replace_column('sdss_q_mode', new_sdssq_col)

            mega_table = vstack([full_table, sdss_archive_table])
        else:
            full_table.add_column(Table.Column(data=[False] * len(full_table), name='SDSS_only'))
            mega_table = full_table
    else:
        full_table.add_column(Table.Column(data=[False] * len(full_table), name='SDSS_only'))
        mega_table = full_table

    for key, val in header1.items():
        if 'FIBER' in key:
            continue
        elif 'TFORM' in key:
            continue
        elif 'TTYPE' in key:
            continue
        elif 'NAXIS' in key:
            continue
        elif key in ['BITPIX','XTENSION','PCOUNT','GCOUNT','TFIELDS','COMMENT']:
            continue
        if key in mega_table.meta.keys() and key != 'HISTORY':
            print("There was a conflicting key that I've overwritten: {}".format(key))
            print("Values of the conflict: {}  {}".format(val,mega_table.meta[key]))
        mega_table.meta[key] = val


    if 'description' in dict(mega_table.meta).keys():
        desc = mega_table.meta.pop('description')
        mega_table.meta['DESCRP'] = desc


    if 'full' not in outfile:
        outfile = outfile + '_full'
    if '.csv' not in outfile:
        outfile = outfile + '.csv'

    mega_table.write(outfile.replace('.csv', '.fits'), format='fits', overwrite=True)#,\
                     #output_verify="fix")

    mega_table.meta['comments'] = []
    mega_table.write(outfile, format='ascii.csv',overwrite=True)



if __name__ == '__main__':
    vizier_catalogs = ['sdss12']  # 'panstarrs','sdss'
    field_prefix, semester, fieldnum = 'M2FS16', 'A', '04'

    field = semester + fieldnum
    plate_name = None  # 'Kremin_2017A_A02_07_08_11.plate'
    field_name = 'kremin_{}{}{}.field'.format(field_prefix, semester, fieldnum)

    catalog_path = os.path.abspath('../../OneDrive - umich.edu/Research/M2FSReductions/catalogs')
    data_path = os.path.abspath('../../OneDrive - umich.edu/Research/M2FSReductions/' + field + '/raw_data')

    overwrite_field = True  # Ignored if the field is defined in place of a plate above

    data_filenum = '0904'  # as a 0 offset string of a4 digit number  ex '0614'

    overwrite_redshifts = False

    ## Assuming standard format outputs, nothing below this point should need to be changed
    dataname = '{cam}' + data_filenum + 'c1.fits'
    plate_path = os.path.join(catalog_path, 'plates')
    field_path = os.path.join(catalog_path, 'fields')

    if field_name is None:
        if plate_name is None:
            raise (IOError, "Either a drilled file or a target file must be supplied")
        field_name_template = '{}_field_drilled.csv'
        field_name = field_name_template.format(field_prefix + field)

    fieldtarget_path = os.path.join(catalog_path, 'fields')

    targeting_name = None
    targeting_path = os.path.join(catalog_path, 'labelmapping')

    redshifts_name = field_prefix + '_' + field + '_redshifts_' + vizier_catalogs[0] + '.csv'
    redshifts_path = os.path.join(catalog_path, 'redshifts')

    mtlz_name = 'mtlz_' + field_prefix + '_' + field
    mtlz_path = os.path.join(catalog_path, 'merged_target_lists')

    vizier_catalogs = ['sdss12']  # 'panstarrs','sdss'
    overwrite_field = True  # Ignored if the field is defined in place of a plate above
    overwrite_redshifts = False
    #main(io_config,vizier_catalogs,overwrite_field,overwrite_redshifts)