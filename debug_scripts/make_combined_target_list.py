import matplotlib
matplotlib.use('Qt5Agg')
import numpy as np
import os
from astropy.table import Table,hstack,vstack

# from collections import OrderedDict
# from quickreduce_funcs import digest_filenumbers
# from create_merged_target_list import create_mtl
#
# import configparser
corcut = 0.3

def main(mtlz_path,mtlz_name,correlation_cut=0.2,summary_column_subset = True):

    maskname = mtlz_name.split('_')[2]
    print(maskname)
    tab = Table.read(os.path.join(mtlz_path,mtlz_name),format='ascii.csv')

    ## Make appropriate cuts
    if 'SDSS_only' in tab.colnames:
        if type(tab['SDSS_only'][0]) in [bool,np.bool,np.bool_]:
            boolcut = [(not row) for row in tab['SDSS_only']]
        else:
            boolcut = [row.lower()=='false' for row in tab['SDSS_only']]
        tab = tab[boolcut]
    if 'cor' in tab.colnames:
        tab = tab[tab['cor']>=correlation_cut]
    if 'ID' in tab.colnames:
        tab = tab[['GAL' in id for id in tab['ID']]]
        outnames = []
        for name in tab['ID']:
            outnames.append(name.replace('GAL', '{}-'.format(maskname)))

        tab.remove_column('ID')
        tab.add_column(Table.Column(name='TARGETID', data=outnames))

    ## Make sure the names look good
    # ra_colname_bool = np.any([col.strip(' \t\r').upper() == 'RA' for col in tab.colnames])
    # if not ra_colname_bool:
    if 'RA' not in tab.colnames:
        tab.rename_column('RA_targeted','RA')
        tab.rename_column('DEC_targeted','DEC')

    # sdss_colname_bool = np.any([col.strip(' \t\r').upper() == 'SDSS_SDSS12' for col in tab.colnames])
    # if sdss_colname_bool:
    if 'sdss_SDSS12' in tab.colnames:
        newcol = []
        for ii in range(len(tab)):
            newcol.append('SDSS'+str(tab['sdss_SDSS12'][ii]))
        tab.add_column(Table.Column(data=newcol,name='SDSS12_OBJID'))
        tab.remove_column('sdss_SDSS12')
    else:
        tab.add_column(tab.MaskedColumn(data=['']*len(tab),name='SDSS12_OBJID'))#,mask=np.ones(len(tab)).astype(bool)))
    if 'sdss_zsp' in tab.colnames:
        tab.rename_column('sdss_zsp','SDSS_zsp')
    else:
        tab.add_column(Table.MaskedColumn(data=np.zeros(len(tab)),name='SDSS_zsp'))#,mask=np.ones(len(tab)).astype(bool)))
    if 'z_est_bary' in tab.colnames:
        tab.add_column(Table.Column(data=tab['z_est_bary'].copy(),name='z'))
    if 'Proj_R_asec' in tab.colnames:
        tab.rename_column('Proj_R_asec','R [asec]')
    if 'velocity' in tab.colnames:
        tab.rename_column('velocity','v [km/s]')
    if 'FIBNAME' in tab.colnames:
        tab.rename_column('FIBNAME', 'FIBERNUM')
    ## Load up in the right order
    if summary_column_subset:
        tab = tab[['TARGETID','FIBERNUM','RA','DEC','SDSS12_OBJID','z','R [asec]','v [km/s]','SDSS_zsp']]

    if 'description' in dict(tab.meta).keys():
        desc = tab.meta.pop('description')
        tab.meta['DESCRP'] = desc

    return tab.filled()

if __name__ == '__main__':
    #catalog_loc = "/Users/kremin/M2FSdata/catalogs/merged_target_lists"
    # catalog_loc = "/home/kremin/value_storage/M2FS_analysis/data/catalogs/merged_target_lists"
    catalog_loc = "/home/kremin/M2FSdata/catalogs/merged_target_lists"
    # catalog_loc = '../data/catalogs/merged_target_lists/'
    first = True
    do_subset = False
    corcut = 0.
    for fil in os.listdir(catalog_loc):
        if 'full.csv' in fil and 'mtlz_' in fil and 'allzs_' not in fil and 'A267' not in fil:
            print(fil)
            itter_tab = main(catalog_loc, fil, correlation_cut=corcut, summary_column_subset=do_subset)
            if first:
                outtab = itter_tab.copy()
                first = False
            else:
                outtab = vstack([outtab,itter_tab])
    outname = os.path.join(catalog_loc,'full_dataset_table.{savetype}')
    outtab.write(outname.format(savetype='csv'),format='ascii.csv',overwrite=True)
    outtab.write(outname.format(savetype='fits'), format='fits',overwrite=True)
    if do_subset:
        outtab.write(outname.format(savetype='aas.tex'), format='ascii.aastex',overwrite=True)
        outtab.write(outname.format(savetype='latex'), format='ascii.latex',overwrite=True)
