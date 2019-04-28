import matplotlib
matplotlib.use('Qt5Agg')

import os
from astropy.table import Table,hstack,vstack

from collections import OrderedDict
from quickreduce_funcs import digest_filenumbers
from create_merged_target_list import create_mtl

import configparser

def main(mtl_path,mtl_name):
    maskname = mtl_name.split('_')[2]

    tab = Table.read(os.path.join(mtl_path,mtl_name),format='ascii.csv')
    if 'RA' not in tab.colnames:
        tab.rename_column('RA_targeted','RA')
        tab.rename_column('DEC_targeted','DEC')

    tab = tab[['ID','FIBNAME','RA','DEC','MAG']]


    tab = tab[['GAL' in id for id in tab['ID']]]

    outnames = []
    for name in tab['ID']:
        outnames.append(name.replace('GAL','{}-'.format(maskname)))

    tab.remove_column('ID')
    tab.add_column(Table.Column(name='TARGETID',data=outnames))
    tab.rename_column('MAG','SDSS_rmag')
    tab.rename_column('FIBNAME','FIBERNUM')
    return tab[['TARGETID','RA','DEC','SDSS_rmag','FIBERNUM']]

if __name__ == '__main__':
    catalog_loc = '/nfs/kremin/M2FS_analysis/data/catalogs/merged_target_lists/'
    first = True
    for fil in os.listdir(catalog_loc):
        if 'full.csv' in fil and 'mtl_' in fil:
            if first:
                outtab = main(catalog_loc,fil)
                first = False
            else:
                itter_tab = main(catalog_loc,fil)
                outtab = vstack([outtab,itter_tab])
    outname = os.path.join(catalog_loc,'full_dataset_table.{savetype}')
    outtab.write(outname.format(savetype='csv'),format='ascii.csv',overwrite=True)
    outtab.write(outname.format(savetype='fits'), format='fits',overwrite=True)
    outtab.write(outname.format(savetype='aas.tex'), format='ascii.aastex',overwrite=True)
    outtab.write(outname.format(savetype='latex'), format='ascii.latex',overwrite=True)
