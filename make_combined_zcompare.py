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

def main(mtl_path,mtl_name,correlation_cut=0.2):
    maskname = mtl_name.split('_')[2]

    tab = Table.read(os.path.join(mtl_path,mtl_name),format='ascii.csv')

    ## Make appropriate cuts
    if 'SDSS_only' in tab.colnames:
        tab = tab[np.bitwise_not(tab['SDSS_only'])]
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
    if 'RA' not in tab.colnames:
        tab.rename_column('RA_targeted','RA')
        tab.rename_column('DEC_targeted','DEC')
    if 'sdss_SDSS12' in tab.colnames:
        tab.rename_column('OBJID')
        for ii in len(tab):
            tab['OBJID'][ii] = 'SDSS'+str(tab['OBJID'][ii])
    else:
        tab.add_column(tab.MaskedColumn(data=['']*len(tab),name='OBJID',mask=np.ones(len(tab).astype(bool))))
    if 'sdss_zsp' in tab.colnames:
        tab.rename_column('sdss_zsp','SDSS_zsp')
    else:
        tab.add_column(Table.MaskedColumn(data=np.zeros(len(tab)),name='SDSS_zsp',mask=np.ones(len(tab)).astype(bool)))
    if 'z_est_bary' in tab.colnames:
        tab.rename_column('z_est_bary','z')
    if 'Proj_R_asec' in tab.colnames:
        tab.rename_column('Proj_R_asec','R [asec]')
    if 'velocity' in tab.colnames:
        tab.rename_column('velocity','v [km/s]')
    if 'FIBNAME' in tab.colnames:
        tab.rename_column('FIBNAME', 'FIBERNUM')
    ## Load up in the right order
    tab = tab[['TARGETID','FIBERNUM','RA','DEC','OBJID','z','R [asec]','v [km/s]','SDSS_zsp']]

    return tab.filled('--')

if __name__ == '__main__':
    catalog_loc = '../data/catalogs/merged_target_lists/'
    first = True
    for fil in os.listdir(catalog_loc):
        if 'full.csv' in fil and 'mtlz_' in fil:
            if first:
                outtab = main(catalog_loc,fil,correlation_cut=corcut)
                first = False
            else:
                itter_tab = main(catalog_loc,fil,correlation_cut=corcut)
                outtab = vstack([outtab,itter_tab])
    outname = os.path.join(catalog_loc,'full_dataset_table.{savetype}')
    outtab.write(outname.format(savetype='csv'),format='ascii.csv',overwrite=True)
    outtab.write(outname.format(savetype='fits'), format='fits',overwrite=True)
    outtab.write(outname.format(savetype='aas.tex'), format='ascii.aastex',overwrite=True)
    outtab.write(outname.format(savetype='latex'), format='ascii.latex',overwrite=True)
