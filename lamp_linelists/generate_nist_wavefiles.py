

from astropy.table import vstack,Table
import os
import requests
import numpy as np


nist_loc = os.path.join(os.path.abspath('./'),'lamp_linelists','Nist')
nist_template = '{element}_air_{wavelow}-{wavehigh}.csv'

webaddr_template = 'https://physics.nist.gov/cgi-bin/ASD/lines1.pl?spectra={element}&limits_type=0&low_w={wavelow}&upp_w={wavehigh}&unit=0&de=0&format=2&line_out=3&remove_js=on&en_unit=1&output=0&page_size=15&show_obs_wl=1&unc_out=1&order_out=0&max_low_enrg=&show_av=2&max_upp_enrg=&tsb_value=0&min_str=&A_out=0&intens_out=on&max_str=&allowed_out=1&forbid_out=1&min_accur=&min_intens=&submit=Retrieve+Data'

elements = ['He','Ne','Hg','Ar','Th','Xe']
for element in elements:
    webaddr = webaddr_template.format(element=element,wavelow=2000,wavehigh=14000)
    nist_name = nist_template.format(element=element,wavelow=2000,wavehigh=14000)
    nist_path = os.path.join(nist_loc,nist_name)
    if not os.path.exists(nist_path):
        response = requests.get(webaddr)
        with open(nist_path,'w') as outfil:
            outfil.write(response.text.replace('"="','').replace('"',''))

##### At this point, you need to do some regular expression work to get rid of a bunch of improper formattings


tabs = []
for element in elements:
    print(element)
    nist_name = nist_template.format(element=element,wavelow=2000,wavehigh=14000)
    nist_path = os.path.join(nist_loc,nist_name)
    tab = Table.read(nist_path,format='ascii.csv')
    tab.remove_columns(['Aki(s^-1)','Type'])
    numcol, symbcol = [], []
    for row in tab['intens']:
        num, symb = '', ''
        for elem in str(row):
            if elem.isnumeric():
                num += elem
            elif elem == '.':
                num += elem
            else:
                symb += elem
        if num == '':
            num = '-99'
        numcol.append(float(num))
        symbcol.append(symb)
    tab.add_column(Table.Column(name='intensity', data=numcol))
    tab.remove_column('intens')
    tab.add_column(Table.Column(name='intensity flags', data=symbcol))
    if tab['Acc'].dtype != str:
        if type(tab['Acc']) == Table.MaskedColumn:
            tab.replace_column('Acc', Table.Column(name='Acc', data=tab['Acc'].astype(str).filled('')))
        else:
            tab.replace_column('Acc', Table.Column(name='Acc', data=tab['Acc'].astype(str)))
    tab.rename_column('Acc','Grade')
    nist_name = nist_template.format(element=element+'_cleaned',wavelow=2000,wavehigh=14000)
    nist_path = os.path.join(nist_loc,nist_name)
    tab.write(nist_path,format='ascii.csv',overwrite=True)
    tabs.append(tab)
# stk = vstack(tabs, 'outer')
#
# np.where((stk['unc_obs_wl']/stk['obs_wl_air(A)'])*(3e5)>10)[0]