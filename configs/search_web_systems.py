
import os
#import beautifulsoup as bs4
from astropy.table import Table
import webbrowser


webbrowser.open('', new=0)

ned_byname_base = 'https://ned.ipac.caltech.edu/byname?objname={PERCNAME}&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1'
simbad_byname_base = 'http://simbad.u-strasbg.fr/simbad/sim-basic?Ident={PLUSNAME}&submit=SIMBAD+search'

ned_bycoord_base = 'https://ned.ipac.caltech.edu/conesearch?search_type=Near%20Position%20Search&coordinates={RA}%20%2B{DEC}&radius={RADIUS}&in_csys=Equatorial&in_equinox=J2000.0&out_csys=Equatorial&out_equinox=J2000.0&hconst=67.8&omegam=0.308&omegav=0.692&wmap=4&corr_z=1'
simbad_bycoord_base = 'http://simbad.u-strasbg.fr/simbad/sim-coo?Coord={RA}+%2B{DEC}&CooFrame=FK5&CooEpoch=2000&CooEqui=2000&CooDefinedFrames=none&Radius={RADIUS}&Radius.unit=arcmin&submit=submit+query&CoordList='
vizier_bycoord_base = 'http://vizier.u-strasbg.fr/viz-bin/VizieR-3?-ref=VIZ5ccc9c5a59a9&-to=-2b&-from=-2&-this=-2&%2F%2Fc={RA}+%2B{DEC}&-out.max=50&%2F%2FCDSportal=http%3A%2F%2Fcdsportal.u-strasbg.fr%2FStoreVizierData.html&-out.form=HTML+Table&-out.add=_r&-out.add=_RAJ%2C_DEJ&%2F%2Foutaddvalue=default&-sort=_r&-order=I&-oc.form=dega&-nav=key%3Ac%3D{RA}+%2B{DEC}%26pos%3A{RA}+%2B{DEC}%28+++{RADIUS}+arcmin+J2000%29%26HTTPPRM%3A&%2F%2Fnone=on&-c={RA}+%2B{DEC}&-c.eq=J2000&-c.r=++{RADIUS}&-c.u=arcmin&-c.geom=r&-file=.&-meta.ucd=2&-meta=1&-meta.foot=1&-usenav=1&-bmark=GET'


webbrowser.open('http://google.co.kr', new=2)



