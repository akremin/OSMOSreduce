import sqlcl
import pandas as pd
import numpy as np
from astropy.coordinates import SkyCoord
import pdb

#ra,dec = np.loadtxt('../estimated_redshifts2.tab',dtype='string',usecols=(0,1),unpack=True)
def query_galaxies(ra,dec):
    gal_sdss_data = []
    gal_sdss_PID = []
    gal_sdss_SID = []
    for i in range(ra.size):
        ra1 = ra[i].split(':')
        dec1 = dec[i].split(':')
        coord = SkyCoord(ra1[0]+'h'+ra1[1]+'m'+ra1[2]+'s '+dec1[0]+'d'+dec1[1]+'m'+dec1[2]+'s',frame='icrs')
        print coord.ra.deg
        print coord.dec.deg
        #query = sqlcl.query("SELECT  gn.objid, ISNULL(s.specobjid,0) AS specobjid, p.ra, p.dec,p.Petromag_u-p.extinction_u AS U_mag,p.Petromag_g-p.extinction_g AS G_mag,p.Petromag_r-p.extinction_r AS R_mag,p.Petromag_i-p.extinction_i AS I_mag,p.Petromag_z-p.extinction_z AS Z_mag, ISNULL(s.z, 0) AS z, ISNULL(pz.z, 0) AS pz FROM  (Galaxy AS p JOIN dbo.fGetNearbyObjEq("+str(coord.ra.deg)+","+str(coord.dec.deg)+","+str(0.05)+") AS GN  ON p.objID = GN.objID LEFT OUTER JOIN SpecObj s ON s.bestObjID = p.objID) LEFT OUTER JOIN Photoz pz on pz.objid = p.objid WHERE p.Petromag_r-p.extinction_r < 19.1 and p.clean = 1").readlines()
        query = sqlcl.query("SELECT  gn.objid, ISNULL(s.specobjid,0) AS specobjid, p.ra, p.dec,p.Petromag_u-p.extinction_u AS U_mag,p.Petromag_g-p.extinction_g AS G_mag,p.Petromag_r-p.extinction_r AS R_mag,p.Petromag_i-p.extinction_i AS I_mag,p.Petromag_z-p.extinction_z AS Z_mag, ISNULL(s.z, 0) AS z, ISNULL(s.zErr, 0) AS z_err, ISNULL(pz.z, 0) AS pz FROM  (Galaxy AS p JOIN dbo.fGetNearbyObjEq("+str(coord.ra.deg)+","+str(coord.dec.deg)+","+str(0.05)+") AS GN  ON p.objID = GN.objID LEFT OUTER JOIN SpecObj s ON s.bestObjID = p.objID) LEFT OUTER JOIN Photoz pz on pz.objid = p.objid WHERE p.Petromag_r-p.extinction_r < 19.1").readlines()
        if len(query) > 4:
            print 'oops! More than 1 candidate found'
        if len(query) == 2:
            print 'No targets found'
            gal_sdss_data.append([0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
            gal_sdss_PID.append(0)
            gal_sdss_SID.append(0)
            continue

        gal_sdss_data.append(map(float,query[2].split(',')))
        gal_sdss_PID.append(query[2].split(',')[0])
        gal_sdss_SID.append(query[2].split(',')[1])
        print 'Done with galaxy',i

    gal_sdss_data = np.array(gal_sdss_data)
    S_df = pd.DataFrame(gal_sdss_data,columns=['#objID','SpecObjID','ra','dec','umag','gmag','rmag','imag','zmag','spec_z','spec_z_err','photo_z'])
    S_df['#objID'] = gal_sdss_PID
    S_df['SpecObjID'] = gal_sdss_SID
    return S_df


