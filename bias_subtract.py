import numpy as np
from astropy.io import fits

def bias_subtract(all_hdus,date):
    biases = {}
    headers = {}
    for (camera, filenum, imtype, opamp),hdu in all_hdus.items():
        if imtype == 'bias':
            if (camera,opamp) not in biases.keys():
                biases[(camera,opamp)] = []
                headers[(camera,opamp)] = hdu.header
            biases[(camera,opamp)].append(hdu.data)

    median_biases = {}
    all_out_hdus = {}
    for (camera,opamp),bias_array_list in biases.items():
        bias_3d_array = np.asarray(bias_array_list)
        median_2d_array = np.median(bias_3d_array,axis=0)
        median_biases[(camera,opamp)] = median_2d_array
        outheader = headers[(camera,opamp)]
        outheader.add_history("Median Master Bias done by quickreduce on {}".format(date))
        outhdu = fits.PrimaryHDU(data=median_2d_array ,header=outheader)
        all_out_hdus[(camera, 'master', 'bias', opamp)] = outhdu

    for (camera, filenum, imtype, opamp), hdu in all_hdus.items():
        if imtype == 'bias':
            continue

        bias_subd = hdu.data - median_biases[camera]
        out_header = hdu.header
        out_header.add_history("Bias Subtracted done by quickreduce on {}".format(date))
        all_out_hdus[(camera, filenum, imtype, opamp)] = fits.PrimaryHDU(data=bias_subd ,header=out_header)

    return all_out_hdus