import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

def bias_subtract(all_hdus,date,strategy='median',convert_adu_to_e=True):
    biases = {}
    headers = {}
    for (camera, filenum, imtype, opamp),hdu in all_hdus.items():
        if imtype == 'bias':
            converted_hdu = remove_bias_lines(cur_hdu=hdu,use_bias_cols=True,convert_adu_to_e=convert_adu_to_e)
            if (camera,opamp) not in biases.keys():
                biases[(camera,opamp)] = []
                headers[(camera,opamp)] = converted_hdu.header
            biases[(camera,opamp)].append(converted_hdu.data)

    merged_biases = {}
    all_out_hdus = {}

    if strategy == 'median':
        for (camera,opamp),bias_array_list in biases.items():
            bias_3d_array = np.asarray(bias_array_list)
            median_2d_array = np.median(bias_3d_array,axis=0)
            merged_biases[(camera,opamp)] = median_2d_array
            outheader = headers[(camera,opamp)]
            outheader.add_history("Median Master Bias done by quickreduce on {}".format(date))
            outhdu = fits.PrimaryHDU(data=median_2d_array ,header=outheader)
            all_out_hdus[(camera, 'master', 'bias', opamp)] = outhdu
            dat = outhdu.data
            dat = dat - dat.min() + 1
            logdat = np.log(dat)
            plt.figure()
            plt.imshow(logdat, 'gray', origin='lowerleft')
            plt.title("Master Bias")
            plt.tight_layout()
            plt.show()
    else:
        raise(TypeError,"The only bias strategy currently supported in median")

    for (camera, filenum, imtype, opamp), hdu in all_hdus.items():
        if imtype == 'bias':
            continue
        converted_hdu = remove_bias_lines(cur_hdu=hdu,use_bias_cols=True,convert_adu_to_e=True)

        bias_subd = converted_hdu.data - merged_biases[(camera,opamp)]
        out_header = converted_hdu.header
        out_header.add_history("Bias Subtracted done by quickreduce on {}".format(date))
        all_out_hdus[(camera, filenum, imtype, opamp)] = fits.PrimaryHDU(data=bias_subd ,header=out_header)

    return all_out_hdus


def remove_bias_lines(cur_hdu,use_bias_cols=False,convert_adu_to_e=False):
    cur_dat_header = cur_hdu.header
    cur_dat_data = cur_hdu.data

    datasec = cur_dat_header['DATASEC'].strip('[]')
    (x1, x2), (y1, y2) = [[int(x) - 1 for x in va.split(':')] for va in datasec.split(',')]

    if use_bias_cols:
        nbiases = cur_dat_header['NBIASLNS']
        bias_lines = cur_dat_data[-nbiases:,:]
        median_bias_pix = np.median(bias_lines)
        cut_dat_data = cur_dat_data[y1:y2 + 1, :] - median_bias_pix

        out_array = np.ndarray(shape=(y2+1-y1,x2+1-x1))
        for row in range(cut_dat_data.shape[0]):
            current_line = cut_dat_data[row,:]
            overscans = np.median(current_line[x2+1:])
            out_array[row,:] = current_line[x1:x2+1] - overscans
    else:
        out_array = cur_dat_data[y1:y2 + 1, x1:x2 + 1]

    cur_dat_header = scrub_header(cur_dat_header, out_array.shape)
    if convert_adu_to_e:
        out_array = (out_array * cur_dat_header['EGAIN'])  # - cur_dat_header['ENOISE']
        cur_dat_header['EGAIN'] = 1.0
        # cur_dat_header['ENOISE'] = 0.0

    outhdu = fits.PrimaryHDU(data=out_array, header=cur_dat_header)
    return outhdu

def scrub_header(header,array_shape):
    header.remove('NOVERSCN')
    header.remove('NBIASLNS')
    header.remove('BIASSEC')
    header['NAXIS1'] = int(array_shape[0])
    header['NAXIS2'] = int(array_shape[1])
    return header