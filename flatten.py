



if do_step['flat']:
    for imtype in ['twiflat' ,'fibmap']:
        for camera in cr_removed_data[imtype].keys():
            filenum_3d_array = np.asarray(list(data[imtype][camera].values()))
            header =list(headers[imtype][camera].values())[0]

            for exposure in range(filenum_3d_array.shape[0]):
                ## get exposure and make sure there are no negative values
                current_exposure = filenum_3d_array[exposure, :, :]
                current_exposure -= np.min(current_exposure)
                ## get a median smoothed version to remove any peculiarities, find it's max
                median_exposure = median_filter(current_exposure, size=5)
                ## divide by the max of the median exposure to normalize (excluding outliers)
                current_exposure /= np.max(median_exposure)
                filenum_3d_array[exposure, :, :] = current_exposure

            filenum_summed_array = np.median(filenum_3d_array, axis=0)
            header.add_history("Summed Master {} done by quickreduce on {}".format(imtype, date))

            outhdu = fits.PrimaryHDU(data=filenum_summed_array, header=header)
            outname = master_info['master_templates'].format(cam=camera, imtype=imtype, maskname=mask_name,
                                                             tags=save_info['tags'])
            filename = os.path.join(save_info['datadir'], outname)
            outhdu.writeto(filename, overwrite=True)

            ## Plot the image
            plt.figure()
            plot_array = outhdu.data - np.min(outhdu.data) + 1e-4
            plt.imshow(np.log(plot_array), 'gray', origin='lowerleft')
            plt.savefig(filename.replace('.fits', '.png'), dpi=1200)
            plt.show()

            headers[imtype][camera] = header
            data[imtype][camera] = filenum_summed_array

    master_twiflat_data = data['twiflat']
    flat_data, flat_headers = {}, {}
    for imtype in filenumbers.keys():
        flat_data[imtype] = {}
        flat_headers[imtype] = {}
        for camera, master_twiflat in master_twiflat_data.items():
            master_twiflat /= np.max(master_twiflat)
            flat_data[imtype][camera] = {}
            flat_headers[imtype][camera] = {}
            datadict = data[imtype][camera]

            headerdict = headers[imtype][camera]
            for filnum, filearray in datadict.items():
                filearray = filearray.astype(float)
                header = headerdict[filnum]
                filearray /= master_twiflat.astype(float)
                header.add_history("Flat correction done by quickreduce on {}".format(date))
                outhdu = fits.PrimaryHDU(data=filearray, header=header)
                filename = save_info['template'].format(cam=camera, imtype=imtype,
                                                        maskname=mask_name,
                                                        filenum=filnum, \
                                                        tags=save_info['tags'])
                filename = os.path.join(save_info['datadir'], filename)
                outhdu.writeto(filename, overwrite=True)

                ## Plot the image
                plt.figure()
                plot_array = outhdu.data - np.min(outhdu.data) + 1e-4
                plt.imshow(np.log(plot_array), 'gray', origin='lowerleft')
                plt.savefig(filename.replace('.fits', '.png'), dpi=1200)
                plt.show()

                flatnd_data[imtype][camera][filnum] = header
                flatnd_headers[imtype][camera][filnum] = filearray

    print("Completed flattening for {}".format(imtype))
    print("Results saved to {}".format(save_info['datadir']))
    del cr_removed_data, cr_removed_headers
