# Copyright 2012 Bernd Husemann
#
#
#This file is part of PyCosmic.
#
#PyCosmic is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License  as published by
#the Free Software Foundation, either version 3 of the License, or
#any later version.
#
#PyCosmic is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with PyCosmic.  If not, see <http://www.gnu.org/licenses/>.

try:
  import pyfits
except ImportError:
  from astropy.io import fits as pyfits
import numpy
from scipy import ndimage
from scipy import stats

__version__ = "0.4"

class Header(object):
    def __init__(self, header=None, origin=None):
        """
            Creates an Header object

            Parameters
            --------------
            header : pyfits.header object, optional
                    Fits header as header
            origin : string, optional
                    Name of the Fits file as the origin for the header,
                    can be the full path of the file

        """
        if header is not None:
            # Assign private variable and convert header to card list
            self._header = header
        else:
            # Create empty Header and CardList objects
            self._header = None

        # Set the Fits file origin of the header if given
        if origin is not None:
            self._origin = origin
        else:
            self._origin = None

    def setHeader(self, header, origin=None):
        self._header = header
        self._origin=origin
  

    def getHdrValue(self, keyword):
        """
            Returns the value of a certain keyword in the header

            Parameters:
            ---------------
            keyword : string
                        valid keyword in the header

            Returns:
            ---------------
            out : string, integer or float
                        stored value in the header for the given keyword
        """
        return self._header[keyword]

    def getHdrKeys(self):
        """
            Returns all valid keywords of the Header

            Returns:
            ---------------
            out : list
                        list of strings representing the keywords in the header
        """
        return list(self._header.keys())

    def getHeader(self):
        return self._header

  


class Image(Header):
    def __init__(self, data=None, header=None,  mask=None, error=None, origin=None):
        Header.__init__(self, header=header, origin=origin)
        self._data = data
        if self._data is not None:
            self._dim = self._data.shape
        else:
            self._dim = None
        self._mask= mask
        self._error = error
        self._origin = origin

    def __add__(self, other):
        """
        Operator to add two Images or add another type if possible
        """
        if isinstance(other, Image):
            # define behaviour if the other is of the same instance
   
            img = Image(header = self._header, origin=self._origin)

            # add data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data+other._data
                img.setData(data = new_data)
            else:
                img.setData(data = self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(self._error**2+other._error**2)
                img.setData(error=new_error)
            else:
                img.setData(error=self._error)

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                new_mask = numpy.logical_or(self._mask, other._mask)
                img.setData(mask=new_mask)
            else:
                img.setData(mask=self._mask)
            return img
   

        elif isinstance(other,  numpy.ndarray):
            img = Image(error=self._error, mask=self._mask, header = self._header, origin=self._origin)

            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                #add ndarray according do its dimensions
                if self._dim == dim:
                    new_data= self._data+other
                elif len(dim)==1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data+other[:, numpy.newaxis]
                    elif self._dim[1] == dim[0]:
                        new_data = self._data+other[numpy.newaxis, :]
                else:
                    new_data = self._data
                img.setData(data=new_data)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                new_data = self._data+other
                img = Image(data = new_data, error=self._error, mask=self._mask, header = self._header, origin=self._origin)
                return img
            except:
                #raise exception if the type are not matching in general
                raise exceptions.TypeError("unsupported operand type(s) for +: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))

    def __radd__(self, other):
        self.__add__(other)


    def __sub__(self, other):
        """
        Operator to subtract two Images or subtract another type if possible
        """
        if isinstance(other, Image):
            # define behaviour if the other is of the same instance
   
            img = Image(header = self._header, origin=self._origin)

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data-other._data
                img.setData(data = new_data)
            else:
                img.setData(data = self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt(self._error**2+other._error**2)
                img.setData(error=new_error)
            else:
                img.setData(error=self._error)

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                new_mask = numpy.logical_or(self._mask, other._mask)
                img.setData(mask=new_mask)
            else:
                img.setData(mask=self._mask)
            return img
   

        elif isinstance(other,  numpy.ndarray):
            img = Image(error=self._error, mask=self._mask, header = self._header, origin=self._origin)

            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                #add ndarray according do its dimensions
                if self._dim == dim:
                    new_data= self._data-other
                elif len(dim)==1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data-other[:, numpy.newaxis]
                    elif self._dim[1] == dim[0]:
                        new_data = self._data-other[numpy.newaxis, :]
                else:
                    new_data = self._data
                img.setData(data=new_data)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                new_data = self._data-other
                img = Image(data = new_data, error=self._error, mask=self._mask, header = self._header, origin=self._origin)
                return img
            except:
                #raise exception if the type are not matching in general
                raise exceptions.TypeError("unsupported operand type(s) for -: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))

    def __truediv__(self, other):
        """
        Operator to divide two Images or divide by another type if possible
        """
        if isinstance(other, Image):
            # define behaviour if the other is of the same instance
   
            img = Image(header = self._header, origin=self._origin)

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data/other._data
                img.setData(data = new_data)
            else:
                img.setData(data = self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt((self._error/other._data)**2+(self._data*other._error/other._data**2)**2)
                img.setData(error=new_error)
            else:
                img.setData(error=self._error)

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                new_mask = numpy.logical_or(self._mask, other._mask)
                img.setData(mask=new_mask)
            else:
                img.setData(mask=self._mask)
            return img
   

        elif isinstance(other,  numpy.ndarray):
            img = Image(error=self._error, mask=self._mask, header = self._header, origin=self._origin)

            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                #add ndarray according do its dimensions
                if self._dim == dim:
                    new_data= self._data/other
                    if self._error is not None:
                        new_error = self._error/other
                    else:
                        new_error = None
                elif len(dim)==1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data/other[:, numpy.newaxis]
                        if self._error is not None:
                            new_error = self._error/other[:, numpy.newaxis]
                        else:
                            new_error is not None
                    elif self._dim[1] == dim[0]:
                        new_data = self._data/other[numpy.newaxis, :]
                        if self._error is not None:
                            new_error = self._error/other[numpy.newaxis, :]
                        else:
                            new_error is not None
                else:
                    new_data = self._data
                img.setData(data=new_data,  error = new_error)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                new_data = self._data/other
                if self._error is not None:
                    new_error = self._error/other
                else:
                    new_error = None
                img = Image(data = new_data, error=new_error, mask=self._mask, header = self._header, origin=self._origin)
                return img
            except:
                #raise exception if the type are not matching in general
                raise exceptions.TypeError("unsupported operand type(s) for /: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))

    def __mul__(self, other):
        """
        Operator to divide two Images or divide by another type if possible
        """
        if isinstance(other, Image):
            # define behaviour if the other is of the same instance
   
            img = Image(header = self._header, origin=self._origin)

            # subtract data if contained in both
            if self._data is not None and other._data is not None:
                new_data = self._data*other._data
                img.setData(data = new_data)
            else:
                img.setData(data = self._data)

            # add error if contained in both
            if self._error is not None and other._error is not None:
                new_error = numpy.sqrt((self._error*other._data)**2+(self._data*other._error)**2)
                img.setData(error=new_error)
            else:
                img.setData(error=self._error)

            # combined mask of valid pixels if contained in both
            if self._mask is not None and other._mask is not None:
                new_mask = numpy.logical_or(self._mask, other._mask)
                img.setData(mask=new_mask)
            else:
                img.setData(mask=self._mask)
            return img
   

        elif isinstance(other,  numpy.ndarray):
            img = Image(error=self._error, mask=self._mask, header = self._header, origin=self._origin)

            if self._data is not None:  # check if there is data in the object
                dim = other.shape
                #add ndarray according do its dimensions
                if self._dim == dim:
                    new_data= self._data*other
                elif len(dim)==1:
                    if self._dim[0] == dim[0]:
                        new_data = self._data*other[:, numpy.newaxis]
                    elif self._dim[1] == dim[0]:
                        new_data = self._data*other[numpy.newaxis, :]
                else:
                    new_data = self._data
                img.setData(data=new_data)
            return img
        else:
            # try to do addtion for other types, e.g. float, int, etc.
            try:
                new_data = self._data*other
                img = Image(data = new_data, error=self._error, mask=self._mask, header = self._header, origin=self._origin)
                return img
            except:
                #raise exception if the type are not matching in general
                raise exceptions.TypeError("unsupported operand type(s) for *: %s and %s"%(str(type(self)).split("'")[1], str(type(other)).split("'")[1]))

    def __rmul__(self, other):
        self.__mul__(other)

    def __lt__(self, other):
        return self._data<other

    def __le__(self, other):
        return self._data<=other

    def __eq__(self, other):
        return self._data==other

    def __ne__(self, other):
        return self._data!=other

    def __gt__(self, other):
        return self._data>other

    def __ge__(self, other):
        return self._data>=other

    def sqrt(self):
        """
            Computes the square root  of the image

            Returns
            -----------
            Image : data_model.Image object
                A full Image object

        """
        if self._data is not None:
            new_data = numpy.sqrt(self._data) #sqrt of the data
        else:
            new_data = None

        if self._error is not None and self._data is not None:
            new_error = 1/(2*numpy.sqrt(self._data))*self._error #corresponding error
        else:
            new_error = None

        return Image(data=new_data, error =new_error,  mask = self._mask, header=self._header,  origin=self._origin) # return new Image object with corresponding data

    def getDim(self):
        """
            Returns the dimension of the image

            Returns
            -----------
            _dim :  tuple
                The dimension of the image (y,x)

        """
        return self._dim

    def getData(self):
        """
            Returns the stored data of the image

            Returns
            -----------
            _data :  numpy.ndarray
                The stored data of the image

        """
        return self._data

    def getMask(self):
        """
            Returns the bad pixel mask of the image

            Returns
            -----------
            _mask :  numpy.ndarray
                The bad pixel mask of the image

        """
        return self._mask

    def getError(self):
        """
            Returns the associated error of the image

            Returns
            -----------
            _error :  numpy.ndarray
                The associated error of the image

        """
        return self._error


    def setData(self, data=None, error=None, mask=None, header=None, select=None):
        """
            Set data for an Image. Specific data values can replaced according to a specific selection.

            Parameters
            --------------
            data : numpy.ndarray(float), optional with default = None
                array corresponding to the data to be set
            error : numpy.ndarray(float), optional with default = None
                array corresponding to the data to be set
            mask : numpy.ndarray(bool), optional with default = None
                array corresponding to the bad pixel to be set
            header : Header object, optional with default = None
            select : numpy.ndarray(bool), optional with default = None
                array defining the selection of pixel to be set

        """
        # if not select given set the full image
        if select is None:
            if data is not None:
                self._data = data # set data if given
                self._dim = data.shape # set dimension

            if mask is not None:
                self._mask = mask # set mask if given
                self._dim = mask.shape # set dimension

            if error is not None:
                self._error = error  # set mask if given
                self._dim = error.shape # set dimension
            if header is not None:
                self.setHeader(header) # set header
        else:
            # with select definied only partial data are set
            if data is not None:
                self._data[select] = data
            if mask is not None:
                self._mask[select] = mask
            if error is not None:
                self._error[select] = error
            if header is not None:
                self.setHeader(header) # set header


    def removeError(self):
        self._error=None

    def loadFitsData(self, filename,  extension_data=None, extension_mask=None, extension_error=None):
        """
            Load data from a FITS image into an Image object, If no specific extensions are given, the  primary extension is
            assumed to contain the data. All previous extension will be associated according to the EXTNAME keyword either
            as an error image or a bad pixel mask.

            Parameters
            --------------
            filename : string
                Name or Path of the FITS image from which the data shall be loaded
            extension_data : int, optional with default: None
                Number of the FITS extension containing the data
            extension_mask : int, optional with default: None
                Number of the FITS extension containing the masked pixels
            extension_error : int, optional with default: None
                Number of the FITS extension containing the errors for the values
        """
        hdu = pyfits.open(filename, ignore_missing_end=True) #open FITS file
        if extension_data is None and extension_mask is None and extension_error is None:
                self._data = hdu[0].data
                self._dim = self._data.shape # set dimension
                if len(hdu)>1:
                    for i in range(1, len(hdu)):
                        if hdu[i].header['EXTNAME'].split()[0]=='ERROR':
                            self._error = hdu[i].data
                        elif hdu[i].header['EXTNAME'].split()[0]=='BADPIX':
                            self._mask = hdu[i].data

        else:
            if extension_data is not None:
                self._data = hdu[extension_data].data # take data
                self._dim = self._data.shape # set dimension

            if extension_mask is not None:
                self._mask = hdu[extension_mask].data # take data
                self._dim = self._mask.shape # set dimension

            if extension_error is not None:
                self._error = hdu[extension_error].data  # take data
                self._dim = self._error.shape # set dimension


        self.setHeader(hdu[0].header) # get header  from the first FITS extension
        hdu.close()



    def writeFitsData(self, filename,  extension_data=None, extension_mask=None, extension_error=None):
        """
            Save information from an Image into a FITS file. A single or multiple extension file can be created.
            If all optional paramters are set to None, all data if contained will be stored in to extension of the FITS file.

            Parameters
            --------------
            filename : string
                Name or Path of the FITS image from which the data shall be loaded
            extension_data : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the data
            extension_mask : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the masked pixels
            extension_error : int (0, 1, or 2), optional with default: None
                Number of the FITS extension containing the errors for the values
        """
        hdus=[None, None, None] # create empty list for hdu storage

        # create primary hdus and image hdus
        # data hdu
        if extension_data is None and extension_error is None and extension_mask is None:
            hdus[0] = pyfits.PrimaryHDU(self._data)
            if self._error is not None:
                hdus[1] = pyfits.ImageHDU(self._error, name='ERROR')
            if self._mask is not None:
                hdus[2] = pyfits.ImageHDU(self._mask.astype('uint8'), name='BADPIX')
        else:
            if extension_data == 0:
                hdus[0] = pyfits.PrimaryHDU(self._data)
            elif extension_data is not None and extension_data>0:
                hdus[extension_data] = pyfits.ImageHDU(self._data, name='DATA')

            # mask hdu
            if extension_mask == 0:
                hdu = pyfits.PrimaryHDU(self._mask.astype('uint8'))
            elif extension_mask is not None and extension_mask>0:
                hdus[extension_mask] = pyfits.ImageHDU(self._mask.astype('uint8'), name='BADPIX')

            # error hdu
            if extension_error == 0:
                hdu = pyfits.PrimaryHDU(self._error)
            elif extension_error is not None and extension_error>0:
                hdus[extension_error] = pyfits.ImageHDU(self._error, name='ERROR')

        # remove not used hdus
        for i in range(len(hdus)):
            try:
                hdus.remove(None)
            except:
                break

        if len(hdus)>0:
            hdu = pyfits.HDUList(hdus) # create an HDUList object
            if self._header is not None:
                hdu[0].header = self.getHeader() # add the primary header to the HDU
                hdu[0].update_header()
        hdu.writeto(filename, clobber=True) # write FITS file to disc

    def replaceMaskMedian(self, box_x, box_y, replace_error=1e20):
        """
            Replace bad pixels with the median value of pixel in a rectangular filter window

            Parameters
            --------------
            box_x : int
                Pixel size of filter window in x direction
            box_y : int
                Pixel size of filter window in y direction
            replace_error : float, optional with default: None
                Error that should be set for bad pixel

            Returns
            -----------
            new_image :  Image object
                Subsampled image
        """
        idx = numpy.indices(self._dim) # create an index array
        # get x and y coordinates of bad pixels

        y_cors = idx[0][self._mask]
        x_cors = idx[1][self._mask]

        out_data = self._data
        out_error = self._error

        # esimate the pixel distance form the bad pixel to the filter window boundary
        delta_x = numpy.ceil(box_x/2.0)
        delta_y = numpy.ceil(box_y/2.0)

        # iterate over bad pixels
        for m in range(len(y_cors)):
            # computes the min and max pixels of the filter window in x and y
            range_y = numpy.clip([y_cors[m]-delta_y, y_cors[m]+delta_y+1], 0, self._dim[0]-1).astype(int)
            range_x = numpy.clip([x_cors[m]-delta_x, x_cors[m]+delta_x+1], 0, self._dim[1]-1).astype(int)
            # compute the masked median within the filter window and replace data
            select = self._mask[range_y[0]:range_y[1],range_x[0]:range_x[1]]==0
            out_data[y_cors[m],x_cors[m]] = numpy.median(self._data[range_y[0]:range_y[1],range_x[0]:range_x[1]][select])
            if self._error is not None and replace_error is not None:
                # replace the error of bad pixel if defined
                out_error[y_cors[m],x_cors[m]] = replace_error

        # create new Image object
        new_image = Image(data=out_data, error = out_error,  mask = self._mask,  header = self._header)
        return new_image



    def subsampleImg(self):
        """
            Subsample the image by a factor of 2, e.g. each pixel is divided into 4 pixel so that their sum is 4 times the original one.

            Returns
            -----------
            new_image :  Image object
                Subsampled image

        """
        # create empty array with 2 time larger size in both axes
        new_dim = (self._dim[0]*2,self._dim[1]*2)
        if self._data is not None:
            new_data = numpy.zeros(new_dim,dtype=numpy.float32)
        else:
            new_data = None
        if self._error is not None:
            new_error = numpy.zeros(new_dim,dtype=numpy.float32)
        else:
            new_error = None
        if self._mask is not None:
            new_mask= numpy.zeros(new_dim,dtype='bool')
        else:
            new_mask = None


        # set pixel for the subsampled data, error and mask
        if self._data is not None:
            new_data[::2,::2] = self._data
            new_data[::2,1::2] = self._data
            new_data[1::2,::2] = self._data
            new_data[1::2,1::2] = self._data
        if self._error is not None:
            new_error[::2,::2] = self._error
            new_error[::2,1::2] = self._error
            new_error[1::2,::2] = self._error
            new_error[1::2,1::2] = self._error
        if self._mask is not None:
            new_mask[::2,::2] = self._mask
            new_mask[::2,1::2] = self._mask
            new_mask[1::2,::2] = self._mask
            new_mask[1::2,1::2] = self._mask

        # create new Image object with the new subsample data
        new_image = Image(data=new_data, error=new_error,  mask=new_mask)
        return new_image


    def rebin(self, bin_x, bin_y):
        """
            Rebin the image by regullarly summing up the pixel in a regual rectangular binning window with size bin_x times bin_y.
            Make sure that the size of the binning window matches with the total number of pixel in the original image.

            Parameters
            --------------
            bin_x : int
                Pixel size of the binning window in x direction
            bin_y : int
                Pixel size of the binning window in y direction

            Returns
            -----------
            new_image :  Image object
                Subsampled image
        """
        # sum over the data array over each axis by the given pixel
        ##hack !!! changed division to integer division
        new = numpy.sum(numpy.reshape(self._data,(self._dim[0],self._dim[1]//bin_x,bin_x)),2)
        new2 = numpy.sum(numpy.reshape(new,(self._dim[0]//bin_y,bin_y, self._dim[1]//bin_x)),1)

        if self._error is not None:
            # sum over the error array (converted to variance and back) over each axis by the given pixel
            ##hack !!! changed division to integer division
            error_new = numpy.sum(numpy.reshape(self._error**2,(self._dim[0],self._dim[1]//bin_x,bin_x)),2)
            error_new2 = numpy.sqrt(numpy.sum(numpy.reshape(error_new,(self._dim[0]//bin_y,bin_y, self._dim[1]//bin_x)),1))
        else:
            error_new2=None

        if self._mask is not None:
            # create the new  bad pixel mask
            ##hack !!! changed division to integer division
            mask_new = numpy.sum(numpy.reshape(self._mask,(self._dim[0],self._dim[1]//bin_x,bin_x)),2)
            mask_new2 = numpy.sum(numpy.reshape(mask_new,(self._dim[0]//bin_y,bin_y, self._dim[1]//bin_x)),1)
            # if only one bad pixel in the binning pixel exists the binned pixel will have the bad pixel status
            new_mask = mask_new2>0
        else:
            new_mask=None
        # create new Image object and return
        new_img = Image(data=new2, error=error_new2, mask=new_mask, header=self._header, origin=self._origin)
        return new_img

    def convolveImg(self, kernel, mode='nearest'):
        """
            Convolves the data of the Image with a given kernel. The mask and error information will be unchanged.

            Parameters
            --------------
            kernel : ndarray
                Convolution kernel
            mode :  string, optional with default: 'nearest'
                Set the mode how to handle the boundarys within the convolution
  

            Returns
            -----------
            new_image :  Image object
                Convolved image
        """

        # convolve the data array with the given convolution kernel
        new = ndimage.filters.convolve(self._data,kernel,mode=mode)
        if self._error is not None:
            new_error = numpy.sqrt(ndimage.filters.convolve(self._error**2, kernel, mode=mode))
        else:
            new_error = None
        # create new Image object with the error and the mask unchanged and return
        new_image = Image(data=new, error=new_error,  mask=self._mask, header = self._header, origin=self._origin)
        return new_image

    def convolveGaussImg(self, sigma_x, sigma_y, mode='nearest', mask=False):
        """
            Convolves the data of the Image with a given kernel. The mask and error information will be unchanged.

            Parameters
            --------------
            sigma_x : float
                With of the Gaussian in pixels along the x direction
            sigma_y : float
                With of the Gaussian in pixels along the y direction
            mode :  string, optional with default: 'nearest'
                Set the mode how to handle the boundarys within the convolution


            Returns
            -----------
            new_image :  Image object
                Convolved Image
        """
        # convolve the data array with the 2D Gaussian convolution kernel

        if self._mask is not None and mask==True:
            mask_data=self._data[self._mask]
            self._data[self._mask]=0
            gauss =ndimage.filters.gaussian_filter(self._data,(sigma_y,sigma_x),mode=mode)
            scale =ndimage.filters.gaussian_filter((self._mask==False).astype('float32'),(sigma_y,sigma_x),mode=mode)
            new=gauss/scale
            self._data[self._mask]=mask_data
        else:
            new=ndimage.filters.gaussian_filter(self._data,(sigma_y,sigma_x),mode=mode)
        # create new Image object with the error and the mask unchanged and return
        new_image = Image(data=new, error=self._error,  mask=self._mask, header = self._header, origin=self._origin)
        return new_image

    def medianImg(self, size, mode='nearest'):
        """
            Return a new Image that has been median filtered with a filter window of given size.

            Parameters
            --------------
            size : tuple of int
                Size of the filter window
            mode : string, optional with default: nearest
                Set the mode how to handle the boundarys within the convolution
                Possilbe modes are: reflect, constant, nearest, mirror,  wrap

            Returns
            -----------
            image :  Image object
                An Image object with the median filter data
        """

        new_data = ndimage.filters.median_filter(self._data,size ,mode=mode) # applying the median filter
        image = Image(data=new_data, header = self._header, error = self._error,  mask = self._mask) # create a new Image object
        return image

    def meanImg(self,size,sigma=5.0,mode='nearest'):
        if sigma == 0:
            new_data = ndimage.generic_filter(self._data,numpy.mean,size,mode=mode)
        else:
            new_data = ndimage.generic_filter(self._data,stats.sigmaclip,size,mode=mode,extra_keywords={'high':sigma})
        image = Image(data=new_data, header=self._header, error=self._error, mask=self._mask)  # create a new Image object
        return image

def loadImage(infile,  extension_data=None, extension_mask=None, extension_error=None):

    image = Image()
    image.loadFitsData(infile,  extension_data=extension_data,  extension_mask=extension_mask, extension_error=extension_error)

    return image
