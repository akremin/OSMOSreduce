# zestiPy

Redshift (z) ESTImation Code developed in Python with minimal dependencies.
Scipy, Astropy, and Numpy are the only 'specialty' libraries required.

The base code (in z_est.py) is heavily based on Dan Gifford's zPy:
https://github.com/giffordw/zpy


This code uses cross-correlation of template spectra (currently from SDSS
but any templates can be used) to determine
the most likely redshift of an input spectra. Currently takes fits files
with a specific structure *SPECIFY STRUCTURE HERE*

