M2FSreduce
=========

Note:
--- 
* Reduction software for Magellan M2FS spectra

* Initially forked from Dan Gifford's OSMOSreduce package used to reduce 2.4m MDM telescope slit-mask spectra.
   * Code has been significantly altered. A few subfunctions share inspiration and lines of code from the original.

* Code produced by Anthony Kremin for M2FS on the 6.5 Magellan-Clay telescope. 

Eventually documentation for using the pipeline will be available as an IPython notebook in the docs/ subdirectory.

The code is run from a "quickreduce file." The standard quickreduce can be run from the command
line with:

    "-m", "--maskname":  defines the name of the mask
    "-i", "--iofile":  defines the input-ouput configuration file location
    "-o", "--obsfile":   defines the observation configuration file location
    "-p", "--pipefile":  defines the pipeline configuration file location
    
The maskname can be used alone if standards for the three configuration file is used.
That standard is as follows:

'./configs/obs_{maskname}.ini'
'./configs/io_{maskname}.ini'
'./configs/pipeline.ini'


The basic steps performed by the code are as follows:

 1. Define everything, load the bias files, generate any directories that don't exist.
 2. Create master bias file, Save master bias file  
 3. Open all other files, subtract the master bias, save  (*c?.b.fits)
 3. Stitch the four opamps together.
 4. Remove cosmics from all file types except bias  (*c?.bc.fits)
 9. Use fibermaps or flats to identify apertures of each spectrum.
 9. Use fitted apertures to cut out 2-d spectra in thar,comp,science,twiflat
 10. Collapse the 2-d spectra to 1-d.
 4. Perform a rough calibration using low-res calibration lamps.
 4. Use the rough calibrations to identify lines in a higher density calibration lamp (e.g. ThAr).
 5. Fit a fifth order polynomial to every spectrum of each camera for each exposure.
 5. save fits for use as initial guesses of future fits.
 5. Create master skyflat file, save.
 6. Open all remaining types and divide out master flat, then save  (*c?.bcf.fits)
 13. Create master sky spectra by using a median fit of all sky fibers
 13. Remove continuums of the science and sky spectra and iteratively remove each sky line.
 13. Subtract the sky continuum from the science continuum and add back to the science.
 14. Combine the multiple sky subtracted, wavelength calibrated spectra from the multiple exposures.
 15. Fit the combined spectra to redshfit using cross-correlation with SDSS templates.

