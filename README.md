M2FSreduce
=========

Note:
--- 
* Reduction software for Magellan M2FS spectra

* Several subfunctions were based on Dan Gifford's OSMOSreduce package used to reduce 2.4m MDM telescope slit-mask spectra.

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

There is also an implicit assumed configuration file for the specific instrument setup. Example:
'instrument_11C_kremin.ini'

You can also create your own quickreduce where those variables are defined within. 
Example: A02_quickreduce.py

