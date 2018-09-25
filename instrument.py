


class InstrumentState:
    def __init__(self,cameras=['r','b'],opamps=[1,2,3,4],deadfibers=None,binning='2x2',\
                 readout='Slow',resolution='LoRes',filter=None,configuration=None):
        self.cameras = cameras
        self.opamps = opamps
        self.deadfibers = deadfibers
        self.binning = binning
        self.readout = readout
        self.resolution = resolution
        self.filter = filter
        self.configuration = configuration
        if self.resolution.lower() != 'lowres':
            print("WARNING: Only low resolution is supported. Single order medres or highres may work, but is untested.")
        if self.binning != '2x2':
            print("WARNING: This has only been used and tested for 2x2. Others should proceed with caution,\
                    especially during wavelength calibration")