import numpy as np
from collections import OrderedDict

class InstrumentState:
    def __init__(self, obs_config=None):
        if obs_config is None:
             self.cameras=['r','b']
             self.opamps=[1,2,3,4]
             self.deadfibers=[]
             self.swapped_fibers=[]
             self.binning='2x2'
             self.readout='Slow'
             self.resolution='LowRes'
             self.filter=None
             self.configuration=None
             self.wavemin = 4500
             self.wavemax = 7500
             self.coarse_lamp_names = ['Hg','Ne'] #['Hg','Ar', 'Ne', 'Xe']
             self.fine_lamp_names = ['Th', 'Ar'] #['ThAr']
        else:
            self.binning = obs_config['CCDS']['binning']
            self.readout = obs_config['CCDS']['readout_speed']
            self.resolution = obs_config['INSTRUMENT']['m2fs_res_mode']
            self.configuration = obs_config['INSTRUMENT']['config']
            self.wavemin = float(obs_config['INSTRUMENT']['wavemin'])
            self.wavemax = float(obs_config['INSTRUMENT']['wavemax'])

            self.cameras =  self.str_listify(obs_config['CCDS']['cameras'],expected_type=str)
            self.opamps =  self.str_listify(obs_config['CCDS']['opamps'],expected_type=int)
            self.deadfibers = self.str_listify(obs_config['INSTRUMENT']['deadfibers'],expected_type=str)
            self.swapped_fibers = self.str_listify(obs_config['INSTRUMENT']['replacements'],expected_type=str)
            self.coarse_lamp_names = self.str_listify(obs_config['LAMPS']['coarse_lamp_names'],expected_type=str)
            self.fine_lamp_names = self.str_listify(obs_config['LAMPS']['fine_lamp_names'],expected_type=str)

            self.filter = obs_config['CCDS']['filter']
            if self.filter.lower() == 'none':
                self.filter = None

        self.lower_half_fibs,self.upper_half_fibs = OrderedDict(),OrderedDict()
        self.full_fibs,self.overlapping_fibs = {},{}

        for camera in self.cameras:
            self.create_proper_spatial_splits(camera)

        if self.opamps is None:
            self.opamps = [None]

        if self.resolution.lower() != 'lowres':
            print("WARNING: Only low resolution is supported. Single order medres or highres may work, but is untested.")
        if self.binning != '2x2':
            print("WARNING: This has only been used and tested for 2x2. Others should proceed with caution,\
                    especially during wavelength calibration")

    def str_listify(self,rawstring,expected_type=int):
        string = str(rawstring).strip('[]() \t\n')

        def classify_str(strval,expected_type):
            strval = strval.strip(' \t')
            if strval.lower() == 'none':
                return None
            elif expected_type in [int, float, np.float, np.int, np.float64] and strval.isnumeric:
                return expected_type(strval)
            elif expected_type is str and strval.isalnum:
                return expected_type(strval)
            elif strval.isnumeric:
                return expected_type(strval)
            elif strval.isalpha:
                return strval
            else:
                print("When digesting instrument calibration file, I couldn't understand {}".format(strval))
                return strval

        if ',' in string:
            out_num_list = []
            vallist = str(string).split(',')
            for splitstr in vallist:
                out_num_list.append(classify_str(splitstr,expected_type))
        else:
            out_num_list = [classify_str(string, expected_type)]

        ## Correct bug where empty brackets returned a list with an empty string instead of an empty list
        if len(out_num_list) == 1 and out_num_list[0] == '':
            out_num_list = []
        return np.asarray(out_num_list)

    def create_proper_spatial_splits(self,camera):
        full_fibs = []
        upper_half_fibs = []
        lower_half_fibs = []

        if camera == 'b':
            tet_order = np.arange(8,0,-1)
        else:
            tet_order = np.arange(1,9,1)

        tet_order_b = tet_order[:4]
        tet_order_u = tet_order[4:]

        for tet in tet_order_b:
            for fib in np.arange(1,17):
                fibnm = '{}{}{:02d}'.format(camera,tet,fib)
                if fibnm not in self.deadfibers:
                    full_fibs.append(fibnm)
                    lower_half_fibs.append(fibnm)

        for tet in tet_order_u:
            for fib in np.arange(1, 17):
                fibnm = '{}{}{:02d}'.format(camera, tet, fib)
                if fibnm not in self.deadfibers:
                    full_fibs.append(fibnm)
                    upper_half_fibs.append(fibnm)

        ## because we want to go from outside-inward, flip the order of the top fibs
        lower_half_fibs = np.array(lower_half_fibs)
        upper_half_fibs = np.array(upper_half_fibs)[::-1]

        last_low = lower_half_fibs[-1].copy()
        last_upper = upper_half_fibs[-1].copy()

        self.lower_half_fibs[camera] = lower_half_fibs
        self.upper_half_fibs[camera] = upper_half_fibs

        self.full_fibs[camera] = np.array(full_fibs)
        self.overlapping_fibs[camera] = np.array([last_low,last_upper])
