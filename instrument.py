import numpy as np


class InstrumentState:
    def __init__(self, obs_config=None):
        if obs_config is None:
             self.cameras=['r','b']
             self.opamps=[1,2,3,4]
             self.deadfibers=None
             self.binning='2x2'
             self.readout='Slow'
             self.resolution='LowRes'
             self.filter=None
             self.configuration=None
             self.coarse_lamp_names = ['HgAr', 'NeAr', 'Xe']
             self.fine_lamp_names = ['ThAr']
        else:
            self.binning = obs_config['CCDS']['binning']
            self.readout = obs_config['CCDS']['readout_speed']
            self.resolution = obs_config['INSTRUMENT']['m2fs_res_mode']
            self.configuration = obs_config['INSTRUMENT']['config']

            self.cameras =  self.str_listify(obs_config['CCDS']['cameras'],expected_type=str)
            self.opamps =  self.str_listify(obs_config['CCDS']['opamps'],expected_type=int)
            self.deadfibers = self.str_listify(obs_config['INSTRUMENT']['deadfibers'],expected_type=str)
            self.coarse_lamp_names = self.str_listify(obs_config['LAMPS']['coarse_lamp_names'],expected_type=str)
            self.fine_lamp_names = self.str_listify(obs_config['LAMPS']['fine_lamp_names'],expected_type=str)

            self.filter = obs_config['CCDS']['filter']
            if self.filter.lower() == 'none':
                self.filter = None

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

        return np.asarray(out_num_list)
