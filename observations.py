
import numpy as np
from collections import OrderedDict



#TODO Use all initial comps (Average) if more than one is paired to a single final comp
class Observations:
    def __init__(self,filenumbers,obs_pairing_strategy):
        self.comparc_cs = filenumbers['coarse_comp']
        if filenumbers['fine_comp'] is None:
            self.comparc_fs = np.array([None]*len(self.comparc_cs))
        elif len(filenumbers['fine_comp']) == 1 and filenumbers['fine_comp'][0] is None:
            self.comparc_fs = np.array([None] * len(self.comparc_cs))
        else:
            self.comparc_fs = filenumbers['fine_comp']

        self.flats = filenumbers['twiflat']
        self.scis = filenumbers['science']
        self.nobs = len(self.comparc_cs)


        self.two_step_comparc = (self.comparc_fs[0] is not None)

        self.observations = OrderedDict()
        self.comparc_pairs = OrderedDict()
        self.obs_set_index_lookup = OrderedDict()
        self.cal_pair_index_lookup = OrderedDict()

        self.obs_pairing_strategy = obs_pairing_strategy

        if self.obs_pairing_strategy not in ['nearest','user']: # 'unique',
            # 'nearest' chooses closest filenumber to sci
            # 'unique' pairs comps with science as uniquely as possible
            #             while it's second priority is to pair closest
            #             NOTE YET IMPLEMENTED
            # 'user'  pairs in the exact order given in the filenum lists
            self.obs_pairing_strategy = 'nearest'

        self.assign_observational_sets()

    def assign_observational_sets(self):
        """
        # 'nearest' chooses closest filenumber to sci
        # 'unique' pairs comps with science as uniquely as possible
        #             while it's second priority is to pair closest
        # 'user'  pairs in the exact order given in the filenum list
        :return: None
        """
        comparc_cs = self.comparc_cs.astype(int)
        if self.two_step_comparc:
            comparc_fs = self.comparc_fs.astype(int)
        else:
            comparc_fs = self.comparc_fs

        flats = self.flats
        scis = self.scis

        comparc_pairs = OrderedDict()
        observation_sets = OrderedDict()
        obs_set_index_lookup = OrderedDict()
        comparc_c_pair_index_lookup = OrderedDict()
        comparc_f_pair_index_lookup = OrderedDict()
        if self.obs_pairing_strategy == 'unique':
            print("Unique observation pairing isn't yet implemented, defaulting to 'nearest'")
            self.obs_pairing_strategy = 'nearest'

        if self.obs_pairing_strategy == 'user':
            for ii,sci,flat,comparc_c,comparc_f in enumerate(zip(scis,flats,comparc_cs,comparc_fs)):
                comparc_pairs[ii] = (comparc_c,comparc_f)
                observation_sets[ii] = (sci,flat,comparc_c,comparc_f)

        if self.obs_pairing_strategy == 'nearest':
            if self.two_step_comparc:
                for ii,comparc_f in enumerate(comparc_fs):
                    nearest_comparc_c = comparc_cs[np.argmin(np.abs(comparc_cs - comparc_f))]
                    comparc_pairs[ii] = (nearest_comparc_c,comparc_f)
                    comparc_c_pair_index_lookup[nearest_comparc_c] = ii
                    comparc_f_pair_index_lookup[comparc_f] = ii
            else:
                for ii, comparc_c in enumerate(comparc_cs):
                    comparc_pairs[ii] = (comparc_c, None)
                    comparc_c_pair_index_lookup[comparc_c] = ii
                    comparc_f_pair_index_lookup[comparc_c] = ii
            for ii, sci in enumerate(scis):
                nearest_comparc_f = comparc_fs[np.argmin(np.abs(comparc_fs - sci))]
                nearest_comparc_pair_ind = comparc_f_pair_index_lookup[nearest_comparc_f]
                comparc_c,comparc_f = comparc_pairs[nearest_comparc_pair_ind]
                observation_sets[ii] = (sci, comparc_c,comparc_f,nearest_comparc_pair_ind)
                obs_set_index_lookup[sci] = ii

        self.observations = observation_sets
        self.comparc_pairs = comparc_pairs
        self.obs_set_index_lookup = obs_set_index_lookup
        self.comparc_c_pair_index_lookup = comparc_c_pair_index_lookup
        self.comparc_f_pair_index_lookup = comparc_f_pair_index_lookup

    def return_comparc_pairs(self):
        return self.comparc_pairs

    def return_observation_sets(self):
        return self.observations
