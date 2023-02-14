import warnings

import numpy as np
from quippy import descriptors
import ase
import os
import numpy as np

class SOAPDataset(object):
    def __init__(self, df, target_col, workers=16) -> None:
        self.df = df
        self.target_col = target_col
        self.workers = workers

        #self.pool = ThreadPool(self.workers)

    def read_molecules(self, xyz_path):
        mol = []
        for row in self.df.itertuples():
            mol.append(ase.io.read(
                os.path.join(xyz_path, row.Name + '.xyz')))
        self.df['Mol'] = mol

    # Whilst this function does exist, works and is tested, it is not used
    # This function is quite slow all things considered. It doesn't take advantage of
    # Any multiprocessing, since we can't pickle this member function.
    # This function is best used for debugging purposes.
    def calc_soaps(self, parameter_strings):
        soaps = []
        for row in self.df.itertuples():
            soap = []
            for ps in parameter_strings:
                soap += list(descriptors.Descriptor(ps).calc(row.Mol)['data'][0])
            soaps += [np.array(soap)]

        self.df['SOAP'] = soaps

        for row in self.df.itertuples():
            if np.isnan(row.SOAP).any():
                warnings.warn("NaN detected in molecule:\n{}".format(row))
        
        self.df['SOAP'] = self.df['SOAP'].apply(lambda x: np.nan_to_num(x))

    def to_numpy(self):
        soaps = []
        for row in self.df.itertuples():
            soaps.append(row.SOAP)
        return np.array(soaps), self.df[self.target_col].to_numpy()

    def __getitem__(self, index):
        return self.df.iloc[index]['SOAP'], self.df.iloc[index][self.target_col]

    def __len__(self):
        return len(self.df)