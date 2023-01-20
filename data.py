import warnings
import pandas as pd
import numpy as np
from quippy import descriptors
import ase
import os
import numpy as np


class SOAPDataset(object):
    def __init__(self, df, target_col) -> None:
        self.df = df
        self.target_col = target_col

    def read_molecules(self, xyz_path):
        mol = []
        for idx, row in self.df.iterrows():
            mol.append(ase.io.read(
                os.path.join(xyz_path, row['Name'] + '.xyz')))
        self.df['mol'] = mol

    def calc_soaps(self, parameter_strings):
        soaps = []
        for idx, row in self.df.iterrows():
            soaps.append([])
            for ps in parameter_strings:
                soaps[-1] += list(descriptors.Descriptor(ps).calc(row['Mol'])['data'][0])
            soaps[-1] = np.array(soaps[-1])
        self.df['SOAP'] = soaps

        for idx, row in self.df.iterrows():
            if np.isnan(row['SOAP']).any():
                warnings.warn("NaN detected in molecule:\n{}".format(row))
        
        self.df['SOAP'] = self.df['SOAP'].apply(lambda x: np.nan_to_num(x))

    def __getitem__(self, index):
        return self.df.iloc[index]['SOAP'], self.df.iloc[index][self.target_col]

    def __len__(self):
        return len(self.df)