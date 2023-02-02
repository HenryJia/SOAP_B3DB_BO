import warnings
import multiprocessing as mp
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import numpy as np
from quippy import descriptors
import ase
import os
import numpy as np

def _calc_soap(mol, parameter_strings):
    soap = []
    for ps in parameter_strings:
        soap += list(descriptors.Descriptor(ps).calc(mol)['data'][0])
    return np.array(soap)

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

    def calc_soaps(self, parameter_strings):
        #soaps = self.pool.starmap(
            #_calc_soap, [(row.Mol, parameter_strings) for row in self.df.itertuples()])
        soaps = [_calc_soap(row.Mol, parameter_strings) for row in self.df.itertuples()]

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