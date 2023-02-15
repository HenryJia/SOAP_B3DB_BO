import warnings

import numpy as np
from quippy import descriptors
import ase
import os
import numpy as np

class MoleculeDataset(object):
    '''
    A class to hold a dataset of molecules.
    This class is intended to be used with the SOAPWorker class.

    This is meant to be a thin wrapper around a pandas dataframe.
    It mainly exists to provide some convenience functions for reading molecules from xyz files

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing the molecules to be processed.
        The dataframe must contain a column named 'Name' which contains the name of the molecule.
        The dataframe will be modified to contain a column named 'Mol' which contains the molecule as an ase.Atoms object.
    target_col : str
        The name of the column in the dataframe which contains the target values.
    '''
    def __init__(self, df) -> None:
        self.df = df

    '''
    Read the molecules from the xyz files and store them in the dataframe.
    '''
    def read_molecules(self, xyz_path):
        mol = []
        for row in self.df.itertuples():
            mol.append(ase.io.read(
                os.path.join(xyz_path, row.Name + '.xyz')))
        self.df['Mol'] = mol

    '''
    Calculate the SOAP vectors for the molecules in the dataset.
    Note that this is a very slow process and should be done in parallel using the SOAPWorker class.
    This function is provided for testing purposes mainly.
    '''
    def calc_soaps(self, parameter_strings):
        soaps = []
        for row in self.df.itertuples():
            soap = []
            for ps in parameter_strings:
                soap += list(descriptors.Descriptor(ps).calc(row.Mol)['data'][0])

            soap = np.array(soap)
            if np.isnan(soap).any():
                warnings.warn("NaN detected in molecule:\n{}".format(row))

            soaps += [soap]
        
        self.df['SOAP'] = soaps
        #return np.stack(soaps, axis=0)

    def __len__(self):
        return len(self.df)