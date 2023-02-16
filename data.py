import warnings

import numpy as np
from quippy import descriptors
import ase
import os
import numpy as np

'''
Read the molecules from the xyz files and store them in the dataframe.
'''
def read_molecules(df, xyz_path):
    mol = []
    for row in df.itertuples():
        mol.append(ase.io.read(
            os.path.join(xyz_path, row.Name + '.xyz')))
    return mol

def soap_worker(df, parameter_strings):
    soaps = []
    for row in df.itertuples():
        soap = []
        for ps in parameter_strings:
            soap += list(descriptors.Descriptor(ps).calc(row.Mol)['data'][0])

        soap = np.array(soap)
        if np.isnan(soap).any():
            warnings.warn("NaN detected in molecule:\n{}".format(row))

        soaps.append(soap)

    return soaps