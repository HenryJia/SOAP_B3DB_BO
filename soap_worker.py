import warnings

import numpy as np

from quippy import descriptors

def soap_worker(dataset, parameter_strings):
    soaps = []
    for row in dataset.df.itertuples():
        soap = []
        for ps in parameter_strings:
            soap += list(descriptors.Descriptor(ps).calc(row.Mol)['data'][0])

        soap = np.array(soap)
        if np.isnan(soap).any():
            warnings.warn("NaN detected in molecule:\n{}".format(row))

        soaps.append(soap)

    return soaps



        