# Just a minimal example to play with SOAP descriptors
import numpy as np
import quippy
from quippy import descriptors

import ase

at = ase.io.read('BBBP/xyz/26.xyz')

print(at)

desc = descriptors.Descriptor("soap average cutoff=3 l_max=4 n_max=4 atom_sigma=0.5 n_Z=1 Z={6}")
print(desc.calc(at)['data'].shape)

desc = descriptors.Descriptor("soap average cutoff=3 l_max=4 n_max=4 atom_sigma=0.5 n_Z=1 Z={6} n_species=1 species_Z={6}")
print(desc.calc(at)['data'].shape)

desc = descriptors.Descriptor("soap average cutoff=2.5 l_max=4 n_max=4 atom_sigma=0.5 n_Z=1 Z={6} n_species=1 species_Z={6}")
print(desc.calc(at)['data'].shape)