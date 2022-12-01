# Exploratory data analysis of the MoleculeNet dataset
# with the Smooth Overlap of Atomic Positions (SOAP) GA code

import pandas as pd

from quippy import descriptors

from utils import load_xyz

from tqdm import tqdm


num_gens = 100
best_sample, lucky_few, population_size, number_of_children = 4, 2, 12, 4
early_stop = 2
early_number = 3 
min_generations = 5


df = pd.read_csv('BBBP/BBBP_clean.csv')

xyz = []
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    xyz.append(load_xyz('BBBP/xyz/' + str(row['num']) + '.xyz'))

df = df.assign(xyz=xyz)
print(df.head())

soaps = []
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    soaps.append(descriptors.Descriptor("soap cutoff=4 l_max=3 n_max=4 normalize=T atom_sigma=0.5 n_Z=1 Z={12} ").calc(row['xyz']))

df = df.assign(soaps=soaps)

print(df.head())
