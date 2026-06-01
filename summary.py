# Calculate some summary statistics for the data
import argparse
from collections import Counter

import numpy as np
import pandas as pd

import ase

from data import read_molecules

# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--csv', type=str, help='Path to the csv file')
parser.add_argument('--xyz', type=str, help='Path to the xyz files')

# Parse the arguments
args = parser.parse_args()

# Read the csv file
df = pd.read_csv(args.csv)

# Read the molecules
df['Mol'] = read_molecules(df, args.xyz)

atom_counts = {}
for row in df.itertuples():
    atom_count = Counter(row.Mol.get_chemical_symbols())
    for k, v in atom_count.items():
        if k not in atom_counts:
            atom_counts[k] = 0
        atom_counts[k] += v

print("Atom counts:", atom_counts)