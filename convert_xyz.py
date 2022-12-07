import os
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("error")

from rdkit import Chem
from rdkit.Chem import AllChem

import pandas as pd

from tqdm import tqdm

parser = ArgumentParser(description='Convert smiles to xyz')
parser.add_argument('--input', help='input file containing smiles and molecule names')
parser.add_argument('--output', help='output directory')
parser.add_argument('--output_df', help='output dataframe')

args = parser.parse_args()

df = pd.read_csv(args.input)
df_out = pd.DataFrame(columns=['Name', 'ChemName', 'Smiles', 'Class'])
print(df.head())

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        mol = Chem.MolFromSmiles(row['smiles'])
        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        out = Chem.MolToXYZBlock(mol)

        if success != 0:
            raise

        with open(os.path.join(args.output, str(row['num']) + '.xyz'), 'w') as f:
            f.write(out)

        row_out = pd.DataFrame({'Name': row['num'], 'ChemName': row['name'], 'Smiles': row['smiles'], 'Class': row['p_np']}, index=[0])
        df_out = pd.concat([df_out, row_out])
    except Exception as e:
        print('Embedding failed for {} {}'.format(row['name'], row['smiles']))
        print(e)

df_out.to_csv(args.output_df, index=False)