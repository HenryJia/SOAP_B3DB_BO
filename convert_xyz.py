import os
from argparse import ArgumentParser
import warnings
warnings.filterwarnings("error")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.SaltRemover import SaltRemover

import pandas as pd

from tqdm import tqdm

parser = ArgumentParser(description='Convert smiles to xyz')
parser.add_argument('--input', help='input file containing smiles and molecule names')
parser.add_argument('--moleculenet', help='use moleculenet', default=False, action='store_true')
parser.add_argument('--b3db', help='use b3db', default=False, action='store_true')
parser.add_argument('--output', help='output directory')
parser.add_argument('--output_df', help='output dataframe')

args = parser.parse_args()

# Clean either moleculenet or B3DB but not both at the same time
assert args.moleculenet != args.b3db, 'Please choose either moleculenet or B3DB'

if args.moleculenet:
    df = pd.read_csv(args.input)
else:
    df = pd.read_csv(args.input, sep='\t')


print(df.head())

failed = []
remover = SaltRemover(defnData="[Cl,Na,O,Br,Ca,H]")
for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    try:
        smiles = row['smiles'] if args.moleculenet else row['SMILES']
        num = row['num'] if args.moleculenet else row['NO.']

        mol = Chem.MolFromSmiles(smiles)
        mol = remover.StripMol(mol)

        mol = Chem.AddHs(mol)
        success = AllChem.EmbedMolecule(mol, randomSeed=0xf00d)
        out = Chem.MolToXYZBlock(mol)

        if success != 0:
            raise

        with open(os.path.join(args.output, str(num) + '.xyz'), 'w') as f:
            f.write(out)

    except Exception as e:
        print('Embedding failed for {} {}'.format(num, smiles))
        print(e)
        failed += [i]

for i in failed:
    smiles = df.iloc[i]['smiles'] if args.moleculenet else df.iloc[i]['SMILES']
    num = df.iloc[i]['num'] if args.moleculenet else df.iloc[i]['NO.']
    print('Failed to generate XYZ coordinates for {} {}'.format(num, smiles))

df = df.drop(failed)
df.to_csv(args.output_df, index=False)