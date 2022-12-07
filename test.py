import pandas as pd

from genetic_algorithm import Individual, GeneParameters, read_dataset
input_parameters = __import__('input')

df = pd.read_csv('BBBP/BBBP_clean.csv')
df['Name'] = df['Name'].astype(str)
print(df.head())
xyz_path = 'BBBP/xyz/'

print('Reading dataset')
mols = read_dataset(df, xyz_path)

print('Making gene parameters')
gene_parameters = [GeneParameters(**params) for params in input_parameters.descList]

print('Making gene set')
example_gene_set = [params.make_gene_set() for params in gene_parameters]

example_gene_set[0].cutoff = 5

print('Making individual')
example_individual = Individual(example_gene_set)

print('Getting score')
print(example_individual.get_score(df, mols))

print(example_individual.score)
