# Exploratory data analysis of the MoleculeNet dataset
# with the Smooth Overlap of Atomic Positions (SOAP) GA code

import pandas as pd

from quippy import descriptors

#from genetic_algorithm import GeneParameters, Individual, Population
from genetic_algorithm import *
from utils import load_xyz

from tqdm import tqdm


descDict1 = {'lower': 1, 'upper': 10, 'centres': '{6}',
             'neighbours': '{6}', 'mu': 0, 
             'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'mutation_chance': 0.50, 
             'min_cutoff': 4, 'max_cutoff': 10, 'min_sigma': 0.1, 
             'max_sigma': 0.9, 'message_steps': 0}

descDict2 = {'lower': 1, 'upper': 5, 'centres': '{6}',
             'neighbours': '{6}', 'mu': 0, 
             'mu_hat': 0, 'nu': 2, 'nu_hat': 0, 'mutation_chance': 0.50, 
             'min_cutoff': 4, 'max_cutoff': 5, 'min_sigma': 0.1, 
             'max_sigma': 0.9, 'message_steps': 0}

num_gens = 5
best_sample, lucky_few, population_size, number_of_children = 4, 2, 12, 4
early_stop = 2
early_number = 3 
min_generations = 5

params1 = GeneParameters(**descDict1)
params2 = GeneParameters(**descDict2)

example_gene_set = params1.make_gene_set()

example_gene_set.get_soap_string()

example_gene_set_two = params2.make_gene_set()
gene_set_list = [example_gene_set, example_gene_set_two]
example_individual = Individual(gene_set_list[:1])

gene_parameters = [params1, params2]
pop = Population(best_sample, lucky_few, population_size, 
                 number_of_children, gene_parameters, 
                 maximise_scores = True)

print('Initialise population')
pop.initialise_population()

print('Training num_gens')
for _ in range(num_gens):
    pop.next_generation()
pop.print_population()

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
