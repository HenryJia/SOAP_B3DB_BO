import faulthandler

faulthandler.enable()

from genetic_algorithm import *
input_parameters = __import__('input')

gene_parameters = [GeneParameters(**params) for params in input_parameters.descList]

example_gene_set = [params.make_gene_set() for params in gene_parameters]

example_gene_set[0].cutoff = 5

example_individual = Individual(example_gene_set)

print(example_individual.get_score())

print(example_individual.score)
