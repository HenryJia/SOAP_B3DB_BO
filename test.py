import pandas as pd

import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from genetic_algorithm import Individual, Population, BestHistory, GeneParameters

import matplotlib.pyplot as plt

class SVCIndividual(Individual):
    def get_model_score(dataset, train_idx, test_idx):
        X, y = dataset.to_numpy()

        clf = SVC(
            kernel='rbf', random_state=0)
        clf.fit(X[train_idx], y[train_idx])

        pred_train = clf.predict(X[train_idx])
        pred_test = clf.predict(X[test_idx])

        mcc_train = matthews_corrcoef(y[train_idx], pred_train)
        mcc_test = matthews_corrcoef(y[test_idx], pred_test)

        return {'train_scores': mcc_train, 'test_scores': mcc_test}

class RFIndividual(Individual):
    def get_model_score(dataset, train_idx, test_idx):
        X, y = dataset.to_numpy()

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=0)
        clf.fit(X[train_idx], y[train_idx])

        pred_train = clf.predict(X[train_idx])
        pred_test = clf.predict(X[test_idx])

        mcc_train = matthews_corrcoef(y[train_idx], pred_train)
        mcc_test = matthews_corrcoef(y[test_idx], pred_test)

        return {'train_scores': mcc_train, 'test_scores': mcc_test}

input_parameters = __import__('input')

df = pd.read_csv('/home/nvme/BBBP/BBBP_clean.csv')
df['Name'] = df['Name'].astype(str)
print(df.head())
xyz_path = '/home/nvme/BBBP/xyz/'

print(df.head())

print('Making parameters')
gene_parameters = [GeneParameters(**params)
                   for params in input_parameters.descList]
population_parameters = input_parameters.population_parameters

print('Making gene set')
example_gene_set = [params.make_gene_set() for params in gene_parameters]

example_gene_set[0].cutoff = 5

print('Making individual')
example_individual = RFIndividual(
    example_gene_set, df, xyz_path, target_col='Class')

print('Getting score')
example_individual.get_score()

result_dict = example_individual.results_dictionary

mcc = np.mean(result_dict['test_scores'])

print('MCC: {}'.format(mcc))

pop = Population(
    lambda gene_set: RFIndividual(
        gene_set, df, xyz_path, target_col='Class'),
    population_parameters['best_sample'], population_parameters['lucky_few'],
    population_parameters['population_size'], population_parameters['number_of_children'],
    gene_parameters, maximise_scores=True, verbose=True)

pop.initialise_population()

pop.print_population()

num_gens = 5
early_stop = 3
early_number = 3
min_generations = 5

hist = BestHistory(early_stop, early_number, min_generations)

for gen in range(num_gens):
    if hist.converged:
        break
    print(f"Generation {gen}")
    pop.next_generation()
    hist.append(pop)
    for ind in pop.population:
        print(f"{ind} has a MCC of: {np.mean(ind.results_dictionary['test_scores'])}")