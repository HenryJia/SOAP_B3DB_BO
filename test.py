import multiprocessing as mp

import pandas as pd

import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from genetic_algorithm import Individual, Population, BestHistory, GeneParameters

import matplotlib.pyplot as plt

class SVCIndividual(Individual):
    def get_model_score(dataset, train_idx, test_idx):
        X, y = np.stack(dataset.df['SOAP'], axis=0), dataset.df['Class'].to_numpy()

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
        X, y = np.stack(dataset.df['SOAP'], axis=0), dataset.df['Class'].to_numpy()

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=3, random_state=0)
        clf.fit(X[train_idx], y[train_idx])

        pred_train = clf.predict(X[train_idx])
        pred_test = clf.predict(X[test_idx])

        mcc_train = matthews_corrcoef(y[train_idx], pred_train)
        mcc_test = matthews_corrcoef(y[test_idx], pred_test)

        return {'train_scores': mcc_train, 'test_scores': mcc_test}

if __name__ == '__main__':
    # Note: This is absolutely necessary or quippy will get stuck for some reason
    # Idea from: https://github.com/isl-org/Open3D/issues/1552
    mp.set_start_method('forkserver')

    input_parameters = __import__('input')

    df = pd.read_csv('./BBBP/BBBP_clean.csv')
    df['Name'] = df['Name'].astype(str)
    print(df.head())
    xyz_path = './BBBP/xyz/'

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
        example_gene_set, df, xyz_path)

    print('Getting score')
    example_individual.comp_soaps()
    example_individual.evaluate_model()

    result_dict = example_individual.results_dictionary

    mcc = np.mean(result_dict['test_scores'])

    print('MCC: {}'.format(mcc))

    pop = Population(
        lambda gene_set: RFIndividual(
            gene_set, df, xyz_path),
        population_parameters['population_size'],
        gene_parameters, maximise_scores=True, verbose=True)

    pop.initialise_population()

    pop.print_population()

    for gen in range(10):
        print(f"Generation {gen}")
        pop.next_generation()
        for ind in pop.population:
            print(f"{ind} has a MCC and score of: {np.mean(ind.results_dictionary['test_scores'])}, {ind.score}")
