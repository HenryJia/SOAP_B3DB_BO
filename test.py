import pandas as pd

import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier

from genetic_algorithm import Individual, GeneParameters, read_dataset

import matplotlib.pyplot as plt

class RandomForestIndividual(Individual):
    def get_model_score(dataset, train_idx, test_idx):
        X, y = dataset.to_numpy()

        clf = RandomForestClassifier(n_estimators=100)
        clf.fit(X[train_idx], y[train_idx])

        pred_train = clf.predict(X[train_idx])
        pred_test = clf.predict(X[test_idx])

        mcc_train = matthews_corrcoef(y[train_idx], pred_train)
        mcc_test = matthews_corrcoef(y[test_idx], pred_test)

        return {'train_scores': mcc_train, 'test_scores': mcc_test}


input_parameters = __import__('input')

df = pd.read_csv('BBBP/BBBP_clean.csv')
df['Name'] = df['Name'].astype(str)
print(df.head())
xyz_path = 'BBBP/xyz/'

print('Reading dataset')
df = read_dataset(df, xyz_path)
print(df.head())

print('Making gene parameters')
gene_parameters = [GeneParameters(**params) for params in input_parameters.descList]

print('Making gene set')
example_gene_set = [params.make_gene_set() for params in gene_parameters]

example_gene_set[0].cutoff = 5

print('Making individual')
example_individual = RandomForestIndividual(example_gene_set, df, xyz_path, target_col='Class')

print('Getting score')
example_individual.get_score()

result_dict = example_individual.results_dictionary

mcc = np.mean(result_dict['test_scores'])

print('MCC: {}'.format(mcc))