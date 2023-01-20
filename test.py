import pandas as pd

import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay

from genetic_algorithm import Individual, GeneParameters, read_dataset

import matplotlib.pyplot as plt

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
example_individual = Individual(example_gene_set, df, xyz_path, target_col='Class')

print('Getting score')
example_individual.get_score(df)

result_dict = example_individual.results_dictionary

y_test = []
pred_test = []
for y, p in zip(result_dict['y_test_actual'], result_dict['y_test_predictions']):
    y_test.append(y)
    pred_test.append(p)

y_test = np.concatenate(y_test)
pred_test = np.concatenate(pred_test)

# Calculate ROC AUC
roc_auc = roc_auc_score(y_test, pred_test)

# Calculate Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, pred_test)

# Calculate confusion matrix
cm = confusion_matrix(y_test, pred_test, normalize='true')

print('ROC AUC: {}'.format(roc_auc))
print('MCC: {}'.format(mcc))

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BBBP-', 'BBBP+'])
disp.plot(cmap='Blues')
plt.savefig('confusion_matrix.png')