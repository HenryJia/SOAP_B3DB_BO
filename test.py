import multiprocessing as mp
import warnings
warnings.simplefilter("ignore") # Ignore warnings for now

import pandas as pd

import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, callbacks

from data import read_molecules
from genetic_algorithm import Individual, Population, GeneParameters
from modules import MLP, SimpleResNetBlock, SimpleResNet

import matplotlib.pyplot as plt


class NNIndividual(Individual):
    def get_model_score(df, train_index, test_index):
        X, y = np.stack(df['SOAP'], axis=0), df['Class'].to_numpy()
        X = X.astype(np.float32)

        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X[train_index]), torch.Tensor(y[train_index, None]))
        test_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X[test_index]), torch.Tensor(y[test_index, None]))

        # Our dataset is small, so if we set num_workers > 0, we end up spending more time setting up the workers than we do actually training.
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        model = SimpleResNet(input_dim=X.shape[-1], layer_size=128, depth=16)
        trainer = Trainer(
            max_epochs=200,
            accelerator='gpu',
            devices=[0],
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)],
            check_val_every_n_epoch=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, train_loader, test_loader)

        model.eval()

        pred_train = model(torch.Tensor(X[train_index])).detach().numpy().squeeze()
        pred_test = model(torch.Tensor(X[test_index])).detach().numpy().squeeze()

        mcc_train = matthews_corrcoef(y[train_index], pred_train > 0.5)
        mcc_test = matthews_corrcoef(y[test_index], pred_test > 0.5)

        return {'train_scores': mcc_train, 'test_scores': mcc_test}

class SVCIndividual(Individual):
    def get_model_score(df, train_idx, test_idx):
        X, y = np.stack(df['SOAP'], axis=0), df['Class'].to_numpy()

        clf = SVC(
            kernel='rbf', random_state=0)
        clf.fit(X[train_idx], y[train_idx])

        pred_train = clf.predict(X[train_idx])
        pred_test = clf.predict(X[test_idx])

        mcc_train = matthews_corrcoef(y[train_idx], pred_train)
        mcc_test = matthews_corrcoef(y[test_idx], pred_test)

        return {'train_scores': mcc_train, 'test_scores': mcc_test}

class RFIndividual(Individual):
    def get_model_score(df, train_idx, test_idx):
        X, y = np.stack(df['SOAP'], axis=0), df['Class'].to_numpy()

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

    df['Mol'] = read_molecules(df, xyz_path)

    print(df.head())

    print('Making parameters')
    gene_parameters = [GeneParameters(**params)
                    for params in input_parameters.descList]
    population_parameters = input_parameters.population_parameters

    print('Making gene set')
    example_gene_set = [params.make_gene_set() for params in gene_parameters]

    example_gene_set[0].cutoff = 5

    print('Making individual')
    example_individual = NNIndividual(
        example_gene_set, df)

    print('Getting score')
    getter = example_individual.comp_soaps()
    example_individual.evaluate_model(getter)

    result_dict = example_individual.results_dictionary

    mcc = np.mean(result_dict['test_scores'])

    print('MCC: {}'.format(mcc))

    pop = Population(
        lambda gene_set: NNIndividual(
            gene_set, df),
        population_parameters['population_size'],
        gene_parameters, maximise_scores=True, verbose=True)

    pop.initialise_population()

    pop.print_population()

    for gen in range(30):
        print(f"Generation {gen}")
        pop.next_generation()
        mcc = []
        score = []
        ind_str = []
        for ind in pop.population:
            mcc.append(np.mean(ind.results_dictionary['test_scores']))
            score.append(ind.score)
            ind_str.append(str(ind))

        print(f"Best {ind_str[np.argmax(mcc)]} has a MCC and score of: {np.max(mcc)}, {np.max(score)}")
