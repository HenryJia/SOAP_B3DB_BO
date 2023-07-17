# Run just a model on the SOAPs without any genetic algorithm
import os
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score, r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

import umap
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, callbacks

import ase
from data import read_molecules, soap_worker
from genetic_algorithm import Individual, Population, GeneParameters
from modules import MLP, SimpleResNetBlock, SimpleResNet

import matplotlib.pyplot as plt
import plotly.express as px


class NNIndividual(Individual):
    def get_model_score(df, train_index, val_index, test_index):
        X, y = np.stack(df['SOAP'], axis=0), df['Class'].to_numpy()
        X = X.astype(np.float32)

        X_train = X[train_index]
        X_val = X[val_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_val = y[val_index]
        y_test = y[test_index]

        #sm = SMOTE(random_state=42)
        #X_train, y_train = sm.fit_resample(X_train, y_train)

        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train), torch.Tensor(y_train[:, None]))
        val_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_val), torch.Tensor(y_val[:, None]))
        test_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_test), torch.Tensor(y_test[:, None]))

        # Our dataset is small, so if we set num_workers > 0, we end up spending more time setting up the workers than we do actually training.
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=0)
        #test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        model = SimpleResNet(input_dim=X.shape[-1], depth=4, layer_size=64)
        trainer = Trainer(
            max_epochs=100,
            accelerator='gpu',
            devices=[0],
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)],
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, train_loader, val_loader)

        model.eval()

        pred_train = model(torch.Tensor(X_train)).detach().numpy().squeeze()
        pred_val = model(torch.Tensor(X_val)).detach().numpy().squeeze()
        pred_test = model(torch.Tensor(X_test)).detach().numpy().squeeze()

        #mcc_train = matthews_corrcoef(y_train, pred_train > 0.5)
        #mcc_test = matthews_corrcoef(y_test, pred_test > 0.5)

        # Gabriele likes MCC but most other papers use AUC
        # So we will use AUC for comparison's sake
        auc_train = roc_auc_score(y_train, pred_train)
        auc_val = roc_auc_score(y_val, pred_val)
        auc_test = roc_auc_score(y_test, pred_test)

        train_cm = confusion_matrix(y_train, pred_train > 0.5, normalize='true')
        val_cm = confusion_matrix(y_val, pred_val > 0.5, normalize='true')
        test_cm = confusion_matrix(y_test, pred_test > 0.5, normalize='true')

        return {
            'train_scores': auc_train, 'val_scores': auc_val, 'test_scores': auc_test,
            'train_tp': train_cm[1, 1], 'train_tn': train_cm[0, 0],
            'val_tp': val_cm[1, 1], 'val_tn': val_cm[0, 0],
            'test_tp': test_cm[1, 1], 'test_tn': test_cm[0, 0]}

class RFIndividual(Individual):
    def get_model_score(df, train_idx, test_idx):
        X, y = np.stack(df['SOAP'], axis=0), df['Class'].to_numpy()

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=0)
        clf.fit(X[train_idx], y[train_idx])

        #pred_train = clf.predict(X[train_idx])
        #pred_test = clf.predict(X[test_idx])

        #mcc_train = matthews_corrcoef(y[train_idx], pred_train)
        #mcc_test = matthews_corrcoef(y[test_idx], pred_test)

        pred_train = clf.predict_proba(X[train_idx])[:, 1]
        pred_test = clf.predict_proba(X[test_idx])[:, 1]

        auc_train = roc_auc_score(y[train_idx], pred_train)
        auc_test = roc_auc_score(y[test_idx], pred_test)

        return {'train_scores': auc_train, 'test_scores': auc_test}

if __name__ == '__main__':

    # Set up the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='Path to the csv file')
    parser.add_argument('--xyz', type=str, help='Path to the xyz files')
    parser.add_argument('--moleculenet', help='use moleculenet', default=False, action='store_true')
    parser.add_argument('--b3db', help='use b3db', default=False, action='store_true')

    parser.add_argument('--n_fold', help='number of folds for kfold cross validation', default=5 ,type=int)
    parser.add_argument('--n_repeats', help='maximum number of repeats to run our experiments', default=10 ,type=int)
    parser.add_argument('--smote', help='use smote', default=False, action='store_true')
    parser.add_argument('--n_estimators', help='number of estimators for random forest', default=100, type=int)
    parser.add_argument('--max_depth', help='maximum depth of random forest', default=None, type=int)
    parser.add_argument('--verbose', help='verbosity of random forest', default=0, type=int)
    parser.add_argument('--name', help='name of the experiment', default='test', type=str)

    ga_params = parser.add_argument_group('Genetic Algorithm Parameters')
    ga_params.add_argument('--lower', type=int, help='Lower bound of the SOAP', default=2)
    ga_params.add_argument('--upper', type=int, help='Upper bound of the SOAP', default=10)
    ga_params.add_argument('--centres', type=list, nargs='+', help='Centres of the SOAP', default=[1, 6, 7, 8, 9, 16, 17, 35])
    ga_params.add_argument('--neighbours', type=list, nargs='+', help='Centres of the SOAP', default=[1, 6, 7, 8, 9, 16, 17, 35])
    ga_params.add_argument('--nu_R', type=int, help='nu_R of the SOAP', default=1)
    ga_params.add_argument('--nu_S', type=int, help='nu_S of the SOAP', default=0)
    ga_params.add_argument('--mutation_chance', type=float, help='Mutation chance of the GA', default=0.2)
    ga_params.add_argument('--min_cutoff', type=float, help='Minimum cutoff of the SOAP', default=5)
    ga_params.add_argument('--max_cutoff', type=float, help='Maximum cutoff of the SOAP', default=20)
    ga_params.add_argument('--min_sigma', type=float, help='Minimum sigma of the SOAP', default=0.1)
    ga_params.add_argument('--max_sigma', type=float, help='Maximum sigma of the SOAP', default=1.5)
    ga_params.add_argument('--message_steps', type=int, help='Message steps of the SOAP. This does nothing for now', default=0)
    ga_params.add_argument('--population_size', type=int, help='Population size of the GA', default=20)
    ga_params.add_argument('--best_sample', type=int, help='Number of best samples to use for the next generation', default=6)
    ga_params.add_argument('--lucky_few', type=int, help='Number of lucky few to use for the next generation', default=2)
    ga_params.add_argument('--num_children', type=int, help='Number of children to create for the next generation', default=5)
    ga_params.add_argument('--num_generations', type=int, help='Number of generations of the GA', default=30)

    args = parser.parse_args()

    regression = False
    data = pd.read_csv(args.csv)

    if args.moleculenet:
        y = data.p_np.to_numpy().astype(int)
    elif args.b3db: # Note that B3DB data is tab separated
        if 'regression' in args.csv:
            regression = True
            assert not args.smote, 'Cannot use SMOTE with regression, only classification'
    else:
        raise ValueError('Must specify either moleculenet or b3db')

    mol = []
    for idx, row in data.iterrows():
        num = row['num'] if args.moleculenet else row['NO.']
        mol.append(ase.io.read(os.path.join(args.xyz, str(num) + '.xyz')))
    data['Mol'] = mol

    print(data.head())

    # Set up the SOAP descriptor
    num_centres = len(args.centres)
    num_neighbours = len(args.neighbours)

    centres_string = '{' + ', '.join([str(c) for c in args.centres]) + '}'
    neighbours_string = '{' + ', '.join([str(n) for n in args.neighbours]) + '}'

    soap_string = "soap average cutoff={} l_max={} n_max={} atom_sigma={} n_Z={} Z={} n_species={} species_Z={} n_R={} n_S={}".format(
        args.cutoff, args.l_max, args.n_max, args.atom_sigma, num_centres, centres_string, num_neighbours, neighbours_string, args.nu_R, args.nu_S
    )

    print("SOAP string:", soap_string)

    # Compute the soap descriptors
    data['SOAP'] = soap_worker(data, [soap_string])

    X = np.stack(data['SOAP'].to_numpy(), axis=0)
    if args.moleculenet:
        y = data['p_np'].to_numpy().astype(int)
    elif args.b3db:
        if regression:
            y = data['logBB'].to_numpy().astype(float)
        else:
            y = (data['BBB'] == 'BBB+').to_numpy().astype(int)    

    kf = RepeatedKFold(n_splits=args.n_fold, n_repeats=args.n_repeats, random_state=42)
    if regression:
        r2 = 0
        rmse = 0
    else:
        cm = np.zeros((2, 2))
        mcc = 0
        auc = 0

    for i, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if args.smote:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)

        print("Fold:", i, "of ", args.n_fold * args.n_repeats, "(", len(y_train), "training, ", len(y_test), "testing)")

        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train), torch.Tensor(y_train[:, None]))
        test_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_test), torch.Tensor(y_test[:, None]))

        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        model = SimpleResNet(input_dim=X.shape[-1], depth=4, layer_size=64, regression=regression)
        trainer = Trainer(
            max_epochs=100,
            accelerator='gpu',
            devices=[0],
            callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=5)],
            enable_checkpointing=False,
            check_val_every_n_epoch=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )
        trainer.fit(model, train_loader, test_loader)

        model.eval()
        y_out = model(torch.Tensor(X_test)).detach().numpy().squeeze()
        if regression:
            r2 += r2_score(y_test, y_out) / (args.n_fold * args.n_repeats)
            rmse += mean_squared_error(y_test, y_out, squared=False) / (args.n_fold * args.n_repeats)
        else:
            y_pred = np.round(y_out)

            mcc += matthews_corrcoef(y_test, y_pred) / (args.n_fold * args.n_repeats)
            auc += roc_auc_score(y_test, y_out) / (args.n_fold * args.n_repeats)
            cm += confusion_matrix(y_test, y_pred, normalize='true') / (args.n_fold * args.n_repeats)

    if regression:
        print("Overall R2:", r2)
        print("Overall RMSE:", rmse)
    else:
        print("Overall AUC:", auc)
        print("Overall MCC:", mcc)
        plt.figure()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BBBP-', 'BBBP+'])
        disp.plot(cmap='Blues')
        plt.savefig(args.name + '_cm.png')