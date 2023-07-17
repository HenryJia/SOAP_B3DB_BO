# Run just a model on the SOAPs without any genetic algorithm
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import matthews_corrcoef, roc_auc_score
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

from data import read_molecules, soap_worker
from genetic_algorithm import Individual, Population, GeneParameters
from modules import MLP, SimpleResNetBlock, SimpleResNet

import matplotlib.pyplot as plt
import plotly.express as px

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
    parser.add_argument('--umap', help='use umap', default=False, action='store_true')
    parser.add_argument('--tsne', help='use tsne', default=False, action='store_true')
    parser.add_argument('--n_estimators', help='number of estimators for random forest', default=100, type=int)
    parser.add_argument('--max_depth', help='maximum depth of random forest', default=None, type=int)
    parser.add_argument('--verbose', help='verbosity of random forest', default=0, type=int)
    parser.add_argument('--name', help='name of the experiment', default='test', type=str)

    soap_args = parser.add_argument_group('SOAP arguments')
    soap_args.add_argument('--l_max', help='maximum angular momentum', default=6, type=int)
    soap_args.add_argument('--n_max', help='maximum radial order', default=6, type=int)
    soap_args.add_argument('--cutoff', help='cutoff radius', default=5, type=int)
    soap_args.add_argument('--atom_sigma', help='sigma for atom type', default=0.5, type=float)
    soap_args.add_argument('--centres', type=list, nargs='+', help='Centres of the SOAP', default=[1, 6, 7, 8, 9, 16, 17, 35])
    soap_args.add_argument('--neighbours', type=list, nargs='+', help='Number of neighbours to consider', default=[1, 6, 7, 8, 9, 16, 17, 35])
    soap_args.add_argument('--nu_R', type=int, default=1)
    soap_args.add_argument('--nu_S', type=int, default=0)

    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    df['Name'] = df['Name'].astype(str)
    print(df.head())

    df['Mol'] = read_molecules(df, args.xyz)

    print(df.head())

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
    df['SOAP'] = soap_worker(df, [soap_string])

    X, y = np.stack(df['SOAP'].to_numpy(), axis=0), df['Class'].to_numpy()

    kf = RepeatedKFold(n_splits=args.n_fold, n_repeats=args.n_repeats, random_state=42)
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
        trainer.fit(model, train_loader, test_loader)

        model.eval()
        y_prob = model(torch.Tensor(X_test)).detach().numpy().squeeze()
        y_pred = np.round(y_prob)

        mcc += matthews_corrcoef(y_test, y_pred) / (args.n_fold * args.n_repeats)
        auc += roc_auc_score(y_test, y_prob) / (args.n_fold * args.n_repeats)
        cm += confusion_matrix(y_test, y_pred, normalize='true') / (args.n_fold * args.n_repeats)

    print("Overall AUC:", auc)
    print("Overall MCC:", mcc)
    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['BBBP-', 'BBBP+'])
    disp.plot(cmap='Blues')
    plt.savefig(args.name + '_cm.png')