import multiprocessing as mp
import warnings
warnings.simplefilter("ignore") # Ignore warnings for now
import argparse

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
from pytorch_lightning import Trainer, callbacks

from imblearn.over_sampling import SMOTE

from data import read_molecules
from genetic_algorithm import Individual, Population, GeneParameters
from modules import MLP, SimpleResNetBlock, SimpleResNet

import matplotlib.pyplot as plt

class NNIndividual(Individual):
    def get_model_score(df, train_index, test_index):
        X, y = np.stack(df['SOAP'], axis=0), df['Class'].to_numpy()
        X = X.astype(np.float32)

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]

        sm = SMOTE(random_state=42)
        X_train, y_train = sm.fit_resample(X_train, y_train)

        train_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_train), torch.Tensor(y_train[:, None]))
        test_dataset = torch.utils.data.TensorDataset(
            torch.Tensor(X_test), torch.Tensor(y_test[:, None]))

        # Our dataset is small, so if we set num_workers > 0, we end up spending more time setting up the workers than we do actually training.
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

        model = SimpleResNet(input_dim=X.shape[-1], depth=8, layer_size=128)
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

        pred_train = model(torch.Tensor(X_train)).detach().numpy().squeeze()
        pred_test = model(torch.Tensor(X_test)).detach().numpy().squeeze()

        #mcc_train = matthews_corrcoef(y_train, pred_train > 0.5)
        #mcc_test = matthews_corrcoef(y_test, pred_test > 0.5)

        # Gabriele likes MCC but most other papers use AUC
        # So we will use AUC for comparison's sake
        auc_train = roc_auc_score(y_train, pred_train)
        auc_test = roc_auc_score(y_test, pred_test)

        train_cm = confusion_matrix(y_train, pred_train > 0.5, normalize='true')
        test_cm = confusion_matrix(y_test, pred_test > 0.5, normalize='true')

        return {
            'train_scores': auc_train, 'test_scores': auc_test,
            'train_tp': train_cm[1, 1], 'train_tn': train_cm[0, 0],
            'test_tp': test_cm[1, 1], 'test_tn': test_cm[0, 0]}

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
    import wandb

    # Note: This is absolutely necessary or quippy will get stuck for some reason
    # Idea from: https://github.com/isl-org/Open3D/issues/1552
    mp.set_start_method('forkserver')

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='CSV file to use', required=True)
    parser.add_argument('--xyz', type=str, help='Location of the folder of xyz files',required=True)
    parser.add_argument('--wandb_group', type=str, help='Wandb group to use', required=True)

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
    print(args)

    wandb.init(
        # set the wandb project where this run will be logged
        project="SOAP-GAS-TMS",
        group=args.wandb_group,
        config = vars(args)
    )

    df = pd.read_csv(args.csv)
    df['Name'] = df['Name'].astype(str)
    print(df.head())

    df['Mol'] = read_molecules(df, args.xyz)

    print(df.head())

    print('Making parameters')

    # This is a tad inefficient, but it's fine
    gene_args = {
        'lower': args.lower,
        'upper': args.upper,
        'centres': args.centres,
        'neighbours': args.neighbours,
        'nu_R': args.nu_R,
        'nu_S': args.nu_S,
        'mutation_chance': args.mutation_chance,
        'min_cutoff': args.min_cutoff,
        'max_cutoff': args.max_cutoff,
        'min_sigma': args.min_sigma,
        'max_sigma': args.max_sigma,
        'message_steps': args.message_steps
    }

    gene_parameters = [GeneParameters(**gene_args)]

    print('Making gene set')
    example_gene_set = [gene_parameters[0].make_gene_set()]

    '''
    # Some example code for making an individual

    print('Making individual')
    example_individual = NNIndividual(
        example_gene_set, df)

    print('Getting score')
    getter = example_individual.comp_soaps()
    example_individual.evaluate_model(getter)

    result_dict = example_individual.results_dictionary

    mcc = np.mean(result_dict['test_scores'])

    print('MCC: {}'.format(mcc))
    '''

    pop = Population(
        lambda gene_set: NNIndividual(
            gene_set, df),
        args.population_size, args.best_sample, args.lucky_few, args.num_children,
        gene_parameters, maximise_scores=True, verbose=True)

    pop.initialise_population()

    pop.print_population()

    history_score = [[ind.score for ind in pop.population]]
    history_auc = [[np.mean(ind.results_dictionary['test_scores']) for ind in pop.population]]
    for gen in range(1, args.num_generations+1):
        print(f"Generation {gen}")
        pop.next_generation()
        auc = []
        score = []
        ind_str = []
        train_tn = []
        train_tp = []
        test_tn = []
        test_tp = []
        for ind in pop.population:
            auc.append(np.mean(ind.results_dictionary['test_scores']))
            score.append(ind.score)
            ind_str.append(str(ind))
            train_tn.append(np.mean(ind.results_dictionary['train_tn']))
            train_tp.append(np.mean(ind.results_dictionary['train_tp']))
            test_tn.append(np.mean(ind.results_dictionary['test_tn']))
            test_tp.append(np.mean(ind.results_dictionary['test_tp']))

        history_score.append(score)
        history_auc.append(auc)

        print(f"Best {ind_str[np.argmax(auc)]} has a AUC and score of: {np.max(auc)}, {score[np.argmax(auc)]}")
        wandb.log({
            # 'SOAP String': ind_str,
            # 'Test AUC': auc,
            # 'Train AUC': score,
            # 'Train TN': train_tn,
            # 'Train TP': train_tp,
            # 'Test TN': test_tn,
            # 'Test TP': test_tp,
            # 'Best SOAP String': ind_str[np.argmax(auc)],
            'Best SOAP Test AUC': np.max(auc),
            'Best SOAP Train AUC': score[np.argmax(auc)],
            # 'Best SOAP Train TN': train_tn[np.argmax(auc)],
            # 'Best SOAP Train TP': train_tp[np.argmax(auc)],
            # 'Best SOAP Test TN': test_tn[np.argmax(auc)],
            # 'Best SOAP Test TP': test_tp[np.argmax(auc)],
            'Generation': gen
            })

    score_table = wandb.Table(data=[[i, s] for i, scores in enumerate(history_score) for s in scores], columns=["Generation", "Train AUC"])
    auc_table = wandb.Table(data=[[i, m] for i, aucs in enumerate(history_auc) for m in aucs], columns=["Generation", "Test AUC"])
    wandb.log({"Individual Train AUC": wandb.plot.scatter(score_table, "Generation", "Train AUC", title="Genetic Algorithm Train AUC")})
    wandb.log({"Individual Test AUC": wandb.plot.scatter(auc_table, "Generation", "Test AUC", title="Genetic Algorithm Test AUC")})
    print('Finished')
