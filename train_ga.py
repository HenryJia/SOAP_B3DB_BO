# Run just a model on the SOAPs without any genetic algorithm
import os
import argparse

import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import matthews_corrcoef, roc_auc_score, r2_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import Trainer, callbacks

import ase
from genetic_algorithm import Individual, Population, GeneParameters
from modules import SimpleResNet

import wandb

import matplotlib.pyplot as plt
import plotly.express as px


class NNIndividual(Individual):
    def get_model_score(self, df, train_index, val_index, test_index):
        X = np.stack(df['SOAP'], axis=0).astype(np.float32)
        if self.dataset == 'moleculenet':
            y = df['p_np'].to_numpy()
        elif self.dataset == 'b3db':
            if self.regression:
                y = df['logBB'].to_numpy().astype(np.float32)
            else:
                y = (data.BBB == 'BBB+').to_numpy().astype(int)

        X_train = X[train_index]
        X_val = X[val_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_val = y[val_index]
        y_test = y[test_index]

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
            max_epochs=20,
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

        if self.regression:
            r2_train = r2_score(y_train, pred_train)
            r2_val = r2_score(y_val, pred_val)
            r2_test = r2_score(y_test, pred_test)
            rmse_train = mean_squared_error(y_train, pred_train, squared=False)
            rmse_val = mean_squared_error(y_val, pred_val, squared=False)
            rmse_test = mean_squared_error(y_test, pred_test, squared=False)

            return {
                'train_scores': r2_train, 'val_scores': r2_val, 'test_scores': r2_test,
                'train_rmse': rmse_train, 'val_rmse': rmse_val, 'test_rmse': rmse_test}
        else:
            #mcc_train = matthews_corrcoef(y_train, pred_train > 0.5)
            #mcc_test = matthews_corrcoef(y_test, pred_test > 0.5)

            # Gabriele likes MCC but most other papers use AUC
            # So we will use AUC for comparison's sake
            auc_train = roc_auc_score(y_train, pred_train)
            auc_val = roc_auc_score(y_val, pred_val)
            auc_test = roc_auc_score(y_test, pred_test)

            train_cm = confusion_matrix(y_train, np.round(pred_train), normalize='true')
            val_cm = confusion_matrix(y_val, np.round(pred_val), normalize='true')
            test_cm = confusion_matrix(y_test, np.round(pred_test), normalize='true')

            return {
                'train_scores': auc_train, 'val_scores': auc_val, 'test_scores': auc_test,
                'train_tp': train_cm[1, 1], 'train_tn': train_cm[0, 0],
                'val_tp': val_cm[1, 1], 'val_tn': val_cm[0, 0],
                'test_tp': test_cm[1, 1], 'test_tn': test_cm[0, 0]}

class RFIndividual(Individual):
    def get_model_score(self, df, train_index, val_index, test_index):
        X = np.stack(df['SOAP'], axis=0).astype(np.float32)
        if self.dataset == 'moleculenet':
            y = df['p_np'].to_numpy()
        elif self.dataset == 'b3db':
            if self.regression:
                y = df['logBB'].to_numpy().astype(np.float32)
            else:
                y = (data.BBB == 'BBB+').to_numpy().astype(int)

        X_train = X[train_index]
        X_val = X[val_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_val = y[val_index]
        y_test = y[test_index]

        if self.regression:
            model = RandomForestRegressor(
                n_estimators=100, max_depth=7, random_state=0)
        else:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=7, random_state=0)
        model.fit(X_train, y_train)

        if self.regression:
            pred_train = model.predict(X_train)
            pred_val = model.predict(X_val)
            pred_test = model.predict(X_test)

            r2_train = r2_score(y_train, pred_train)
            r2_val = r2_score(y_val, pred_val)
            r2_test = r2_score(y_test, pred_test)
            rmse_train = mean_squared_error(y_train, pred_train, squared=False)
            rmse_val = mean_squared_error(y_val, pred_val, squared=False)
            rmse_test = mean_squared_error(y_test, pred_test, squared=False)

            return {
                'train_scores': r2_train, 'val_scores': r2_val, 'test_scores': r2_test,
                'train_rmse': rmse_train, 'val_rmse': rmse_val, 'test_rmse': rmse_test}
        else:
            pred_train = model.predict_proba(X_train)[:, 1]
            pred_val = model.predict_proba(X_val)[:, 1]
            pred_test = model.predict_proba(X_test)[:, 1]

            # Gabriele likes MCC but most other papers use AUC
            # So we will use AUC for comparison's sake
            auc_train = roc_auc_score(y_train, pred_train)
            auc_val = roc_auc_score(y_val, pred_val)
            auc_test = roc_auc_score(y_test, pred_test)

            train_cm = confusion_matrix(y_train, np.round(pred_train), normalize='true')
            val_cm = confusion_matrix(y_val, np.round(pred_val), normalize='true')
            test_cm = confusion_matrix(y_test, np.round(pred_test), normalize='true')

            return {
                'train_scores': auc_train, 'val_scores': auc_val, 'test_scores': auc_test,
                'train_tp': train_cm[1, 1], 'train_tn': train_cm[0, 0],
                'val_tp': val_cm[1, 1], 'val_tn': val_cm[0, 0],
                'test_tp': test_cm[1, 1], 'test_tn': test_cm[0, 0]}


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
    parser.add_argument('--wandb_group', type=str, help='Wandb group to use', required=True)
    parser.add_argument('--wandb_name', type=str, help='Wandb name to use', required=False)

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

    wandb.init(
        # set the wandb project where this run will be logged
        project="SOAP-GAS-TMS",
        group=args.wandb_group,
        name=args.wandb_name if args.wandb_name else None,
        config = vars(args)
    )

    regression = False
    data = pd.read_csv(args.csv)

    if args.b3db: # Note that B3DB data is tab separated
        if 'regression' in args.csv:
            regression = True
            assert not args.smote, 'Cannot use SMOTE with regression, only classification'

    mol = []
    for idx, row in data.iterrows():
        num = row['num'] if args.moleculenet else row['NO.']
        mol.append(ase.io.read(os.path.join(args.xyz, str(num) + '.xyz')))
    data['Mol'] = mol

    print(data.head())

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

    pop = Population(
        lambda gene_set: RFIndividual( #NNIndividual(
            gene_set, data, regression=regression, dataset='moleculenet' if args.moleculenet else 'b3db'),
        args.population_size, args.best_sample, args.lucky_few, args.num_children,
        gene_parameters, maximise_scores=True, verbose=True)

    pop.initialise_population()
    pop.print_population()

    history_train = [[np.mean(ind.results_dictionary['train_scores']) for ind in pop.population]]
    history_val = [[np.mean(ind.results_dictionary['val_scores']) for ind in pop.population]]
    history_test = [[np.mean(ind.results_dictionary['test_scores']) for ind in pop.population]]
    for gen in range(1, args.num_generations+1):
        print(f"Generation {gen}")
        pop.next_generation()

        ind_str = []
        if regression:
            train_r2 = []
            val_r2 = []
            test_r2 = []
            train_rmse = []
            val_rmse = []
            test_rmse = []
            for ind in pop.population:
                train_r2.append(np.mean(ind.results_dictionary['train_scores']))
                val_r2.append(np.mean(ind.results_dictionary['val_scores']))
                test_r2.append(np.mean(ind.results_dictionary['test_scores']))
                train_rmse.append(np.mean(ind.results_dictionary['train_rmse']))
                val_rmse.append(np.mean(ind.results_dictionary['val_rmse']))
                test_rmse.append(np.mean(ind.results_dictionary['test_rmse']))

                ind_str.append(str(ind))

            history_train.append(train_r2)
            history_val.append(val_r2)
            history_test.append(test_r2)
        else:
            train_auc = []
            val_auc = []
            test_auc = []
            train_tn = []
            train_tp = []
            val_tn = []
            val_tp = []
            test_tn = []
            test_tp = []

            for ind in pop.population:
                train_auc.append(np.mean(ind.results_dictionary['train_scores']))
                val_auc.append(np.mean(ind.results_dictionary['val_scores']))
                test_auc.append(np.mean(ind.results_dictionary['test_scores']))

                train_tn.append(np.mean(ind.results_dictionary['train_tn']))
                train_tp.append(np.mean(ind.results_dictionary['train_tp']))
                test_tn.append(np.mean(ind.results_dictionary['test_tn']))
                test_tp.append(np.mean(ind.results_dictionary['test_tp']))

                ind_str.append(str(ind))

            history_train.append(train_auc)
            history_val.append(val_auc)
            history_test.append(test_auc)

            print(f"Best {ind_str[np.argmax(val_auc)]} has a train, val and test AUC and score of:",
                train_auc[np.argmax(val_auc)], np.max(val_auc), test_auc[np.argmax(val_auc)])
            wandb.log({
                'SOAP String': ind_str,
                'Test AUC': auc,
                'Train AUC': score,
                'Train TN': train_tn,
                'Train TP': train_tp,
                'Test TN': test_tn,
                'Test TP': test_tp,
                'Best SOAP String': ind_str[np.argmax(auc)],
                'Best SOAP Train AUC': train_auc[np.argmax(val_auc)],
                'Best SOAP Val AUC': np.max(val_auc),
                'Best SOAP Test AUC': test_auc[np.argmax(val_auc)],
                'Best SOAP Train TN': train_tn[np.argmax(auc)],
                'Best SOAP Train TP': train_tp[np.argmax(auc)],
                'Best SOAP Test TN': test_tn[np.argmax(auc)],
                'Best SOAP Test TP': test_tp[np.argmax(auc)],
                'Generation': gen
                })

    if regression:
        train_table = wandb.Table(data=[[i, s] for i, scores in enumerate(history_train) for s in scores], columns=["Generation", "Train R2"])
        val_table = wandb.Table(data=[[i, s] for i, scores in enumerate(history_val) for s in scores], columns=["Generation", "Validation R2"])
        test_table = wandb.Table(data=[[i, m] for i, aucs in enumerate(history_test) for m in aucs], columns=["Generation", "Test R2"])
        wandb.log({"Individual Train R2": wandb.plot.scatter(train_table, "Generation", "Train R2", title="Genetic Algorithm Train R2")})
        wandb.log({"Individual Validation R2": wandb.plot.scatter(val_table, "Generation", "Validation R2", title="Genetic Algorithm Validation R2")})
        wandb.log({"Individual Test R2": wandb.plot.scatter(test_table, "Generation", "Test R2", title="Genetic Algorithm Test R2")})
    else:
        train_table = wandb.Table(data=[[i, s] for i, scores in enumerate(history_train) for s in scores], columns=["Generation", "Train AUC"])
        val_table = wandb.Table(data=[[i, s] for i, scores in enumerate(history_val) for s in scores], columns=["Generation", "Validation AUC"])
        test_table = wandb.Table(data=[[i, m] for i, aucs in enumerate(history_test) for m in aucs], columns=["Generation", "Test AUC"])
        wandb.log({"Individual Train AUC": wandb.plot.scatter(train_table, "Generation", "Train AUC", title="Genetic Algorithm Train AUC")})
        wandb.log({"Individual Validation AUC": wandb.plot.scatter(val_table, "Generation", "Validation AUC", title="Genetic Algorithm Validation AUC")})
        wandb.log({"Individual Test AUC": wandb.plot.scatter(test_table, "Generation", "Test AUC", title="Genetic Algorithm Test AUC")})
    print('Finished')