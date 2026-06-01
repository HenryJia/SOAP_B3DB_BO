import multiprocessing as mp
import warnings
warnings.simplefilter("ignore") # Ignore warnings for now


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

from modules import MLP, SimpleResNetBlock, SimpleResNet

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
    def get_model_score(df, train_index, val_index, test_index):
        X, y = np.stack(df['SOAP'], axis=0), df['Class'].to_numpy()
        X = X.astype(np.float32)

        X_train = X[train_index]
        X_val = X[val_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_val = y[val_index]
        y_test = y[test_index]

        clf = RandomForestClassifier(
            n_estimators=100, max_depth=7, random_state=0)
        clf.fit(X[train_idx], y[train_idx])

        #pred_train = clf.predict(X[train_idx])
        #pred_test = clf.predict(X[test_idx])

        #mcc_train = matthews_corrcoef(y[train_idx], pred_train)
        #mcc_test = matthews_corrcoef(y[test_idx], pred_test)

        pred_train = clf.predict_proba(X_train)[:, 1]
        pred_val = clf.predict_proba(X_val)[:, 1]
        pred_test = clf.predict_proba(X_test)[:, 1]

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