from functools import partial

import numpy as np

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import roc_auc_score, matthews_corrcoef, r2_score, confusion_matrix
from sklearn.model_selection import KFold, train_test_split

import quippy
from quippy import descriptors

def soap_worker(mol, soap_str):
    return descriptors.Descriptor(soap_str).calc(mol)['data'][0]

def train_worker(train_idx, test_idx, X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    model = RandomForestClassifier(
        n_estimators=100, max_depth=7, random_state=0)
    model.fit(X_train, y_train)

    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    mcc_train = matthews_corrcoef(y_train, pred_train)
    mcc_test = matthews_corrcoef(y_test, pred_test)

    pred_train = model.predict_proba(X_train)[:, 1]
    pred_test = model.predict_proba(X_test)[:, 1]

    auc_train = roc_auc_score(y_train, pred_train)
    auc_test = roc_auc_score(y_test, pred_test)

    confusion_train = confusion_matrix(y_train, model.predict(X_train), normalize='true')
    confusion_test = confusion_matrix(y_test, model.predict(X_test), normalize='true')

    return {
        'train_mcc': mcc_train, 'test_mcc': mcc_test,
        'train_auc': auc_train, 'test_auc': auc_test,
        'train_confusion': confusion_train, 'test_confusion': confusion_test}

class RFModel:
    def __init__(
        self, df, repeats, nfolds,
        Z, neighbours, nu_R, nu_S,
        pool):

        self.df = df
        self.repeats = repeats
        self.nfolds = nfolds

        self.Z = Z # These are the atomic species for the soap descriptor
        self.neighbours = neighbours
        self.nu_R = nu_R
        self.nu_S = nu_S

        self.pool = pool

    def make_soap_string(self, cutoff, l_max, n_max, sigma):
        Z_str = '{' + ', '.join([str(z) for z in self.Z]) + '}'
        neighbours_str = '{' + ', '.join([str(n) for n in self.neighbours]) + '}'

        cutoff = round(cutoff)
        l_max = round(l_max)
        n_max = round(n_max)

        soap_str = f"soap average cutoff={cutoff} l_max={l_max} "
        soap_str += f"n_max={n_max} atom_sigma={sigma} "
        soap_str += f"n_Z={len(self.Z)} Z={Z_str} n_species={len(self.neighbours)} species_Z={neighbours_str} "
        soap_str += f"nu_R={self.nu_R} nu_S={self.nu_S}"

        return soap_str

    def repeated_kfold(self, df):
        X = np.stack(df['SOAP'], axis=0)
        y = df['BBB+/BBB-'].to_numpy()

        train_idxs = []
        test_idxs = []
        for repeat in range(self.repeats):
            kf = KFold(n_splits=self.nfolds, shuffle=True, random_state=repeat)
            for train_idx, test_idx in kf.split(X):
                train_idxs.append(train_idx)
                test_idxs.append(test_idx)

        scores = self.pool.starmap(
            partial(train_worker, X=X, y=y),
            zip(train_idxs, test_idxs)
        )

        train_mcc = np.mean([s['train_mcc'] for s in scores])
        test_mcc = np.mean([s['test_mcc'] for s in scores])
        train_auc = np.mean([s['train_auc'] for s in scores])
        test_auc = np.mean([s['test_auc'] for s in scores])
        train_confusion = np.stack([s['train_confusion'] for s in scores]).mean(axis=0)
        test_confusion = np.stack([s['test_confusion'] for s in scores]).mean(axis=0)

        return {'train_mcc': train_mcc, 'test_mcc': test_mcc, 'train_auc': train_auc, 'test_auc': test_auc, 'train_confusion': train_confusion, 'test_confusion': test_confusion}

    def evaluate(self, cutoff, l_max, n_max, sigma):
        soap_str = self.make_soap_string(cutoff, l_max, n_max, sigma)

        soaps = self.pool.map(
            partial(soap_worker, soap_str=soap_str),
            self.df['Mol'].to_list()
        )

        df = self.df.copy()
        df['SOAP'] = soaps

        scores = self.repeated_kfold(df)

        return scores

    # This is the function we'll feed to the Bayesian Optimiser. It will take in all the params we want to optimise
    # It should return a single score that BayesOpt will try to maximise 
    def fmax(self, cutoff, l_max, n_max, sigma):
        scores = self.evaluate(cutoff, l_max, n_max, sigma)

        # Maximising test MCC is sufficient
        return scores['test_mcc']