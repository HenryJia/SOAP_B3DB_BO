import multiprocessing as mp
import warnings
warnings.simplefilter("ignore") # Ignore warnings for now
import argparse

import pandas as pd

import numpy as np
from sklearn.metrics import roc_auc_score, matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process.kernels import Matern

#from imblearn.over_sampling import SMOTE

from data import read_molecules

import matplotlib.pyplot as plt

from bayes_opt import BayesianOptimization, acquisition

from bo_soaps import RFModel


if __name__ == '__main__':

    # Note: This is absolutely necessary or quippy will get stuck for some reason
    # Idea from: https://github.com/isl-org/Open3D/issues/1552
    mp.set_start_method('forkserver')

    # Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=str, help='CSV file to use', required=True)
    parser.add_argument('--xyz', type=str, help='Location of the folder of xyz files',required=True)
    parser.add_argument('--workers', type=int, default=4, help='Number of workers to use for parallel processing')

    ga_params = parser.add_argument_group('Genetic Algorithm Parameters')
    ga_params.add_argument('--lower', type=int, help='Lower bound of the SOAP', default=2)
    ga_params.add_argument('--upper', type=int, help='Upper bound of the SOAP', default=10)
    ga_params.add_argument('--centres', type=list, nargs='+', help='Centres of the SOAP', default=[1, 6, 7, 8, 9, 16, 17, 35])
    ga_params.add_argument('--neighbours', type=list, nargs='+', help='Centres of the SOAP', default=[1, 6, 7, 8, 9, 16, 17, 35])
    ga_params.add_argument('--nu_R', type=int, help='nu_R of the SOAP', default=1)
    ga_params.add_argument('--nu_S', type=int, help='nu_S of the SOAP', default=0)
    ga_params.add_argument('--min_cutoff', type=float, help='Minimum cutoff of the SOAP', default=5)
    ga_params.add_argument('--max_cutoff', type=float, help='Maximum cutoff of the SOAP', default=20)
    ga_params.add_argument('--min_sigma', type=float, help='Minimum sigma of the SOAP', default=0.1)
    ga_params.add_argument('--max_sigma', type=float, help='Maximum sigma of the SOAP', default=1.5)

    args = parser.parse_args()
    print(args)

    # wandb.init(
    #     # set the wandb project where this run will be logged
    #     project="SOAP-BO",
    #     group=args.wandb_group,
    #     name=args.wandb_name if args.wandb_name else None,
    #     config = vars(args)
    # )

    print("Reading CSV")
    df = pd.read_csv(args.csv)
    print(df.head())

    print("Reading molecule xyz files")
    df['Mol'] = read_molecules(df, args.xyz, col='NO.')

    print(df.head())

    print("Instantiating the Random Forest model")
    pool = mp.Pool(args.workers, maxtasksperchild=100) # Something keeps leaking memory

    model = RFModel(
        df=df,
        repeats=5,
        nfolds=5,
        Z=args.centres,
        neighbours=args.neighbours,
        nu_R=args.nu_R,
        nu_S=args.nu_S,
        pool=pool
    )

    pbounds = {
        'cutoff': (args.min_cutoff, args.max_cutoff, int),
        'l_max': (args.lower, args.upper, int),
        'n_max': (args.lower, args.upper, int),
        'sigma': (args.min_sigma, args.max_sigma),
    }

    optimiser = BayesianOptimization(
        f=model.fmax,
        acquisition_function=acquisition.ExpectedImprovement(xi=0.01),
        pbounds=pbounds,
        verbose=2,
        random_state=0,
    )

    optimiser.set_gp_params(kernel=Matern(nu=2.5, length_scale=np.ones(len(pbounds))))

    print(f"Running Optimiser\n")
    optimiser.maximize(init_points=10, n_iter=50)

    print(f"Max: {optimiser.max['target']}\n\n")
