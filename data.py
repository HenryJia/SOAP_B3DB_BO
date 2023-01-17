import pandas as pd
import numpy as np
from quippy import descriptors
import ase
import sys
import subprocess
from pathlib import Path
import os
import pickle as pkl
from random import sample, shuffle, choice, choices
import numpy as np
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler
import tensorflow
from tensorflow.keras import layers, optimizers, Model, backend
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from scipy.stats import pearsonr
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from ase.geometry.analysis import Analysis

class SOAPDataset(object):
    def __init__(self, df, xyz_path, parameter_string) -> None:
        self.df = df
        self.xyz_path = xyz_path
        self.parameter_string = parameter_string

        mol = []
        for idx, row in df.iterrows():
            mol.append(ase.io.read(
                os.path.join(self.xyz_path, row['Name'] + '.xyz')))
        df['mol'] = mol

        soaps = []
        for idx, row in df.iterrows():
            soaps.append(descriptors.Descriptor(
                parameter_string).calc(row['Mol'])['data'][0])
        df['SOAP'] = soaps

    def __getitem__(self, index):
        return self.df.iloc[index]['SOAP'], self.df.iloc[index]['Class']

    def __len__(self):
        return len(self.df)