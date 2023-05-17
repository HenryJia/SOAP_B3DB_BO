import argparse
import os
import shutil
import subprocess
from subprocess import Popen, PIPE
from distutils.dir_util import copy_tree
import multiprocessing as mp

import pandas as pd

# This is a little jank, running some more commands from Python but hey, it works

def run_command(cmd):
    print('Running command: ', cmd)
    p = Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)
    output, err = p.communicate()
    output = output.decode('ascii')
    err = err.decode('ascii')
    print(output)
    print(err)
    return output

# Set up argparse
parser = argparse.ArgumentParser(description='Run gmx bar')
parser.add_argument('--input_csv', metavar='input', type=str, help='Path to input .csv file')
parser.add_argument('--start_idx', metavar='start', type=int, help='Index of first molecule to fetch')
parser.add_argument('--end_idx', metavar='end', type=int, help='Index of last molecule to fetch (inclusive)')
parser.add_argument('--working_dir', metavar='working', type=str, help='Path to working directory')

args = parser.parse_args()

# Read in data
data = pd.read_csv(args.input_csv)

for idx in range(args.start_idx, args.end_idx + 1):
    print('Running gmx bar for molecule ', data['Name'].iloc[idx], ' SMILES: ', data['Smiles'].iloc[idx])

    working_dir = os.path.join(args.working_dir, str(data['Name'].iloc[idx]))

    cmd = "gmx bar -f " + os.path.join(working_dir, 'output_files/*/md/main.xvg')
    cmd += " -o " + os.path.join(working_dir, 'output_files/bar.xvg')
    cmd += " -oi " + os.path.join(working_dir, 'output_files/barint.xvg')
    cmd += " -oh " + os.path.join(working_dir, 'output_files/barhist.xvg')
    run_command(cmd)