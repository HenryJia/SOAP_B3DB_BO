import argparse
import os
import shutil
import subprocess
from subprocess import Popen, PIPE
from distutils.dir_util import copy_tree
import multiprocessing as mp

import pandas as pd

# This is a little jank, running slurm jobs from python
# But in all honesty, I need to parse the csv dataset file, and I don't want to do that in bash
# So I'm just going to do it in python

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
parser = argparse.ArgumentParser(description='Array job via SLURM')
parser.add_argument('--input_csv', metavar='input', type=str, help='Path to input .csv file')
parser.add_argument('--start_idx', metavar='start', type=int, help='Index of first molecule to fetch')
parser.add_argument('--end_idx', metavar='end', type=int, help='Index of last molecule to fetch (inclusive)')
parser.add_argument('--working_dir', metavar='working', type=str, help='Path to working directory')

args = parser.parse_args()

# Read in data
data = pd.read_csv(args.input_csv)

for idx in range(args.start_idx, args.end_idx + 1):
    print('Running SLURM for molecule ', data['Name'].iloc[idx], ' SMILES: ', data['Smiles'].iloc[idx])

    working_dir = os.path.join(args.working_dir, str(data['Name'].iloc[idx]))

    # First, run gmx solvate to solvate the molecule and add ions
    # Yes, I know it's janky and bad practice to call a python script from a python script
    # But I'm lazy and I don't want to deal with cleaning up the solvate code for now
    # It's 3am and I'm tired. The directory structure is complicated and doing my head in
    cmd = "python run.py --mol_name 'main' --working_dir " + working_dir + " --solvate --gen_mdp"
    run_command(cmd)

    # Now we start setting up the SLURM jobs
    # Step 1: minimisation with steepest descent
    cmd = "sbatch steep.sbatch " + working_dir
    output = run_command(cmd)
    job_id = output.split(' ')[-1][:-1]

    # Step 2: minimisation with l-bfgs
    cmd = "sbatch --dependency=afterok:" + job_id + " lbfgs.sbatch " + working_dir
    output = run_command(cmd)
    job_id = output.split(' ')[-1][:-1]

    # Step 3: NVT equilibration
    cmd = "sbatch --dependency=afterok:" + job_id + " nvt.sbatch " + working_dir
    output = run_command(cmd)
    job_id = output.split(' ')[-1][:-1]

    # Step 4: NPT equilibration
    cmd = "sbatch --dependency=afterok:" + job_id + " npt.sbatch " + working_dir
    output = run_command(cmd)
    job_id = output.split(' ')[-1][:-1]

    # Step 5: Production run
    cmd = "sbatch --dependency=afterok:" + job_id + " md.sbatch " + working_dir
    output = run_command(cmd)[:-1]

    print('Finished running SLURM for molecule ', data['Name'].iloc[idx], ' SMILES: ', data['Smiles'].iloc[idx])
