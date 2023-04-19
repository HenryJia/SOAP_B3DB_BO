import argparse
import os
import shutil
import subprocess
from subprocess import Popen, PIPE
from distutils.dir_util import copy_tree
import multiprocessing as mp

import numpy as np

def run_command(cmd):
    print('Running command: ', cmd)
    p = Popen(cmd, shell=True)
    p.wait()


parser = argparse.ArgumentParser()
#parser.add_argument('--csv', type=str, help='CSV file containing the data')

parser.add_argument('--smiles', type=str, help='SMILES')

# IMPORTANT NOTE: All of our other files are hardcoded to use the name 'main' for the molecule/residue
# Using a different name will almost certainly break things
parser.add_argument('--mol_name', type=str, help='Molecule name', default='main')
parser.add_argument('--working_dir', type=str, help='Working directory')

parser.add_argument('--lam', type=int, help='Lambda value')

parser.add_argument('--ntomp', type=int, help='Number of OpenMP threads per MPI rank to start')
parser.add_argument('--ntmpi', type=int, help='Number of thread-MPI ranks to start')

parser.add_argument('--gen_mol2', action='store_true', help='Generate mol2 files')
parser.add_argument('--charmm2gmx', action='store_true', help='Run charmm2gmx to generate topology files')
parser.add_argument('--gen_mdp', action='store_true', help='Generate mdp files')

parser.add_argument('--solvate', action='store_true', help='Create the box and the solvation molecules')
parser.add_argument('--em_steep', action='store_true', help='Run em_steep')
parser.add_argument('--em_lbfgs', action='store_true', help='Run em_lbfgs')
parser.add_argument('--nvt', action='store_true', help='Run nvt equilibration')
parser.add_argument('--npt', action='store_true', help='Run npt equilibration')
parser.add_argument('--md', action='store_true', help='Run the production MD')

args = parser.parse_args()

#df = pd.read_csv(args.csv)

lam = list(range(32))
input_dir = os.path.join(args.working_dir, 'input_files')
output_dir = os.path.join(args.working_dir, 'output_files')
solvate_dir = os.path.join(output_dir, 'solvate')
lambda_dir = os.path.join(output_dir, str(args.lam))

em_steep_dir = os.path.join(lambda_dir, 'em_steep')
em_lbfgs_dir = os.path.join(lambda_dir, 'em_l-bfgs')
nvt_dir = os.path.join(lambda_dir, 'nvt')
npt_dir = os.path.join(lambda_dir, 'npt')
md_dir = os.path.join(lambda_dir, 'md')

# Create input directory if it doesn't exist
if not os.path.exists(input_dir):
    os.makedirs(input_dir)

# Create output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Copy the files we need to the output directory
copy_tree('./charmm36-jul2021.ff', os.path.join(input_dir, 'charmm36-jul2021.ff'))

# Generate mol2 files
if args.gen_mol2:
    # Write the SMILES to a file so we can feed it to obabel a bit later
    with open(os.path.join(input_dir, args.mol_name + '.smi'), 'w') as f:
        f.write(args.smiles)

    # Now we need to generate a mol2 file from the SMILES
    # For explanation of the syntax here, -ismi is the input format, -omol2 is the output format
    # -O is the output file, and --gen3d is the flag to generate 3D coordinates
    cmd = 'obabel -ismi \'' + os.path.join(input_dir, args.mol_name + '.smi') + '\' -omol2 -O \''
    cmd += os.path.join(input_dir, args.mol_name + '.mol2') + '\' --gen3d'
    run_command(cmd)

    # Now we need to edit the mol2 file's residue name to match the name of the molecule
    # By default, obabel names the residue '*****'
    # Now, we could do this in a pythonic way, but I'm lazy and I don't want to deal with that
    # So we'll just use sed
    # Extra note: The residue name MUST be in all CAPS
    cmd = 'sed -i \'s/\*\*\*\*\*/' + args.mol_name.upper() + '/g\' \'' + os.path.join(input_dir, args.mol_name + '.mol2') + '\''
    run_command(cmd)

    # After this, we need to use cgenff to generate a CHARMM force field parameter file
    # This bit has to be done online. Annoyingly, I can't do this on my local machine here
    # So I'll just link to the online version here in this comment for now: https://cgenff.umaryland.edu/

# After this, we need to use charmm2gmx to generate topology files
if args.charmm2gmx:
    # There's probably a better way to run a python script from another python script
    # But I'm lazy and I don't want to deal with that
    # So we'll just use subprocess again
    # Extra note: The residue name MUST be in all CAPS
    cmd = 'python ./cgenff_charmm2gmx_py3_nx2.py ' + args.mol_name.upper() + ' ' + os.path.join(input_dir, args.mol_name + '.mol2')
    cmd += ' ' + os.path.join(input_dir, args.mol_name + '.str') + ' ./charmm36-jul2021.ff'
    run_command(cmd)

    # Annoyingly, charmm2gmx just drops the output files in the current directory
    # So we need to move them to the input directory
    os.rename(args.mol_name + '.top', os.path.join(input_dir, args.mol_name + '.top'))
    os.rename(args.mol_name + '.prm', os.path.join(input_dir, args.mol_name + '.prm'))
    os.rename(args.mol_name + '.itp', os.path.join(input_dir, args.mol_name + '.itp'))
    os.rename(args.mol_name + '_ini.pdb', os.path.join(input_dir, args.mol_name + '_ini.pdb'))

# Generate mdp files
if args.gen_mdp:
    # We need to copy the mdp files to the lambda directory
    if not os.path.exists(os.path.join(output_dir, 'mdp')):
        os.makedirs(os.path.join(output_dir, 'mdp'))
    if not os.path.exists(os.path.join(output_dir, 'mdp', 'EM_steep')):
        os.makedirs(os.path.join(output_dir, 'mdp', 'EM_steep'))
    if not os.path.exists(os.path.join(output_dir, 'mdp', 'EM_l-bfgs')):
        os.makedirs(os.path.join(output_dir, 'mdp', 'EM_l-bfgs'))
    if not os.path.exists(os.path.join(output_dir, 'mdp', 'NVT')):
        os.makedirs(os.path.join(output_dir, 'mdp', 'NVT'))
    if not os.path.exists(os.path.join(output_dir, 'mdp', 'NPT')):
        os.makedirs(os.path.join(output_dir, 'mdp', 'NPT'))
    if not os.path.exists(os.path.join(output_dir, 'mdp', 'MD')):
        os.makedirs(os.path.join(output_dir, 'mdp', 'MD'))

    shutil.copy('./mdp/em_steep.mdp', os.path.join(output_dir, 'mdp', 'EM_steep', 'em_steep.mdp'))
    shutil.copy('./mdp/em_l-bfgs.mdp', os.path.join(output_dir, 'mdp', 'EM_l-bfgs', 'em_l-bfgs.mdp'))
    shutil.copy('./mdp/nvt.mdp', os.path.join(output_dir, 'mdp', 'NVT', 'nvt.mdp'))
    shutil.copy('./mdp/npt.mdp', os.path.join(output_dir, 'mdp', 'NPT', 'npt.mdp'))
    shutil.copy('./mdp/md.mdp', os.path.join(output_dir, 'mdp', 'MD', 'md.mdp'))

    # We need to run the perl script to generate the lambda mdp files
    # Note the perl scripy doesn't like it if we start our directory path with a './'
    fn = output_dir if output_dir[0] != '.' else output_dir[2:]
    cmd = 'perl write_mdp.pl ' + os.path.join(fn, 'mdp', 'EM_steep', 'em_steep.mdp')
    run_command(cmd)
    cmd = 'perl write_mdp.pl ' + os.path.join(fn, 'mdp', 'EM_l-bfgs', 'em_l-bfgs.mdp')
    run_command(cmd)
    cmd = 'perl write_mdp.pl ' + os.path.join(fn, 'mdp', 'NVT', 'nvt.mdp')
    run_command(cmd)
    cmd = 'perl write_mdp.pl ' + os.path.join(fn, 'mdp', 'NPT', 'npt.mdp')
    run_command(cmd)
    cmd = 'perl write_mdp.pl ' + os.path.join(fn, 'mdp', 'MD', 'md.mdp')
    run_command(cmd)


# We are now ready to start running GROMACS

# First, we need to solvate the molecule with a solvent in a box, as well as add ions to neutralize the system
if args.solvate:
    # This generates a box around the molecule
    # We need to do this before we can run energy minimization
    # First, make the directory for it
    if not os.path.exists(solvate_dir):
        os.makedirs(solvate_dir)

    # Run editconf
    # I'm not sure what value to use for -d, so I'm just going to use 1.2 for now.
    # This is the same value used in the GROMACS tutorial for free energy of solvation for ethanol
    cmd = 'gmx editconf -f ' + os.path.join(input_dir, args.mol_name + '_ini.pdb') + ' -o '
    cmd += os.path.join(solvate_dir, args.mol_name + '_box.pdb') + ' -bt cubic -d 1.2'
    print('Running command: ', cmd)
    p = Popen(cmd, shell=True)
    p.wait()

    # Run solvate
    # NOTE: GROMACS will overwrite the topology file with the new one. But it will back it up
    cmd = 'gmx solvate -cp ' + os.path.join(solvate_dir, args.mol_name + '_box.pdb') + ' -cs -o '
    cmd += os.path.join(solvate_dir, args.mol_name + '_solvate.pdb') + ' -p ' + os.path.join(input_dir, args.mol_name + '.top')
    print('Running command: ', cmd)
    p = Popen(cmd, shell=True)
    p.wait()

    # Run grompp to setup genion and to generate the tpr file
    cmd = 'gmx grompp -f ./mdp/ions.mdp -c ' + os.path.join(solvate_dir, args.mol_name + '_solvate.pdb') + ' -p '
    cmd += os.path.join(input_dir, args.mol_name + '.top') + ' -o ' + os.path.join(solvate_dir, args.mol_name + '_ions.tpr')
    print('Running command: ', cmd)
    p = Popen(cmd, shell=True)
    p.wait()

    # IMPORTANT: The CHARMM force field does not use the standard ion names from the periodic table
    # Instead, it uses SOD for sodium and CLA for chlorine. We need to use these names here
    # Note: -conc 0.1 is the default value used PX923 and http://www.mdtutorials.com/gmx/umbrella/03_solvation.html
    # Also note: -neutral is used to neutralize the system by adding the appropriate number of ions
    # This part is also interactive. So it cannot be run in parallel or headless for now
    cmd = 'gmx genion -s ' + os.path.join(solvate_dir, args.mol_name + '_ions.tpr') + ' -o '
    cmd += os.path.join(solvate_dir, args.mol_name + '_ions.pdb') + ' -p ' + os.path.join(input_dir, args.mol_name + '.top')
    cmd += ' -pname SOD -nname CLA -neutral -conc 0.1'
    print('Running command: ', cmd)
    p = Popen(cmd, shell=True)
    p.wait()

if args.em_steep:
    # Create directories for the lambda values
    if not os.path.exists(lambda_dir):
        os.makedirs(lambda_dir)

    # Prepare to run the energy minimization 1: steep
    # First, make the directory for it
    if not os.path.exists(em_steep_dir):
        os.makedirs(em_steep_dir)

    # Run grompp
    cmd = 'gmx grompp -f ' + os.path.join(output_dir, 'mdp', 'EM_steep', 'em_steep_' + str(args.lam) + '.mdp')
    cmd += ' -c ' + os.path.join(solvate_dir, args.mol_name + '_ions.pdb')
    cmd += ' -p ' + os.path.join(input_dir, args.mol_name + '.top')
    cmd += ' -o ' + os.path.join(em_steep_dir, args.mol_name + '.tpr')
    cmd += ' -maxwarn 1' # Everyone seems to use this, even though I'm personally hesistant when it comes ignoring warnings
    run_command(cmd)

    # Run mdrun
    # Note, we need to set -deffnm to the name of the tpr file without the extension
    cmd = 'gmx mdrun -v -deffnm ' + os.path.join(em_steep_dir, args.mol_name) + ' -pin on'
    run_command(cmd)

if args.em_lbfgs:
    # Prepare to run the energy minimization 2: l-bfgs
    # First, make the directory for it
    if not os.path.exists(em_lbfgs_dir):
        os.makedirs(em_lbfgs_dir)

    # Run grompp
    # Note: Our input gro file is the output gro file from the previous step
    cmd = 'gmx grompp -f ' + os.path.join(output_dir, 'mdp', 'EM_l-bfgs', 'em_l-bfgs_' + str(args.lam) +'.mdp')
    cmd += ' -c ' + os.path.join(em_steep_dir, args.mol_name + '.gro')
    cmd += ' -p ' + os.path.join(input_dir, args.mol_name + '.top')
    cmd += ' -o ' + os.path.join(em_lbfgs_dir, args.mol_name + '.tpr')
    cmd += ' -maxwarn 1' # Everyone seems to use this, even though I'm personally hesistant when it comes ignoring warnings
    run_command(cmd)

    # Run mdrun
    # Note, we need to set -deffnm to the name of the tpr file without the extension
    # Also note, lbfgs cannot be run in parallel, so we need to set -nt to 1
    cmd = 'gmx mdrun -nt 1 -v -deffnm ' + os.path.join(em_lbfgs_dir, args.mol_name) + ' -pin on'
    run_command(cmd)

if args.nvt:
    # Prepare to run the NVT equilibration
    # First, make the directory for it
    if not os.path.exists(nvt_dir):
        os.makedirs(nvt_dir)

    # Run grompp
    # Note: Our input gro file is the output gro file from the previous step
    cmd = 'gmx grompp -f ' + os.path.join(output_dir, 'mdp', 'NVT', 'nvt_' + str(args.lam) + '.mdp')
    cmd += ' -c ' + os.path.join(em_lbfgs_dir, args.mol_name + '.gro')
    cmd += ' -p ' + os.path.join(input_dir, args.mol_name + '.top')
    cmd += ' -o ' + os.path.join(nvt_dir, args.mol_name + '.tpr')
    run_command(cmd)

    # Run mdrun
    # Note, we need to set -deffnm to the name of the tpr file without the extension
    cmd = 'gmx mdrun -v -ntomp ' + str(args.ntomp) + ' -ntmpi ' + str(args.ntmpi) + ' -deffnm ' + os.path.join(nvt_dir, args.mol_name) + ' -pin on'
    run_command(cmd)

if args.npt:
    # Prepare to run the NPT equilibration
    # First, make the directory for it
    if not os.path.exists(npt_dir):
        os.makedirs(npt_dir)

    # Run grompp
    # Note: Our input gro file is the output gro file from the previous step
    cmd = 'gmx grompp -f ' + os.path.join(output_dir, 'mdp', 'NPT', 'npt_' + str(args.lam) + '.mdp')
    cmd += ' -c ' + os.path.join(nvt_dir, args.mol_name + '.gro')
    cmd += ' -p ' + os.path.join(input_dir, args.mol_name + '.top')
    cmd += ' -t ' + os.path.join(nvt_dir, args.mol_name + '.cpt')
    cmd += ' -o ' + os.path.join(npt_dir, args.mol_name + '.tpr')
    run_command(cmd)

    # Run mdrun
    # Note, we need to set -deffnm to the name of the tpr file without the extension
    cmd = 'gmx mdrun -v -ntomp ' + str(args.ntomp) + ' -ntmpi ' + str(args.ntmpi) + ' -deffnm ' + os.path.join(npt_dir, args.mol_name) + ' -pin on'
    run_command(cmd)

if args.md:
    # Prepare to run the production run
    # First, make the directory for it
    if not os.path.exists(md_dir):
        os.makedirs(md_dir)

    # Run grompp
    # Note: Our input gro file is the output gro file from the previous step
    cmd = 'gmx grompp -f ' + os.path.join(output_dir, 'mdp', 'MD', 'md_' + str(args.lam) + '.mdp')
    cmd += ' -c ' + os.path.join(npt_dir, args.mol_name + '.gro')
    cmd += ' -p ' + os.path.join(input_dir, args.mol_name + '.top')
    cmd += ' -t ' + os.path.join(npt_dir, args.mol_name + '.cpt')
    cmd += ' -o ' + os.path.join(md_dir, args.mol_name + '.tpr')
    run_command(cmd)

    # Run mdrun
    # Note, we need to set -deffnm to the name of the tpr file without the extension
    cmd = 'gmx mdrun -v -ntomp ' + str(args.ntomp) + ' -ntmpi ' + str(args.ntmpi) + ' -deffnm ' + os.path.join(md_dir, args.mol_name) + ' -pin on'
    run_command(cmd)