#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # number of cores
#SBATCH --cpus-per-task=48 # number of cpus
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=3700 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.jia.2@warwick.ac.uk

module purge
module load GCCcore/8.3.0 Python/3.7.4 git
module load GCC/8.3.0 OpenMPI/3.1.4
module load GROMACS/2020

srun sh mdsim_example.sh