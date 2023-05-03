#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32 # We are wasting a small amount of resources here, Avon has 48 cores per node
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00 # We do not need a lot of time for each of this step. Asking for less time on the queue means we get scheduled faster
#SBATCH --mem-per-cpu=3700 
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.jia.2@warwick.ac.uk

module purge
module load GCCcore/8.3.0 Python/3.7.4 git
module load GCC/8.3.0 OpenMPI/3.1.4
module load GROMACS/2020

mkdir -p logs

for lam in {0..31}
do
    srun pipenv run python run.py --smiles "CCC(=O)N(C1CCN(CC1)CCc2ccccc2)c3ccccc3" --mol_name 'main' --working_dir 'fentanyl/' --em_lbfgs --lam $lam --ntomp 1 --ntmpi 1 &> logs/em_lbfgs_$lam.log
done
wait