#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32 # We are wasting a small amount of resources here, Avon has 48 cores per node
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00 # The production MD might take a while to run...
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
    # Note we need to specify --exact or Slurm will think we are asking for 32 tasks even though we explicitly specify --ntasks=1
    # This took me too bloody long to figure out
    # Slurm is a twat sometimes
    srun --ntasks=1 --exact pipenv run python run.py --smiles "CCC(=O)N(C1CCN(CC1)CCc2ccccc2)c3ccccc3" --mol_name 'main' --working_dir 'fentanyl/' --md --lam $lam --ntomp 1 --ntmpi 1 &> logs/md_$lam.log &
done
wait
