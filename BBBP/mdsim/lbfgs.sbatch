#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00 # We do not need a lot of time for each of this step. Asking for less time on the queue means we get scheduled faster
#SBATCH --mem-per-cpu=3850 
#SBATCH --account=su007-gcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.jia.2@warwick.ac.uk

module purge
module load GCCcore/8.3.0 Python/3.7.4 git
module load GCC/8.3.0 OpenMPI/3.1.4
module load GROMACS/2020

mkdir -p logs
mkdir -p logs/$1

for lam in {0..31}
do
    # Note we need to specify --exact or Slurm will think we are asking for 32 tasks even though we explicitly specify --ntasks=1
    # This took me too bloody long to figure out
    # Slurm is a twat sometimes
    srun --ntasks=1 --exact pipenv run python run.py --mol_name 'main' --working_dir "$1" --em_lbfgs --lam $lam --ntomp 1 --ntmpi 1 &> logs/$1/em_lbfgs_$lam.log &
done
wait
