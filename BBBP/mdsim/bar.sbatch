#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=00:30:00 # The gmx bar command is pretty fast
#SBATCH --mem-per-cpu=3850 
#SBATCH --account=su007-gcs
#SBATCH --mail-type=ALL
#SBATCH --mail-user=h.jia.2@warwick.ac.uk

module purge
module load GCCcore/8.3.0 Python/3.7.4 git
module load GCC/8.3.0 OpenMPI/3.1.4
module load GROMACS/2020

srun gmx bar -f $(ls -v $1/output_files/*/md/main.xvg) -o $1/output_files/bar.xvg -oi $1/output_files/barint.xvg -oh $1/output_files/barhist.xvg &> $1/output_files/bar.log
