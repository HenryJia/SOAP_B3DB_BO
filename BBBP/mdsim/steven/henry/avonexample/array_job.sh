#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1  # number of cores
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=3700
#SBATCH --array=0-63   
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your email here for updates 


module purge
module load GCC/8.3.0  OpenMPI/3.1.4
module load GROMACS/2020


export OMP_NUM_THREADS=1

# Use SLURM_ARRAY_TASK_ID to:
# (1) Get the molecule that production run will be done on
# (2) Get the lambda value

job_id=$SLURM_ARRAY_TASK_ID

# Calculate lambda value
LAMBDA=$((job_id % 32))

# Get molecule name for production run
i=$((job_id/32 + 1))
mol=$(sed -n "$i"p data.csv)

# run workflow for lambda value %a (gets all values defined above)
./job.sh $mol $LAMBDA 
