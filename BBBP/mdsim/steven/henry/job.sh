#!/bin/bash

# if not already set, set number of CORES default value (MPI only, dont use openmp threading)
CORES=1
echo "Will use $CORES threads for jobs (except l-bfgs in serial)"

# name of molecule
mol=$1

#calculate this lambda state value only
LAMBDA=$2

# Go into directory for current molecule
cd molfiles/$mol

FREE_ENERGY=$(pwd)
echo "Free energy home directory set to $FREE_ENERGY"

MDP=$FREE_ENERGY/MDP
echo ".mdp files are stored in $MDP"

# Create and go into directory for current lambda state
mkdir -p L$LAMBDA
cd L$LAMBDA
echo "The current directory is " `pwd`


#################################
# ENERGY MINIMIZATION 1: STEEP  #
#################################

echo "Starting steep energy minimization for lambda = $LAMBDA..."

mkdir -p EM_1
cd EM_1
echo "The current directory is " `pwd`

gmx grompp -f $MDP/EM/em_steep_$LAMBDA.mdp -c $FREE_ENERGY/$mol/$mol.gro -p $FREE_ENERGY/$mol/$mol.top -o min$LAMBDA.tpr >& log
gmx mdrun -nt $CORES -ntomp 1 -deffnm min$LAMBDA -pin on -v >& logg

sleep 1


################################
# ENERGY MINIMIZATION 2: L-BFGS #
#################################

echo "Starting L-BFGS Minimization ..."

cd ../
mkdir -p EM_2
cd EM_2
echo "The current directory is " `pwd`

# We use -maxwarn 1 here because grompp incorrectly complains about use of a plain cutoff; this is a minor issue
# that will be fixed in a future version of Gromacs
gmx grompp -f $MDP/EM/em_l-bfgs_$LAMBDA.mdp -c ../EM_1/min$LAMBDA.gro -p ../../$mol/$mol.top -o min$LAMBDA.tpr -maxwarn 1 >& log

# Run L-BFGS in serial (cannot be run in parallel)
gmx mdrun -nt 1 -deffnm min$LAMBDA -pin on -v >& logg

echo "L-BFGS Minimization complete."

sleep 1

#####################
# NVT EQUILIBRATION #
#####################

echo "Starting constant volume equilibration ..."

cd ../
mkdir -p NVT
cd NVT
echo "The current directory is " `pwd`

gmx grompp -f $MDP/NVT/nvt_$LAMBDA.mdp -c ../EM_2/min$LAMBDA.gro -p ../../$mol/$mol.top -o nvt$LAMBDA.tpr >& log
gmx mdrun -nt $CORES -ntomp 1 -deffnm nvt$LAMBDA -pin on -v >& logg

echo "Constant volume equilibration complete."

sleep 1

#####################
# NPT EQUILIBRATION #
#####################

echo "Starting constant pressure equilibration..."

cd ../
mkdir -p NPT
cd NPT
echo "The current directory is " `pwd`

gmx grompp -f $MDP/NPT/npt_$LAMBDA.mdp -c ../NVT/nvt$LAMBDA.gro -p ../../$mol/$mol.top -t ../NVT/nvt$LAMBDA.cpt -o npt$LAMBDA.tpr >& log
gmx mdrun -nt $CORES -ntomp 1 -deffnm npt$LAMBDA -pin on -v >& logg

echo "Constant pressure equilibration complete."

sleep 1

#################
# PRODUCTION MD #
#################
echo "Starting production MD simulation..."

cd ../
mkdir -p MD
cd MD
echo "The current directory is " `pwd`

gmx grompp -f $MDP/MD/md_$LAMBDA.mdp -c ../NPT/npt$LAMBDA.gro -p ../../$mol/$mol.top -t ../NPT/npt$LAMBDA.cpt -o md$LAMBDA.tpr >& log
gmx mdrun -nt $CORES -ntomp 1 -deffnm md$LAMBDA -pin on -v >& logg

echo "Production MD complete."

# End
echo "Ending. Job completed for lambda = $LAMBDA"


