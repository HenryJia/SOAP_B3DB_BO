#!/bin/bash

# This script prepares gromacs mdp files for each lambda value

mol=$1
molpath=$2


mkdir -p $molpath/MDP/EM
mkdir -p $molpath/MDP/NVT
mkdir -p $molpath/MDP/NPT
mkdir -p $molpath/MDP/MD


mv $molpath/em_steep.mdp $molpath/MDP/EM/em_steep.mdp
mv $molpath/em_l-bfgs.mdp $molpath/MDP/EM/em_l-bfgs.mdp
mv $molpath/nvt.mdp $molpath/MDP/NVT/nvt.mdp
mv $molpath/npt.mdp $molpath/MDP/NPT/npt.mdp
mv $molpath/md.mdp $molpath/MDP/MD/md.mdp


perl write_mdp.pl $molpath/MDP/EM/em_steep.mdp
perl write_mdp.pl $molpath/MDP/EM/em_l-bfgs.mdp
perl write_mdp.pl $molpath/MDP/NVT/nvt.mdp
perl write_mdp.pl $molpath/MDP/NPT/npt.mdp
perl write_mdp.pl $molpath/MDP/MD/md.mdp

