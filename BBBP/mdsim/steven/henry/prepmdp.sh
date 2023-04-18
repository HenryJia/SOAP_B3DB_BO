# create simulation files for md simulations
#!/bin/bash

# source /usr/local/gromacs/bin/GMXRC

data="data.csv"
folder="molfiles"


mkdir -p $folder


prod_mdp="md.mdp"
npt_mdp="npt.mdp"
em1_mdp="em_steep.mdp"
em2_mdp="em_l-bfgs.mdp"
nvt_mdp="nvt.mdp"


N_mols=0
exec < $data
while read mol
do

  echo "Preparing $mol MDP files"
  
  path_mol="$folder/$mol/"
  mkdir -p $path_mol

  # change moltype in mdp files and save mdp in correct path
  python update_mdp_moltype.py $em1_mdp $mol $path_mol
  python update_mdp_moltype.py $em2_mdp $mol $path_mol
  python update_mdp_moltype.py $nvt_mdp $mol $path_mol
  python update_mdp_moltype.py $npt_mdp $mol $path_mol
  python update_mdp_moltype.py $prod_mdp $mol $path_mol

  ./makemdps.sh $mol "$folder/$mol"

  ((N_mols+=1))
  
done

echo "Number of molecule mdp files prepared: $N_mols"

