# Get molecule files for md simulations into proper format and folders
#!/bin/bash

# source /usr/local/gromacs/bin/GMXRC

# parameters that may need to be changed
folder="molfiles"  # folder to store all sim files of each molecule for avon/sulis
path_files="molfiles2/"  # path to where molecule str, pdb, prm, etc files are located
data="data.csv"   # text file containing list of all molecule names (to use as residue names also)
ff="charmm36-jul2021.ff"    # name of forcefield folder 


# mdp file names
prod_mdp="md.mdp"
npt_mdp="npt.mdp"
em1_mdp="em_steep.mdp"
em2_mdp="em_l-bfgs.mdp"
nvt_mdp="nvt.mdp"


# extensions of molecule files that are needed for simulation
str='.str'
pdb='.pdb'
prm='.prm'
itp='.itp'
mdp='.mdp'
top='.top'


mkdir -p $folder 


N_mols=0
exec < $data
# read header     # comment back in if you have a header in data.csv file
while read mol
do

  echo "Preparing $mol files"
  mlwr=$(echo $mol | tr '[:upper:]' '[:lower:]')

  path_mol="$folder/$mol/"   # path to store sim files (mdp,topology,etc) of molecule 
  path_molfiles="$folder/$mol/$mol/"  # path to store topology and param files of molecule 
  mkdir -p $path_molfiles
  
  # copy ff/topology/param files for molecule into folder for use in md sim
  cp $path_files$mol$str $path_molfiles$mol$str
  cp $path_files$mlwr"_ini"$pdb $path_molfiles$mol$pdb
  cp $path_files$mlwr$itp $path_molfiles$mlwr$itp
  cp $path_files$mlwr$prm $path_molfiles$mlwr$prm
  cp $path_files$mlwr$top $path_molfiles$mol$top
  cp -r $ff $path_molfiles

  # update residue names in molecule top/param files 
  python update_residue.py $mol $path_molfiles
  
  # change moltype in mdp files
  python update_mdp_moltype.py $em1_mdp $mol $path_mol
  python update_mdp_moltype.py $em2_mdp $mol $path_mol
  python update_mdp_moltype.py $nvt_mdp $mol $path_mol
  python update_mdp_moltype.py $npt_mdp $mol $path_mol
  python update_mdp_moltype.py $prod_mdp $mol $path_mol

  # prepare MDP files for all lambda values 
  ./makemdps.sh $mol "$folder/$mol"

  ((N_mols+=1))

done

echo "Number of compound files prepared: $N_mols"

