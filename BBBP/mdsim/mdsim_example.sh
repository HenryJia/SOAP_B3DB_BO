python -m pipenv shell

#python run.py --smiles "CCC(=O)N(C1CCN(CC1)CCc2ccccc2)c3ccccc3" --mol_name 'main' --working_dir 'fentanyl/' --solvate
python run.py --smiles "CCC(=O)N(C1CCN(CC1)CCc2ccccc2)c3ccccc3" --mol_name 'main' --working_dir 'fentanyl/' --gen_mdp --em_steep --em_lbfgs --npt --nvt --md --lam 0 --ntomp 48 --ntmpi 1 &> mdsim_example.log