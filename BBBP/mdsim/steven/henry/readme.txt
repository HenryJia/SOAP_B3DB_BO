To create sim files for use on avon/sulis


Simple version of instructions: put all your molecule structure files into the "molfiles2 folder" and then run "prepmolfiles.sh" and then skip to step 5 below. 



1. Change "path_files" in "prepmolfiles.sh" to path/folder where you have all molecule files (top,str,prm,etc), here I have my structure files in the "molfiles2" folder

2. Have all mdp files, data.csv file containing names of molecule (residue) and forcefield folder in the same folder as "prepmolfiles.sh"

3. Run "prepmolfiles.sh" to gather the molecule files and create mdp files needed

4. Move molfiles folder containing molecule files into a new folder (here it is avonexample as an example) 

5. Adjust "array_job.sh" for the number of jobs you will be running, this number is equal to number of molecules times number of lambda values, e.g., 12 molecules with 32 lambdas each equals 384 jobs so you write "#SBATCH --array=0-383" in the sbatch script to launch those jobs which will run over 8 nodes on avon. You will likely need more than one of these array_job.sh files depending on the number of molecules you have. So to run the next 12 molecules, you would just make a copy of that script and change "0-383" to "384-767"

6. "array_job.sh" executes "job.sh" which contains all of the gromacs commands 

7. Move job scripts "job.sh" and "array_job.sh" and "data.csv" into that new folder 

8. Send that folder onto avon/sulis and launch jobs with "sbatch array_job.sh" 


Note: the "prepmdp.sh" makes only the MDP files, including it in case you may need it later on 
