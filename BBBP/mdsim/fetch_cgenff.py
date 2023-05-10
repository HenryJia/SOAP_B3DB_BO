import argparse
import os
from subprocess import Popen, PIPE

import pandas as pd
from xml.dom import minidom
import mechanize

def run_command(cmd):
    print('Running command: ', cmd)
    p = Popen(cmd, shell=True)
    p.wait()


# - Make an account at https://cgenff.umaryland.edu/userRegistration/
# - Create a .csv file containing the names/ID and SMILES of the molecules of interest (in the first and second columns)
# - Ensure you have a the cgenff_charmm2gmx_py3.py script and your chosen forcefield in your working directory, updating the self.ff variable to match if needs be
# - Run the script ($ python conversionScript.py), entering your new account username, password and file paths as prompted

username = "HenryJia"
password = "94wda8qKXk8SGjP!"

# Set up argparse
parser = argparse.ArgumentParser(description='Fetch CGenFF parameters for molecules in a .csv file')
parser.add_argument('--input_csv', metavar='input', type=str, help='Path to input .csv file')
parser.add_argument('--output_dir', metavar='output', type=str, help='Path to output directory')
parser.add_argument('--start_idx', metavar='start', type=int, help='Index of first molecule to fetch')
parser.add_argument('--end_idx', metavar='end', type=int, help='Index of last molecule to fetch (inclusive)')

args = parser.parse_args()

# Read in data
data = pd.read_csv(args.input_csv)


if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
 
for idx in range(args.start_idx, args.end_idx + 1):
    print('Fetching ', data['Name'].iloc[idx], ' SMILES: ', data['Smiles'].iloc[idx])
    print('Generating Mol2 file...')
    output_dir = os.path.join(args.output_dir, str(data['Name'].iloc[idx]), 'input_files')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Write the SMILES to a file so we can feed it to obabel a bit later
    with open(os.path.join(output_dir, 'main.smi'), 'w') as f:
        f.write(data['Smiles'].iloc[idx])

    # Now we need to generate a mol2 file from the SMILES
    # For explanation of the syntax here, -ismi is the input format, -omol2 is the output format
    # -O is the output file, and --gen3d is the flag to generate 3D coordinates
    cmd = 'obabel -ismi \'' + os.path.join(output_dir, 'main.smi') + '\' -omol2 -O \''
    cmd += os.path.join(output_dir, 'main.mol2') + '\' --gen3d'
    run_command(cmd)

    # Now we need to edit the mol2 file's residue name to match the name of the molecule
    # By default, obabel names the residue '*****'
    # Now, we could do this in a pythonic way, but I'm lazy and I don't want to deal with that
    # So we'll just use sed
    # Extra note: The residue name MUST be in all CAPS
    cmd = 'sed -i \'s/\*\*\*\*\*/MAIN/g\' \'' + os.path.join(output_dir, 'main.mol2') + '\''
    run_command(cmd)
    print('Done generating mol2 file')

    print('Connecting to CGenFF...')
    br = mechanize.Browser()
    br.set_handle_robots(False)   # ignore robots
    br.set_handle_refresh(False)  # can sometimes hang without this
    br.addheaders = [('User-agent', 'Firefox.')]
    br.set_handle_redirect(mechanize.HTTPRedirectHandler)

    url = "https://cgenff.paramchem.org/userAccount/userLogin.php"
    response = br.open(url)
    br.form = list(br.forms())[0]
    usrName = br.form.find_control("usrName")
    curPwd = br.form.find_control("curPwd")
    usrName.value = username
    curPwd.value = password
    response = br.submit()

    br.form = list(br.forms())[0]
    br.form.add_file(open(os.path.join(output_dir, 'main.mol2')), 'text/plain', os.path.join(output_dir, 'main.mol2'))
    response = br.submit()
    xml = response.read().strip()
    dom = minidom.parseString(xml)
    print(dom.getElementsByTagName('path'))
    path = dom.getElementsByTagName('path')[0]
    inputf = dom.getElementsByTagName('mol2')[0]
    outputf = dom.getElementsByTagName('output')[0]

    url = "https://cgenff.paramchem.org/initguess/filedownload.php?file={}/{}".format(path.firstChild.data, outputf.firstChild.data)
    response = br.open(url)
    topology = response.read()
    file = open(os.path.join(output_dir, 'main.str'),"w+")
    file.write(topology.decode("utf-8"))
    file.close()

    print('Done generating .str file')

    print('Running charmm2gmx...')
    # There's probably a better way to run a python script from another python script
    # But I'm lazy and I don't want to deal with that
    # So we'll just use subprocess again
    # Extra note: The residue name MUST be in all CAPS
    cmd = 'python ./cgenff_charmm2gmx_py3_nx2.py MAIN ' + os.path.join(output_dir, 'main.mol2')
    cmd += ' ' + os.path.join(output_dir, 'main.str') + ' ./charmm36-jul2021.ff'
    run_command(cmd)

    # Annoyingly, charmm2gmx just drops the output files in the current directory
    # So we need to move them to the input directory
    os.rename('main.top', os.path.join(output_dir, 'main.top'))
    os.rename('main.prm', os.path.join(output_dir, 'main.prm'))
    os.rename('main.itp', os.path.join(output_dir, 'main.itp'))
    os.rename('main_ini.pdb', os.path.join(output_dir, 'main_ini.pdb'))
    print('Done generating .top, .prm, .itp and .pdb files\n')
