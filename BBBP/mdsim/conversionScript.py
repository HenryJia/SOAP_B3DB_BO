import pandas as pd
import os
from bs4 import BeautifulSoup
from xml.dom import minidom
import mechanize

# USAGE
# - Make an account at https://cgenff.umaryland.edu/userRegistration/
# - Create a .csv file containing the names/ID and SMILES of the molecules of interest (in the first and second columns)
# - Ensure you have a the cgenff_charmm2gmx_py3.py script and your chosen forcefield in your working directory, updating the self.ff variable to match if needs be
# - Run the script ($ python conversionScript.py), entering your new account username, password and file paths as prompted

class data:
    def __init__(self):
        self.usrName = "USERNAMEHERE"
        self.curPwd = "PASSWORDHERE"
        self.inputPath = "data.csv"
        self.outputPath = input("Path to folder where output will be saved. Leave blank to save to current folder: ")
        self.ff = "charmm36-jul2021.ff"

userInput = data()

data = pd.read_csv(userInput.inputPath)
nameList = data.iloc[:,0].values
smilesList = list(data.iloc[:,1].values)

for i,smile in enumerate(smilesList):
    smilesList[i] = smile.strip()


print(smilesList)

br = mechanize.Browser()
br.set_handle_robots(False)   # ignore robots
br.set_handle_refresh(False)  # can sometimes hang without this
br.addheaders = [('User-agent', 'Firefox.')]
br.set_handle_redirect(mechanize.HTTPRedirectHandler)

tupleSet = set(zip(smilesList, nameList))
for item in tupleSet:
    url = "https://cgenff.paramchem.org/userAccount/userLogin.php"
    response = br.open(url)
    br.form = list(br.forms())[0]
    usrName = br.form.find_control("usrName")
    curPwd = br.form.find_control("curPwd")
    usrName.value = userInput.usrName
    curPwd.value = userInput.curPwd
    response = br.submit()

    smiles = item[0]
    name = item[1][:4]
    file = open(userInput.outputPath + name + '.smi',"w+")
    file.write("{} {}".format(smiles, name))
    file.close()

    print("obabel {} -h -O {} --gen3d".format(userInput.outputPath + name + '.smi', userInput.outputPath + name + '.mol2'))

    os.system("obabel {} -h -O {} --gen3d".format(userInput.outputPath + name + '.smi', userInput.outputPath + name + '.mol2'))

    print('after obabel')

    filename = userInput.outputPath + name + '.mol2'
    br.form = list(br.forms())[0]
    br.form.add_file(open(filename), 'text/plain', filename)
    response = br.submit()
    xml = response.read().strip()
    dom = minidom.parseString(xml)
    print(dom.getElementsByTagName('path'))
    path = dom.getElementsByTagName('path')[0]
    inputf = dom.getElementsByTagName('mol2')[0]
    outputf = dom.getElementsByTagName('output')[0]

    url = "https://cgenff.paramchem.org/initguess/filedownload.php?file={}/{}".format(path.firstChild.data, outputf.firstChild.data )
    response = br.open(url)
    topology = response.read()
    file = open(userInput.outputPath + name + '.str',"w+")
    file.write(topology.decode("utf-8") )
    file.close()

    os.system("python cgenff_charmm2gmx_py3_nx1.py {} {}.mol2 {}.str {}".format(name, userInput.outputPath + name, userInput.outputPath + name, userInput.ff))
    os.system("mv {}* {}".format(name, userInput.outputPath))
