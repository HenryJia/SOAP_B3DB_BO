import sys
import numpy as np

def update_itp_residue(itp_text):
    i_beg=itp_text.index('[ atoms ]')+3
    i_end=itp_text.index('[ bonds ]')
    residue=itp_text[i_beg].split()[3]
    n_delete=len(residue)-3
    for i in range(i_beg,i_end):
        line=itp_text[i]
        try:
            if line.split()[3]==residue:
                line_list=list(line)
                del line_list[28:28+n_delete]
                line=''.join(line_list)
                itp_text[i]=line
#                 print('changing residue')
        except:
            pass
    return itp_text

def update_str_residue(str_text):
    for i in range(len(str_text)):
        if 'RESI' in str_text[i]:
            i_res=i
            res_line_list=list(str_text[i])
            n_residue=len(str_text[i].split()[1])
            n_replace=n_residue-3
            break

    for i in range(8,8+n_replace):
        res_line_list[i]=' '

    str_text[i_res]=''.join(res_line_list)
    return str_text


def update_pdb_residue(pdb_text):
    residue=pdb_text[0].split()[3]
    # n_residue=len(residue)
    n_delete=len(residue)-3
    for i,line in enumerate(pdb_text):
        if 'ATOM' in line:
            line_list=list(line)
            del line_list[20:20+n_delete]
            pdb_text[i]=''.join(line_list)
    return pdb_text


# Write text to file
def write_file(filename,filetext):
    with open(filename,'w') as f:
        for line in filetext:
            f.write("%s\n"%line)
    return None


def Main(name,path):

    itp_file=path+'{}.itp'.format(name.lower())
    with open(itp_file,'r') as f:
        itp_old=f.read().splitlines()

    str_file=path+'{}.str'.format(name)
    with open(str_file,'r') as f:
        str_old=f.read().splitlines()

    pdb_file=path+'{}.pdb'.format(name)
    with open(pdb_file,'r') as f:
        pdb_old=f.read().splitlines()

    print('read itp, str and pdb files')

    itp_new=update_itp_residue(itp_old)
    str_new=update_str_residue(str_old)
    pdb_new=update_pdb_residue(pdb_old)


    print('created itp, str and pdb files')

    upper_name=name.upper()
    lower_name=name.lower()
    write_file(path+lower_name+'.itp',itp_new)
    write_file(path+upper_name+'.str',str_new)
    write_file(path+upper_name+'.pdb',pdb_new)


    print('wrote itp, str and pdb files')


if __name__ == '__main__':

    name=sys.argv[1]
    path=sys.argv[2]
    Main(name,path)
