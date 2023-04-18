import sys

# Change the moltype to current molecule 
def update_moltype(mdp_text,moltype):
    for i in range(len(mdp_text)):
        if 'couple-moltype' in mdp_text[i]:
            i_moltype=i
            moltype_line_list=list(mdp_text[i])
            break
    i_delete=moltype_line_list.index('=')+1
    n_delete=len(moltype_line_list)-moltype_line_list.index('=')
    del moltype_line_list[i_delete:i_delete+n_delete]
    moltype_line_list.append(' '+moltype)
    mdp_text[i_moltype]=''.join(moltype_line_list)
    return mdp_text

# Write text to file
def write_file(filename,filetext):
    with open(filename,'w') as f:
        for line in filetext:
            f.write("%s\n"%line)
    return None


def Main(filename,moltype,path):
    mdp_file=filename
    with open(mdp_file,'r') as f:
        mdp_old=f.read().splitlines()
    mdp_new=update_moltype(mdp_old,moltype)
    outfile=path+filename
    write_file(outfile,mdp_new)
    print('wrote mdp file')


if __name__ == '__main__':

    filename=sys.argv[1]
    moltype=sys.argv[2]
    path=sys.argv[3]
    Main(filename,moltype,path)
