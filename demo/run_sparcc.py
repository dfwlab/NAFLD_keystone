# using python2.7
# python './yonatanf-sparcc-3aff6141c3f1/SparCC.py' -h

import os

def run_cor(SPARCC_PATH, DATA_PATH, OUT_PATH, in_f, out_f, method):
    cmd = "python "+"'"+SPARCC_PATH+"SparCC.py' '"+DATA_PATH+in_f+"' --cor_file='"+OUT_PATH+out_f+"' -a "+method
    os.system(cmd)

def run_bootsrap(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_f, out_p_f, method, times=1000):
    cmd = "python "+"'"+SPARCC_PATH+"MakeBootstraps.py' '"+DATA_PATH+in_f+"' -n "+str(times)+" -t permutation_#.txt -p "+BS_PATH
    os.system(cmd)
    bs_data_path = BS_PATH
    bs_out_path = BS_PATH
    for i in range(times):
        bs_in_f = 'permutation_'+str(i)+'.txt'
        bs_out_f = 'perm_cor_'+str(i)+'.txt'
        run_cor(SPARCC_PATH, bs_data_path, bs_out_path, bs_in_f, bs_out_f, method)
    cmd = "python "+"'"+SPARCC_PATH+"PseudoPvals.py' '"+OUT_PATH+out_f+"' "+BS_PATH+"perm_cor_#.txt "+str(times)+" -o '"+OUT_PATH+out_p_f+"_one_sided.tsv' -t one_sided"
    os.system(cmd)

def main(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_r_f, out_p_f, method, times):
    run_cor(SPARCC_PATH, DATA_PATH, OUT_PATH, in_f, out_r_f, method)
    run_bootsrap(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_r_f, out_p_f, method, times)


    
SPARCC_PATH = './yonatanf-sparcc-3aff6141c3f1/'
DATA_PATH = './Demo/'
OUT_PATH = './'
method = 'sparcc'
BOOTSTRAP_TIMES = 999

# Contorl
BS_PATH = './control_sparcc_bs/'
in_f = 'AGP_control_count_1.tsv'
out_r_f = 'AGP_control_corr.tsv' 
out_p_f = 'AGP_control_p'
main(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_r_f, out_p_f, method, BOOTSTRAP_TIMES)

# Cancer
BS_PATH = './cancer_sparcc_bs/'
in_f = 'AGP_cancer_count_1.tsv'
out_r_f = 'AGP_cancer_corr.tsv' 
out_p_f = 'AGP_cancer_p'
main(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_r_f, out_p_f, method, BOOTSTRAP_TIMES)