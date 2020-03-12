# using python2.7
#python '/Users/dingfengwu/Desktop/NAFLD3/yonatanf-sparcc-3aff6141c3f1/SparCC.py' -h

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
    cmd = "python "+"'"+SPARCC_PATH+"PseudoPvals.py' '"+OUT_PATH+out_f+"' "+BS_PATH+"perm_cor_#.txt "+str(times)+" -o '"+OUT_PATH+out_p_f+"_one_sided.csv' -t one_sided"
    os.system(cmd)
    cmd = "python "+"'"+SPARCC_PATH+"PseudoPvals.py' '"+OUT_PATH+out_f+"' "+BS_PATH+"perm_cor_#.txt "+str(times)+" -o '"+OUT_PATH+out_p_f+"_two_sided.csv' -t two_sided"
    os.system(cmd)

def main(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_r_f, out_p_f, method, times):
    run_cor(SPARCC_PATH, DATA_PATH, OUT_PATH, in_f, out_r_f, method)
    run_bootsrap(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_r_f, out_p_f, method, times)


SPARCC_PATH = '/Users/dingfengwu/Desktop/NAFLD3/yonatanf-sparcc-3aff6141c3f1/'
DATA_PATH = '/Users/dingfengwu/Desktop/NAFLD3/Data/RohitNC_Feature_60/'
OUT_PATH = '/Users/dingfengwu/Desktop/NAFLD3/SparCC/RohitNC_Feature_60/'
BS_PATH = '/Users/dingfengwu/Desktop/NAFLD3/SparCC/RohitNC_Feature_60/bootstrap/'
BOOTSTRAP_TIMES = 999

#in_f = 'counts_df.csv' # counts_df.csv counts_df_normal.csv counts_df_obese.csv counts_df_nash.csv
#method = 'spearman' # spearman sparcc
#out_r_f = in_f.split('.csv')[0]+'_'+method+'.csv'
#out_p_f = in_f.split('.csv')[0]+'_'+method+'_p'
#main(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_r_f, out_p_f, method, BOOTSTRAP_TIMES)

# 'counts_df.csv', 'counts_df_normal.csv', 'counts_df_obese.csv', 'counts_df_nash.csv'
for method in ['sparcc', 'spearman']:
    for in_f in ['counts_df_normal.csv', 'counts_df_nash.csv']:
        out_r_f = in_f.split('.csv')[0]+'_'+method+'.csv'
        out_p_f = in_f.split('.csv')[0]+'_'+method+'_p'
        main(SPARCC_PATH, DATA_PATH, OUT_PATH, BS_PATH, in_f, out_r_f, out_p_f, method, BOOTSTRAP_TIMES)
