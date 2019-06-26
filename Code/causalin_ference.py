#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 13:22:09 2018

@author: dingfengwu
"""
import sys
import random
import copy
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, bernoulli, pearsonr
import dowhy
from dowhy.do_why import CausalModel
import matplotlib.pyplot as plt
DefaultStdout = sys.stdout

def getPvalue(v0, data, treatment, outcome, common_causes, t=300):
    p = 1
    rdata = copy.deepcopy(data)
    trycount = 0
    for i in range(t-1):
        try:
            rdata.loc[:, treatment] = random.sample(list(rdata.loc[:, treatment]), len(rdata.index))
            rdata.loc[:, outcome] = random.sample(list(rdata.loc[:, outcome]), len(rdata.index))
            model= CausalModel(data=rdata,treatment=treatment,outcome=outcome,common_causes=common_causes)
            identified_estimand = model.identify_effect()
            estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
            value = estimate.value
            if value >= v0:
                p+=1
        except:
            p+=1
            trycount += 1
    p = p/float(t)
    if trycount>0.5*t:
        return 1.0
    return p if p<0.5 else 1-p
    

def getCausalRelated_from_pattern2(step, aucm, data, pattern, rt=100, islog=False):
    if not islog:
        file = open('logger.log', 'w')
        sys.stdout = file
    e_matrix = np.zeros(shape=(len(data.columns), len(data.columns)))
    p_matrix = np.zeros(shape=(len(data.columns), len(data.columns)))
    DG=nx.DiGraph()
    links = []
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            if i!=j and pattern[i, j]==1:
                links.append((j, i))
    DG.add_edges_from(links)
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            if i == j:
                continue
            treatment = data.columns[i]
            outcome = data.columns[j]
            
            try:
                causes_node1 = nx.ancestors(DG, i)
                causes_node2 = nx.ancestors(DG, j)
                common_causes = list(causes_node1.intersection(causes_node2))
                common_causes = [data.columns[c] for c in common_causes]
            except:
                common_causes = []
            #common_causes = [data.columns[c] for c in range(len(data.columns)) if pattern[i][c]==1 and pattern[j][c]==1]
            try:
                model= CausalModel(data=data,treatment=treatment,outcome=outcome,common_causes=common_causes)
                identified_estimand = model.identify_effect()
                estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
                e_matrix[j][i] = estimate.value
                if e_matrix[j][i]==0:
                    p_matrix[j][i] = 1
                else:
                    p_matrix[j][i] = getPvalue(estimate.value, data, treatment, outcome, common_causes, t=100)
            except:
                e_matrix[j][i] = 0
                p_matrix[j][i] = 1
            if not islog:
                sys.stdout = DefaultStdout
                print(step, aucm, 'causal', i, j, e_matrix[j][i], p_matrix[j][i])
                #print(common_causes)
                sys.stdout = file
    if not islog:
        sys.stdout = DefaultStdout
        file.close()
    return e_matrix, p_matrix

def getCausalRelated_from_pattern(step, aucm, data, pattern, rt=100, islog=False):
    e_matrix = np.zeros(shape=(len(data.columns), len(data.columns)))
    p_matrix = np.zeros(shape=(len(data.columns), len(data.columns)))
    DG=nx.DiGraph()
    links = []
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            if i!=j and pattern[i, j]==1:
                links.append((j, i))
    DG.add_edges_from(links)
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            if i == j:
                continue
            treatment = data.columns[i]
            outcome = data.columns[j]
            
            try:
                causes_node1 = nx.ancestors(DG, i)
                causes_node2 = nx.ancestors(DG, j)
                common_causes = list(causes_node1.intersection(causes_node2))
                common_causes = [data.columns[c] for c in common_causes]
            except:
                common_causes = []
            #common_causes = [data.columns[c] for c in range(len(data.columns)) if pattern[i][c]==1 and pattern[j][c]==1]
            try:
                model= CausalModel(data=data,treatment=treatment,outcome=outcome,common_causes=common_causes)
                identified_estimand = model.identify_effect()
                estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
                e_matrix[j][i] = estimate.value
                if e_matrix[j][i]==0:
                    p_matrix[j][i] = 1
                else:
                    p_matrix[j][i] = getPvalue(estimate.value, data, treatment, outcome, common_causes, t=100)
            except:
                e_matrix[j][i] = 0
                p_matrix[j][i] = 1
    return e_matrix, p_matrix

def data_check(data):
    sel_columns = []
    for c in data.median().index:
        if data.median()[c]>0:
            sel_columns.append(c)
    return data.loc[:, sel_columns]

def get_auc(e_matrix, parameter):
    parameter = np.array(parameter)
    e_matrix = np.array(e_matrix)
    plist = []
    for i in range(len(parameter)):
        for j in range(len(parameter[0])):
            if i==j:
                continue
            plist.append(parameter[i][j]!=0)
    elist = []
    for i in range(len(e_matrix)):
        for j in range(len(e_matrix[0])):
            if i==j:
                continue
            elist.append(e_matrix[i][j])
    auc = roc_auc_score(plist, elist)
    return auc


def getRelated(data, islog=False):
    e_matrix = np.zeros(shape=(len(data.columns), len(data.columns)))
    p_matrix = np.zeros(shape=(len(data.columns), len(data.columns)))
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            if i == j:
                continue
            valuei = data.iloc[:, i]
            valuej = data.iloc[:, j]
            r, p = pearsonr(valuei, valuej)
            e_matrix[j][i] = r
            p_matrix[j][i] = p
            if not islog:
                print('relation', i, j, e_matrix[j][i], p_matrix[j][i])
    if not islog:
        sys.stdout = DefaultStdout
    return e_matrix, p_matrix

if __name__ == '__main__':
    dataname = 'NASH'
    data = pd.read_table('Data/'+dataname+'.txt', index_col=0)
    species2 = set(data.index)
    print(len(species2))
    Species = sorted(species2)
    
    #####
    dataname = 'NASH'
    relate_data = pd.read_table('Data/'+dataname+'_ra.txt', index_col=0).T
    relate_data = relate_data.loc[:, Species]

    sparcc = pd.read_table('Data/'+dataname+'_p.txt', index_col=0)
    sparcc = sparcc.loc[Species, Species]
    print(np.array(sparcc))

    pattern = (np.array(sparcc)<0.01)*1.0
    print(pattern)
    aucm = 0.1
    learn = 0.2
    f = open('Result/'+dataname+'_log.txt', 'w')
    rt=100
    for step in range(55):
        p_sel = bernoulli.rvs(pattern)
        new_e_matrix, new_p_matrix = getCausalRelated_from_pattern(step, aucm, relate_data, p_sel, rt=rt, islog=False)
        print(pattern)
        pd.DataFrame(new_e_matrix, columns=Species, index=Species).to_csv('Result/'+dataname+'_common_ematrix_step'+str(step)+'.csv')
        pd.DataFrame(new_p_matrix, columns=Species, index=Species).to_csv('Result/'+dataname+'_common_pmatrix_step'+str(step)+'.csv')
        pd.DataFrame(pattern, columns=Species, index=Species).to_csv('Result/'+dataname+'_common_pattern_step'+str(step)+'.csv')
        new_aucm = get_auc(1-new_p_matrix, p_sel)
        #if random.random()<0.5*(new_aucm/aucm):
        e_matrix = new_e_matrix
        p_matrix = new_p_matrix
        aucm = new_aucm
        pattern += learn*(p_matrix<=0.05)*np.random.uniform(low=0, high=1, size=(pattern.shape[0], pattern.shape[1]))
        pattern -= learn*(p_matrix>=0.3)*np.random.uniform(low=0, high=1, size=(pattern.shape[0], pattern.shape[1]))
        pattern = pattern.clip(0, 1)
        f.write(str(step)+'\t'+str(aucm)+'\n')
        f.flush()
    f.close()
