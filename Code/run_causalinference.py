#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 10:48:53 2019

@author: dingfengwu
"""
# using python3.6 dowhy
import os
import sys
import random
import copy
import math
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.metrics import roc_auc_score
from scipy.stats import norm, bernoulli, pearsonr
from scipy.spatial import distance
from sklearn import linear_model
import dowhy
from dowhy.do_why import CausalModel

# Estimate causal effect
def causal_effect_estimate(data, treatment, outcome, common_causes):
    try:
        model= CausalModel(data=data,treatment=treatment,outcome=outcome,common_causes=common_causes)
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        value = estimate.value
        expr = estimate.realized_estimand_expr
        return value, expr
    except:
        return None, None

# p-value of causal effect by permutation test
# 直接计算线性回归用于计算加速
def get_p_value(e0, data, treatment, outcome, expr, times=300):
    features = data[expr.split('~')[1].split('+')]
    null_estimates = np.zeros(times)
    outcome = data[outcome]
    model = linear_model.LinearRegression()
    for i in range(times):
        outcome = np.random.permutation(outcome)
        model.fit(features, outcome)
        coefficients = model.coef_
        null_estimates[i] = coefficients[0]
    p = (null_estimates>=e0).sum()/float(times)
    return p if p<0.5 else 1-p

# create prior network
def prior_network(pattern):
    network=nx.DiGraph()
    links = []
    for i in range(len(pattern)):
        for j in range(len(pattern[0])):
            if i!=j and pattern[i, j]==1:
                links.append((j, i))
    network.add_edges_from(links)
    return network

# confounder of treatment and target / ancestors
def confounder(network, i, j):
    try:
        causes_node1 = nx.ancestors(network, i)
        causes_node2 = nx.ancestors(network, j)
        common_causes = list(causes_node1.intersection(causes_node2))
        common_causes = [data.columns[c] for c in common_causes]
    except:
        common_causes = []
    return common_causes

# causal inference
def causal_inference_from_prior(data, pattern, times):
    e_matrix = np.zeros(shape=(len(data.columns), len(data.columns))) # effect matrix
    p_matrix = np.zeros(shape=(len(data.columns), len(data.columns))) # p-value matrix
    # create prior microbial network
    network = prior_network(pattern)
    # estimate causal effect and p-value for microbial pairs one by one
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            if i == j:
                e_matrix[j][i] = 1.
                p_matrix[j][i] = 1.
                continue
            treatment = data.columns[i]
            outcome = data.columns[j]
            common_causes = confounder(network, i, j) # cofounders
            effect, expr = causal_effect_estimate(data, treatment, outcome, common_causes)
            e_matrix[j][i] = effect if effect != None else 0
            p_matrix[j][i] = get_p_value(effect, data, treatment, outcome, expr, times) if effect != None else 1
    return e_matrix, p_matrix

# network structrue similarity
def network_similarity(e_matrix, parameter):
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

# load abundance data
def load_abundance(path):
    return pd.read_csv(path, index_col=0, sep='\t').T

# load prior network calculated by SparCC
def load_proir_network(path, threshold=0.01):
    sparcc = pd.read_csv(path, index_col=0, sep='\t')
    pattern = (np.array(sparcc)<threshold)*1.0
    return pattern

# main run of causal inference
def run_causal_inference(data, prior, name, log_f, output_path='Result/', steps = 50, learn_rate=0.2,
                         times=100, p_threshold=0.01, simi_threshold=0.85):
    log = open(output_path+log_f, 'w')
    prior_distance = 0.0
    for step in range(steps):
        inter_indicator = bernoulli.rvs(prior)
        e_matrix, p_matrix = causal_inference_from_prior(data, inter_indicator, times)
        # output
        pd.DataFrame(e_matrix, columns=data.columns, index=data.columns).to_csv(output_path+name+'_ematrix_'+str(step)+'.csv')
        pd.DataFrame(p_matrix, columns=data.columns, index=data.columns).to_csv(output_path+name+'_pmatrix_'+str(step)+'.csv')
        pd.DataFrame(prior, columns=data.columns, index=data.columns).to_csv(output_path+name+'_prior_'+str(step)+'.csv')
        simi_score = network_similarity(1-p_matrix, inter_indicator)
        log.write(str(step)+'\t'+str(prior_distance)+'\t'+str(simi_score)+'\t'+str((prior>0.5).sum())+'\t'+str((prior>0.5).sum()/float(prior.shape[0]*prior.shape[1]))+'\n')
        log.flush()
        prior_new = copy.deepcopy(prior)
        w = learn_rate*math.sqrt(1-simi_score)*np.random.uniform(low=0, high=1, size=prior.shape)
        prior_new += w*(p_matrix<=p_threshold)*(np.random.uniform(low=0, high=1, size=prior.shape)<times*p_threshold)
        prior_new -= w*(p_matrix>p_threshold)*times*p_threshold
        prior_new = prior_new.clip(0, 1)
        prior_distance = distance.euclidean(np.array(prior).reshape(prior.shape[0]*prior.shape[1]), np.array(prior_new).reshape(prior.shape[0]*prior.shape[1]))
        if simi_score >= simi_threshold:
            break
        prior = prior_new
    log.close()
    return e_matrix, p_matrix, prior


DATA_PATH = '/Users/dingfengwu/Desktop/NAFLD3/Data/RohitNC_97OTU_assign_60/'
SPARCC_PATH = '/Users/dingfengwu/Desktop/NAFLD3/SparCC/RohitNC_97OTU_assign_60/'
OUT_PATH = '/Users/dingfengwu/Desktop/NAFLD3/Causalinference/RohitNC_97OTU_assign_60/'
OUT_TEMP_PATH = '/Users/dingfengwu/Desktop/NAFLD3/Causalinference/RohitNC_97OTU_assign_60/Temp/'
P_THRESHOLD = 0.01
BOOTSTRAP_TIMES = 9 # 设置小值便于加速计算，在prior学习中增加对应倍率惩罚
MAX_STEPS = 500 # 取决于BOOTSTRAP_TIMES的大小，BOOTSTRAP_TIMES越小，MAX_STEP需要越大
LEARNING_RATE = 0.5
STRUCTURE_THRESHOLD = 0.99

for name in ['nash']:
    in_f = 'abun_df_'+name+'.csv'
    sparcc_f = 'counts_df_'+name+'_sparcc_p_one_sided.csv'
    out_f = 'abun_df_'+name+'_causal.csv'
    out_p_f = 'abun_df_'+name+'_causal_p_one_sided.csv'
    log_f = name+'_log.txt'
    data = load_abundance(DATA_PATH+in_f)
    prior = load_proir_network(SPARCC_PATH+sparcc_f, threshold=P_THRESHOLD)
    e, p, r = run_causal_inference(data, prior, name, log_f, OUT_TEMP_PATH, MAX_STEPS, LEARNING_RATE,
                         BOOTSTRAP_TIMES, P_THRESHOLD, STRUCTURE_THRESHOLD)
    pd.DataFrame(e, columns=data.columns, index=data.columns).to_csv(OUT_PATH+out_f)
    pd.DataFrame(1-r, columns=data.columns, index=data.columns).to_csv(OUT_PATH+out_p_f)

#####################
#name = 'Normal'
#in_f = 'abun_df_'+name+'.csv'
#sparcc_f = 'counts_df_'+name+'_sparcc_p_one_sided.csv'
#out_f = 'abun_df_'+name+'_causal.csv'
#out_p_f = 'abun_df_'+name+'_causal_p_one_sided.csv'
#log_f = name+'_log.txt'
#data = load_abundance(DATA_PATH+in_f)
#r = pd.read_csv(OUT_TEMP_PATH+name+'_prior_'+str(MAX_STEPS-1)+'.csv', index_col=0)
#####################

#### Final
out_f = 'abun_df_'+name+'_causal_final.csv'
out_p_f = 'abun_df_'+name+'_causal_p_one_sided_final.csv'
inter_indicator = (np.array(r)>=0.99)*1.0
e_matrix, p_matrix = causal_inference_from_prior(data, inter_indicator, 999)
e_matrix = pd.DataFrame(e_matrix, columns=data.columns, index=data.columns)
e_matrix.to_csv(OUT_PATH+out_f)
p_matrix = pd.DataFrame(p_matrix, columns=data.columns, index=data.columns)
p_matrix.to_csv(OUT_PATH+out_p_f)

















