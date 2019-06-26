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
DefaultStdout = sys.stdout

# Estimate causal effect
def causal_effect_estimate(data, treatment, outcome, common_causes):
    try:
        model= CausalModel(data=data,treatment=treatment,outcome=outcome,common_causes=common_causes)
        identified_estimand = model.identify_effect()
        estimate = model.estimate_effect(identified_estimand, method_name="backdoor.linear_regression")
        value = estimate.value
        return value
    except:
        return None

# p-value of causal effect by permutation test
def get_p_value(e0, data, treatment, outcome, common_causes, times=300):
    p = 1
    random_data = copy.deepcopy(data)
    trycount = 0
    for i in range(times-1):
        random_data.loc[:, treatment] = random.sample(list(random_data.loc[:, treatment]), len(random_data.index))
        random_data.loc[:, outcome] = random.sample(list(random_data.loc[:, outcome]), len(random_data.index))
        random_effect = causal_effect_estimate(random_data, treatment, outcome, common_causes)
        if random_effect == None or random_effect > e0:
            p += 1
    p = p/float(times)
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

# get confounder of treatment and target / ancestors
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
def causal_inference_from_prior(data, pattern):
    e_matrix = np.zeros(shape=(len(data.columns), len(data.columns))) # effect matrix
    p_matrix = np.zeros(shape=(len(data.columns), len(data.columns))) # p-value matrix
    # create prior microbial network
    network = prior_network(pattern)
    # estimate causal effect and p-value for microbial pairs one by one
    for i in range(len(data.columns)):
        for j in range(len(data.columns)):
            if i == j:
                continue
            treatment = data.columns[i]
            outcome = data.columns[j]
            common_causes = confounder(network, i, j) # cofounders
            effect = causal_effect_estimate(data, treatment, outcome, common_causes)
            e_matrix[j][i] = effect if effect != None else 0
            p_matrix[j][i] = get_p_value(e0, data, treatment, outcome, common_causes, times=times) if effect != None else 1
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
    return pd.read_table(path, index_col=0).T

# load prior network calculated by SparCC
def load_proir_network(path, threshold=0.01):
    sparcc = pd.read_table(path, index_col=0)
    pattern = (np.array(sparcc)<threshold)*1.0
    return pattern

# main run of causal inference
def run_causal_inference(data, prior, output_path='Result/', steps = 50, learn_rate=0.2,
                         times=100, min_threshold=0.05, max_threshold=0.3, simi_threshold=0.85):
    for step in range(steps):
        inter_indicator = bernoulli.rvs(prior)
        e_matrix, p_matrix = causal_inference_from_prior(data, inter_indicator)
        # output
        pd.DataFrame(e_matrix, columns=Species, index=Species).to_csv(output_path+dataname+'_ematrix_'+str(step)+'.csv')
        pd.DataFrame(p_matrix, columns=Species, index=Species).to_csv(output_path+dataname+'_pmatrix_'+str(step)+'.csv')
        pd.DataFrame(prior, columns=Species, index=Species).to_csv(output_path+dataname+'_prior_'+str(step)+'.csv')
        simi_score = network_similarity(1-p_matrix, inter_indicator)
        prior += learn_rate*(p_matrix<=min_threshold)*np.random.uniform(low=0, high=1, size=(prior.shape[0], prior.shape[1]))
        prior -= learn_rate*(p_matrix>=max_threshold)*np.random.uniform(low=0, high=1, size=(prior.shape[0], prior.shape[1]))
        prior = prior.clip(0, 1)
        if simi_score >= simi_threshold:
            break
    return True

if __name__ == '__main__':
    path_abundace = 'Data/NASH.txt'
    path_SparCC = 'Data/NASH_ra.txt'
    output_path='Result/'
    steps = 100
    learn_rate = 0.2
    times = 1001 # permutation test
    min_threshold = 0.05
    max_threshold = 0.3
    simi_threshold = 0.85
    # load abudance data and SparCC prior network
    data = load_abundance(path_abundace)
    prior = load_proir_network(path_SparCC, threshold=0.01)
    # run causal inference
    run_causal_inference(data, prior, output_path, steps, learn_rate,
                         times, min_threshold, max_threshold, simi_threshold)
