#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 09:54:05 2018

@author: dingfengwu
"""
import re
import random
import copy
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import norm, bernoulli, pearsonr
from scipy.stats import ranksums
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import cloudpickle as pickle

def getRelatedData(dataname = 'Normal'):
    sdata = pd.read_csv('network/nodes.csv', index_col=0)
    Species = sorted(list(sdata.species))
    data = pd.read_table('SparCC/'+dataname+'/'+dataname+'_uf.txt', index_col=0).T
    data = data.loc[:, Species]
    relate_data = copy.deepcopy(data)
    ds = np.array([float(i) for i in data.sum(1)])
    for i in data.columns:
        relate_data[i] = relate_data[i]/np.array(ds)
    return relate_data

def getClass(name):
    k = re.search(r'k__([\w\W]*?)\.', name)
    k = k.group(1).strip() if k else ''
    p = re.search(r'p__([\w\W]*?)\.', name)
    p = p.group(1).strip() if p else ''
    c = re.search(r'c__([\w\W]*?)\.', name)
    c = c.group(1).strip() if c else ''
    o = re.search(r'o__([\w\W]*?)\.', name)
    o = o.group(1).strip() if o else ''
    f = re.search(r'f__([\w\W]*?)\.', name)
    f = f.group(1).strip() if f else ''
    g = re.search(r'g__([\w\W]*?)\.', name)
    g = g.group(1).strip() if g else ''
    return k, p, c, o, f, g

def getRankSumDiff():
    alpha = 0.01
    Normal = getRelatedData('Normal')
    NASH = getRelatedData('NASH')
    Species = list(Normal.columns)
    result = []
    ps = []
    index = 0
    for s in Species:
        k, p, c, o, f, g = getClass(s)
        nor_avg = Normal.loc[:, s].mean()
        nash_avg = NASH.loc[:, s].mean()
        diff = nash_avg-nor_avg
        score, pvalue = ranksums(Normal.loc[:, s], NASH.loc[:, s])
        dire = '1' if diff>0 else '0'
        result.append([index, s, k, p, c, o, f, g, nor_avg, nash_avg, diff, abs(diff), dire, score, pvalue])
        ps.append(pvalue)
        index += 1
    result = np.array(result)
    reject, fdr, _, _ = multipletests(ps, alpha=alpha, method='fdr_bh')
    result = pd.DataFrame(result, columns=['index', 's', 'k', 'p', 'c', 'o', 'f', 'g', 'nor_avg', 'nash_avg', 'diff', 'absdiff', 'diff_dire', 'ranksum', 'pvalue'])
    result['fdr'] = fdr
    result['log_fdr'] = -np.log10(fdr)
    result['diff_0.05'] = [result.loc[i, 'absdiff'] if result.loc[i, 'fdr']<0.05 else 0 for i in result.index]
    result['diff_0.01'] = [result.loc[i, 'absdiff'] if result.loc[i, 'fdr']<0.01 else 0 for i in result.index]
    return Species, result

def buildNetwork(Species, dataname = 'Normal'):
    if dataname == 'Normal':
        data = pd.read_csv('network/'+dataname+'_common_links_step48.csv', index_col=0)
    else:
        data = pd.read_csv('network/'+dataname+'_common_links_step50.csv', index_col=0)
    edges = []
    for i in data.index:
        source = int(data.loc[i, 'source'])
        target = int(data.loc[i, 'target'])
        effect = 1#abs(float(data.loc[i, 'effect']))
        edges.append((source, target, effect))
    DG=nx.DiGraph()
    DG.add_nodes_from(list(range(len(Species))))
    DG.add_weighted_edges_from(edges)        
    return DG

def getPageRank_Hub(net):
    h,a = nx.hits(net, max_iter=1000)
    pr = nx.pagerank(net, alpha=0.85)
#    pr = nx.effective_size(net)
#    pr = {n: v / net.degree(n) for n, v in pr.items()}
    return h, a, pr

def importance_diff(nornet, nashnet):
    Nor_h, Nor_a, Nor_p = getPageRank_Hub(nornet)
    NASH_h, NASH_a, NASH_p = getPageRank_Hub(nashnet)
    result = []
    for index in nornet.nodes():
        h_diff = (Nor_h.get(index, 0)-NASH_h.get(index, 0))
        a_diff = (Nor_a.get(index, 0)-NASH_a.get(index, 0))
        p_diff = (Nor_p.get(index, 0)-NASH_p.get(index, 0))
        result.append([index, Nor_h.get(index, 0), NASH_h.get(index, 0), Nor_a.get(index, 0), 
                       NASH_a.get(index, 0), Nor_p.get(index, 0), NASH_p.get(index, 0),
                       round(h_diff, 3), round(a_diff, 3), round(p_diff, 3)])
    #result = sorted(result, key=lambda x:abs(x[1]), reverse=True)
    result = pd.DataFrame(result, columns=['index', 'Nor_H', 'Nash_H', 'Nor_A', 'Nash_A',
                                           'Nor_P', 'Nash_P', 'H_diff', 'A_diff', 'P_diff'])
    result['H_abs'] = np.abs(result['H_diff'])
    result['A_abs'] = np.abs(result['A_diff'])
    result['P_abs'] = np.abs(result['P_diff'])
    #    pvalue = getPvalue_from_random_network(result, normaledges, nashedges, Species, rt=1000)
    
    return result

def getRandomNetwork(net):
    nodeset = list(net.nodes)
    redges = np.array([(int(i), int(j), float(v)) for i,j,v in net.edges.data('weight')])
    redges[:, 0] = [random.choice(nodeset) for i in range(len(redges))]
    redges[:, 1] = [random.choice(nodeset) for i in range(len(redges))]
    redges = [(int(i), int(j), v) for i,j,v in redges]
    rnet = nx.DiGraph()
    rnet.add_nodes_from(nodeset)
    rnet.add_weighted_edges_from(redges)
    return rnet

def getRandom_importances(net, h, a, p, rt=100):
    p_value = {}
    r = 0
    while(r<rt-1):
        try:
            rnet = getRandomNetwork(net)
            rh, ra, rp = getPageRank_Hub(rnet)
            for index in net.nodes:
                hi = h.get(index, 0)
                ai = a.get(index, 0)
                pi = p.get(index, 0)
                rhi = rh.get(index, 0)
                rai = ra.get(index, 0)
                rpi = rp.get(index, 0)
                p_value[index] = p_value.get(index, [1, 1, 1])
                if rhi >= hi:
                    p_value[index][0] += 1
                if rai >= ai:
                    p_value[index][1] += 1
                if rpi >= pi:
                    p_value[index][2] += 1
            print(r)
            r += 1
        except:
            pass
    for index in p_value.keys():
        p_value[index][0] = p_value[index][0]/float(rt)
        p_value[index][1] = p_value[index][1]/float(rt)
        p_value[index][2] = p_value[index][2]/float(rt)
    return p_value

def importance(net, rt=100):
    h, a, p = getPageRank_Hub(net)
    result = []
    p_value = getRandom_importances(net, h, a, p, rt)
    for index in nornet.nodes:
        hi = h.get(index, 0)
        ai = a.get(index, 0)
        pi = p.get(index, 0)
        phi = p_value[index][0]
        pai = p_value[index][1]
        ppi = p_value[index][2]
        result.append([index, hi, ai, pi, phi, pai, ppi])
    #result = sorted(result, key=lambda x:abs(x[1]), reverse=True)
    result = pd.DataFrame(result, columns=['index', 'H', 'A', 'P', 'H_p', 'A_p', 'P_p'])
    result['log_H_p'] = -np.log10(result['H_p'])
    return result

def network_score(net):
    ts = nx.transitivity(net)
    #acc = [len(c) for c in sorted(nx.connected_components(net.to_undirected()), key=len, reverse=True)][0]
    c = sorted(nx.connected_components(net.to_undirected()), key=len, reverse=True)[0]
    pnet = net.subgraph(c).copy().to_undirected()
    acc = nx.global_efficiency(pnet)
    return ts, acc

def destroyIndex(net, dtype):
    h, a, p = getPageRank_Hub(net)
    rank = []
    for index in net.nodes:
        rank.append((index, h.get(index, 0)))
    rank = sorted(rank, key=lambda x:x[1], reverse=True)
    result = []
    ts, acc = network_score(net)
    result.append([0, 'Origin', ts, acc])
    rnet = net.copy()
    i = 1
    pdata = pd.read_csv('network/'+dtype+'_imp.csv', index_col=1)
    print(pdata)
    for v, s in rank[:10]:
        rnet.remove_node(v)
        ts, acc = network_score(rnet)
        result.append([i, v, ts, acc, round(pdata.loc[v, 'H_p'], 3)])
        i += 1
    result = pd.DataFrame(result, columns=['index', 'node', 'transitivity', 'efficiency', 'p-value'])
    return result
    


if __name__ == '__main__':
    Species, diff_exp = getRankSumDiff()
    #diff_exp.to_csv('network/diff_exp.csv', index=False)
    nornet = buildNetwork(Species, dataname = 'Normal')
    nashnet = buildNetwork(Species, dataname = 'NASH')
    diff_imp = importance_diff(nornet, nashnet)
    #diff_imp.to_csv('network/diff_imp.csv', index=False)
    
#    nor_imp = importance(nornet, rt=1000)
#    nor_imp.to_csv('network/nor_imp.csv')
#    nash_imp = importance(nashnet, rt=1000)
#    nash_imp.to_csv('network/nash_imp.csv')

    nor_des = destroyIndex(nornet, 'nor')
    nash_des = destroyIndex(nashnet, 'nash')
    print(nor_des)
    
    plt.figure(1, (6,6), dpi=300)
    plt.plot(nor_des['index'], nor_des['p-value'], 'g-.', marker='o', alpha=0.8, label='P value')
    plt.plot(nor_des['index'], nor_des['transitivity'], 'r-', marker='D', label='Transitivity')
    plt.plot(nor_des['index'], nor_des['efficiency'], 'b--', marker='s', label='Efficiency')
    plt.axvline(2.5, alpha=0.5, color='gray', linestyle='--', lw=3)
    for i in range(1, 11):
        plt.annotate(nor_des.loc[i, 'p-value'], xy=(nor_des.loc[i, 'index']-0.5, nor_des.loc[i, 'p-value']+0.02))
    plt.xticks(nor_des['index'], nor_des['node'])
    plt.xlabel('Species')
    plt.ylabel('Importance')
    plt.legend(loc='upper right')
    plt.show()
    
    plt.figure(1, (6,6), dpi=300)
    plt.plot(nash_des['index'], nash_des['p-value'], 'g-.', marker='o', alpha=0.8, label='P value')
    plt.plot(nash_des['index'], nash_des['transitivity'], 'r-', marker='D', label='Transitivity')
    plt.plot(nash_des['index'], nash_des['efficiency'], 'b--', marker='s', label='Efficiency')
    plt.axvline(3.5, alpha=0.5, color='gray', linestyle='--', lw=3)
    for i in range(1, 11):
        plt.annotate(nash_des.loc[i, 'p-value'], xy=(nash_des.loc[i, 'index']-0.5, nash_des.loc[i, 'p-value']+0.02))
    plt.xticks(nash_des['index'], nash_des['node'])
    plt.xlabel('Species')
    plt.ylabel('Importance')
    plt.legend(loc='upper right')
    plt.show()
    
