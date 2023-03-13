# -*- coding: utf-8 -*-
"""
Created on Wed Jul 27 19:40:39 2022

@author: Stefan Kienle (stefan.kienle@tum.de)
"""
import numpy as np
from scipy.stats import bernoulli as bern

G_list = [np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),np.array([[0,1,0,0],[0,0,0,0],[1,1,0,0],[0,0,1,0]])] 
G = np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])

def check_DAG_4_var(G, G_list):
    
    # check symmetrie
    G_sym = G + G.T
    test_sym = (G_sym >= 2).any()
    if test_sym: return False

    # diagonal check
    G_diag = np.diag(G)
    test_diag = (G_diag != 0).any()
    if test_diag: return False

    # check if there is a sink node
    G_rowsum = np.sum(G, axis=1)
    test_sink = (G_rowsum > 0).all()
    if test_sink: return False

    # check if there is a source node
    G_colsum = np.sum(G, axis=0)
    test_source = (G_colsum > 0).all()
    if test_source: return False
    
    for G_at in G_list:
        G_diff = abs(G - G_at) 
        G_diff1 = abs(G - 2*G_at)
        G_diff2 = abs(2*G - G_at)
        if (np.sum(G_at) == 0) and np.sum(G_diff) + np.sum(G_diff1) + np.sum(G_diff2) == 0: return False
        if np.sum(G_diff) == 0: return False
                
    #print("Phase 1")
    variables = list(range(len(G[0,:])))
    # subsystems with one variable deleted
    for i in variables:
        G_sub1 = np.delete(G, i, axis=0) #delete i-th row
        G_sub1 = np.delete(G_sub1, i, axis=1) #delete i-th col
        #print(G_sub1)
        
        # check if there is a sink node
        G_sub1_rowsum = np.sum(G_sub1, axis=1)
        test_sink = (G_sub1_rowsum > 0).all()
        if test_sink: return False

        # check if there is a source node
        G_sub1_colsum = np.sum(G_sub1, axis=0)
        test_source = (G_sub1_colsum > 0).all()
        if test_source: return False
        
        #remaining_var = [j for j in variables if j != i]
        remaining_var = list(range(len(G_sub1[0,:])))
        
        for i1 in remaining_var:
            
            G_sub2 = np.delete(G_sub1, i1, axis=0) #delete i-th row
            G_sub2 = np.delete(G_sub2, i1, axis=1) #delete i-th col
            #print(G_sub2)
            
            # check if there is a sink node
            G_sub2_rowsum = np.sum(G_sub2, axis=1)
            test_sink = (G_sub2_rowsum > 0).all()
            if test_sink: return False

            # check if there is a source node
            G_sub2_colsum = np.sum(G_sub2, axis=0)
            test_source = (G_sub2_colsum > 0).all()
            if test_source: return False
    
    return True

test = check_DAG_4_var(G, G_list)

def generate_DAG_candidate_4_var():
    # DAG Generator
    #sample = bern.rvs(0.8, size=6)
    G_sample = np.zeros((4,4))

    index0 = list(range(4-1))
    ind = list()

    for l in index0:
        index1 = list(range(l+1,4))
        for l1 in index1:
            ind.append([l1,l])

    count = 0
    count_ones = 0
    for i in ind:   
        sample = bern.rvs((6-count_ones)/15, size=1)
        count_ones = count_ones + sample
        G_sample[i[0],i[1]] = sample
        if sample == 1:
            	G_sample[i[1],i[0]] = abs(sample - 1)
        sample2 = bern.rvs((6-count_ones)/35, size=1)
        count_ones = count_ones + sample2
        G_sample[i[1],i[0]] = sample2
        count += 1
        
    return G_sample

counter = 0
while len(G_list) < 543:
    np.random.seed(seed=counter)
    counter +=1
    if counter % 50000 == 0:
        print(counter)
    G_can = generate_DAG_candidate_4_var()
    if check_DAG_4_var(G_can, G_list):
        G_list.append(G_can)
        
        
G1_try = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,0]])
G2_try = np.array([[0,1,1,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]])
G3_try = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,0,1,0]])
G4_try = np.array([[0,1,0,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]])
G5_try = np.array([[0,0,1,1],[0,0,1,1],[0,0,0,1],[0,0,0,0]])
G6_try = np.array([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,0,0]])
G7_try = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])

G_Test = [G1_try,G2_try,G3_try,G4_try,G5_try,G6_try,G7_try]

for G_can2 in G_Test:
    if check_DAG_4_var(G_can2, G_list):
        G_list.append(G_can2)


## change distribution a bit and try again
def generate_DAG_candidate_4_var2():
    # DAG Generator
    #sample = bern.rvs(0.8, size=6)
    G_sample = np.zeros((4,4))

    index0 = list(range(4-1))
    ind = list()

    for l in index0:
        index1 = list(range(l+1,4))
        for l1 in index1:
            ind.append([l1,l])

    count = 0
    count_ones = 0
    for i in ind:   
        sample = bern.rvs((6-count_ones)/15, size=1)
        count_ones = count_ones + sample
        G_sample[i[0],i[1]] = sample
        if sample == 1:
            	G_sample[i[1],i[0]] = abs(sample - 1)
        else:
            sample2 = bern.rvs((6-count_ones)/15, size=1)
            count_ones = count_ones + sample2
            G_sample[i[1],i[0]] = sample2
        count += 1
        
    return G_sample

counter = 0
while len(G_list) < 543:
    np.random.seed(seed=counter)
    counter +=1
    if counter % 50000 == 0:
        print(counter)
    G_can = generate_DAG_candidate_4_var2()
    if check_DAG_4_var(G_can, G_list):
        G_list.append(G_can)

#save and load files
np.save('G_list.npy', G_list, allow_pickle=True)
G_list = np.load('G_list.npy', allow_pickle=True)


