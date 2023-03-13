# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 07:23:14 2022

@author: Stefan Kienle (stefan.kienle@tum.de)
"""
import numpy as np
from scipy.stats import bernoulli as bern
import itertools

G_list = [np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),np.array([[0,1,0,0],[0,0,0,0],[1,1,0,0],[0,0,1,0]])] 
#G = np.array([[0,0,1,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]])

def check_DAG_4_var(G, G_list):
    
    # check symmetrie
    G_sym = G + G.T
    test_sym = (G_sym >= 2).any()
    if test_sym:
        print('Graph is symmetric, thus rejected.\n')
        return False

    # diagonal check
    G_diag = np.diag(G)
    test_diag = (G_diag != 0).any()
    if test_diag: 
        print('Trace of Graph is not zero, thus rejected.\n')
        return False

    # check if there is a sink node
    G_rowsum = np.sum(G, axis=1)
    test_sink = (G_rowsum > 0).all()
    if test_sink: 
        print('Graph has no sink node, thus rejected.\n')
        return False

    # check if there is a source node
    G_colsum = np.sum(G, axis=0)
    test_source = (G_colsum > 0).all()
    if test_source:
        print('Graph has no source node, thus rejected.\n')
        return False
    
    for G_at in G_list:
        G_diff = abs(G - G_at) 
        G_diff1 = abs(G - 2*G_at)
        G_diff2 = abs(2*G - G_at)
        if (np.sum(G_at) == 0) and (np.sum(G_diff) + np.sum(G_diff1) + np.sum(G_diff2) == 0):
            print('Graph is the zero Graph and alredy in the list, thus rejected.\n')
            return False
        if np.sum(G_diff) == 0:
            print('Graph already included in list, thus rejected.\n')
            return False
                
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

#test = check_DAG_4_var(G, G_list)

topo_order = [4,3,2,1]

def generate_3_node(topo_order):
    
    G1 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    G2 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    G3 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    G4 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    G5 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    G6 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    G7 = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    indices = [x - 1 for x in topo_order]
    
    #G_basis = np.array([[0,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]])
    
    G_help2 = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
    G1[indices[1],:] = G_help2[indices[0],:]
    G2[indices[1],:] = G_help2[indices[0],:]
    G3[indices[1],:] = G_help2[indices[0],:]
    G4[indices[1],:] = G_help2[indices[0],:]
    G5[indices[1],:] = [0, 0, 0, 0]
    G6[indices[1],:] = [0, 0, 0, 0]
    G7[indices[1],:] = [0, 0, 0, 0]
    
    G1[indices[2],:] = [0, 0, 0, 0]
    G2[indices[2],indices[0]] = 1
    G3[indices[2],indices[1]] = 1
    G4[indices[2],[indices[0],indices[1]]] = [1,1]
    G5[indices[2],[indices[0],indices[1]]] = [1,1]
    G6[indices[2],indices[1]] = 1
    G7[indices[2],indices[0]] = 1
    
    return [G1,G2,G3,G4,G5,G6,G7]
    
    
variables = [1,2,3,4]
TopoOrders = list(itertools.permutations(variables))

for i in TopoOrders:
    to = list(i)
    Gs = generate_3_node(i)
    for j in Gs:
        if check_DAG_4_var(j, G_list):
            G_list.append(j)
    
    
    
    
    
    
    
    
    
    