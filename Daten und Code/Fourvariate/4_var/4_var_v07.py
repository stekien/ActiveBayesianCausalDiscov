# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 10:00:59 2022

@author: Stefan
"""
from functools import reduce
from multiprocessing import Pool, current_process
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel, Matern
from scipy.stats import norm, uniform#,chi2, gamma, lognorm
import matplotlib.pyplot as plt
from skopt import gp_minimize
from joblib import Parallel, delayed
import spyder_kernels
import itertools
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
rng = np.random#.RandomState(12)

##############################################################################
#           FUNCTIONS
##############################################################################

# generate data
G_true = np.array([[0,1,0,0],[0,0,0,0],[1,1,0,0],[0,0,1,0]])

def f_1(x):
    return 0.25*x**3

def f_3(x1, x2):
    return 2*np.tanh(x1) + np.sin(x2)

def f_4(x):
    return 0.5 * x

X2_initial = rng.normal(loc=0.0, scale=1, size=5).tolist()
X1_initial = [f_1(x) + rng.normal(loc=0.0, scale=np.sqrt(0.12), size=None) for x in X2_initial] 
X3_initial = [f_3(x, y) + rng.normal(loc=0.0, scale=np.sqrt(0.08), size=None) for x,y in zip(X1_initial,X2_initial)] 
X4_initial = [f_4(x) + rng.normal(loc=0.0, scale=np.sqrt(0.1), size=None) for x in X3_initial]

data = np.array([X1_initial,X2_initial,X3_initial,X4_initial])

# import list of graphs
data_raw = spyder_kernels.utils.iofuncs.load_dictionary('C:\\Users\\stefa\\Desktop\\Masterarbeit_StefanKienle\\Daten und Code\\Fourvariate\\4_var\\V1.spydata')
G_lst = data_raw[0]['Graphs']


def probDgivenG(G_obj):
    
    G = G_obj["graph"] 
    data = G_obj["data"]
    sigma = G_obj["sigma"]
    gamma = G_obj["gamma"]
    
    rows, columns = G.shape
    variables = list(range(1,rows+1))
    
    res = np.zeros(rows)

    for i in variables:
        
        dim = np.sum(G[i-1,:])
        
        if dim > 0:
        
            index = [int(x*y - 1) for x,y in zip(G[i-1,:].tolist(),variables) if x > 0]
            #print('index: ', index, '\n')
            
            covariates = data[index,:]
            
            target = data[i-1,:]
            
            kernel = ConstantKernel(1, constant_value_bounds="fixed") * Matern(length_scale=gamma[i-1], length_scale_bounds="fixed", nu=2.5)
            
            gaussian_process = GaussianProcessRegressor(
                kernel=kernel, alpha=sigma[i-1]
            )
            # fit the model to the training data
            gaussian_process.fit(covariates.T, target)
                
            res[i-1] = gaussian_process.log_marginal_likelihood_value_
            
        else:
            
            target = data[i-1,:]
            
            res[i-1] = np.sum(np.log(norm.pdf(target, scale=sigma[i-1])))
                  
    return np.exp(np.sum(res))

def attach_graph_params(G_lst, data):
    
    G_lst_con_params = {"graph":[],"data":data,"sigma":[],"gamma":[], "C_G":[]}
    
    for G in G_lst:
        
        rows, columns = G.shape
        indices = list(range(rows))
        
        # initialize error vector and parameter vector
        sigma_G = list(range(rows))
        gamma_G = list(range(rows))
        
        for i in indices:
            
            dim = np.sum(G[i,:])
            
            if dim == 0:
                sigma_G[i] = 1
                gamma_G[i] = 1
            else:
                sigma_G[i] = np.sqrt(0.1)
                gamma_G[i] = 1.75
         
        #G_obj_append = {"graph":G,"data":data,"sigma":sigma_G,"gamma":gamma_G}   
        
        G_lst_con_params['graph'].append(G)
        G_lst_con_params['sigma'].append(sigma_G)
        G_lst_con_params['gamma'].append(gamma_G)
        
    return G_lst_con_params
            
G_lst_params = attach_graph_params(G_lst, data)           
            
# attach probabilities 
for i in range(543):
     
    G_obj_i =  {"graph":G_lst_params['graph'][i],"data":G_lst_params['data'],"sigma":G_lst_params['sigma'][i],"gamma":G_lst_params['gamma'][i]}          
    
    #log_prob = probDgivenG(G_obj_i)        
            
    G_lst_params['C_G'].append(probDgivenG(G_obj_i))       
            
            
post_probs = [x/np.sum(G_lst_params['C_G']) for x in G_lst_params['C_G']]            
            
fav = post_probs.index(np.max(post_probs))            
           
### find optimal intervention ###
# D_1
# G_obj = {"graph":[],"data":data,"sigma":[],"gamma":[], "C_G":[]}
def probDj_G(x,G_obj,j):
    
    #identify source node
    G = G_obj['graph']
    data = G_obj["data"]
    sigma = G_obj["sigma"]
    gamma = G_obj["gamma"]
    C_G = G_obj['C_G']
    
    rows, columns = G.shape
    indices = list(range(rows))     
    
    res = np.zeros(rows)
            
    for i in indices:
         
        dim = np.sum(G[i,:])
         
        if dim > 0 and i != j:
         
            index = [int(p*y) for p,y in zip(G[i,:].tolist(),indices) if p > 0]
            #print('index: ', index, '\n')
             
            covariates = data[index,:]
            #print(covariates.shape)
             
            target = data[i,:]
             
            kernel = ConstantKernel(1, constant_value_bounds="fixed") * Matern(length_scale=gamma[i], length_scale_bounds="fixed", nu=2.5)
             
            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=sigma[i]
            )
            # fit the model to the training data
            gp.fit(covariates.T, target)
             
            mu_int, std_sig_int = gp.predict(x[index].reshape(1,-1), return_std=True)
             
            res[i] = np.log(norm.pdf(x[i].reshape(-1,), loc=mu_int, scale=np.sqrt(std_sig_int**2 + sigma[i]**2)))
             
        else:
            if i == j:
                res[i] = np.log(1)
            else:
                res[i] = np.log(norm.pdf(x[i], loc=0.0, scale=sigma[i]))
               
    return np.exp(np.sum(res) + np.log(C_G))   

# intervention data point
X2_int = [1]
X1_int = [f_1(x) + rng.normal(loc=0.0, scale=np.sqrt(0.12), size=None) for x in X2_int] 
X3_int = [f_3(x, y) + rng.normal(loc=0.0, scale=np.sqrt(0.08), size=None) for x,y in zip(X1_int,X2_int)] 
X4_int = [f_4(x) + rng.normal(loc=0.0, scale=np.sqrt(0.1), size=None) for x in X3_int]

x = np.array([X1_int[0],X2_int[0],X3_int[0],X4_int[0]])

G_obj_test =  {"graph":G_lst_params['graph'][1],"data":G_lst_params['data'],"sigma":G_lst_params['sigma'][1],"gamma":G_lst_params['gamma'][1],"C_G":G_lst_params['C_G'][1]} 

test_fct = probDj_G(x,G_obj_test,0)


def opt(G_obj,j):
    
    # generate data sequence which will be predicted in the later graphs
    X_predict = np.linspace(start=-3, stop=3, num=1_000).reshape(-1, 1)
    y_predict = np.linspace(start=-3, stop=3, num=1_000).reshape(-1, 1)
    
    #identify source node
    G = G_obj['graph']
    data = G_obj["data"]
    sigma = G_obj["sigma"]
    gamma = G_obj["gamma"]
    C_G = G_obj['C_G']
    
    rows, columns = G.shape
    indices = list(range(rows))     
    
    res = np.zeros(rows)
            
    for i in indices:
         
        dim = np.sum(G[i,:])
         
        if dim > 0 and i != j:
         
            index = [int(p*y) for p,y in zip(G[i,:].tolist(),indices) if p > 0]
            #print('index: ', index, '\n')
             
            covariates = data[index,:]
            #print(covariates.shape)
             
            target = data[i,:]
             
            kernel = ConstantKernel(1, constant_value_bounds="fixed") * Matern(length_scale=gamma[i], length_scale_bounds="fixed", nu=2.5)
             
            gp = GaussianProcessRegressor(
                kernel=kernel, alpha=sigma[i]
            )
            # fit the model to the training data
            gp.fit(covariates.T, target)
             
            mu_int, std_sig_int = gp.predict(x[index].reshape(1,-1), return_std=True)
             
            res[i] = np.log(norm.pdf(x[i].reshape(-1,), loc=mu_int, scale=np.sqrt(std_sig_int**2 + sigma[i]**2)))
             
        else:
            if i == j:
                res[i] = np.log(1)
            else:
                res[i] = np.log(norm.pdf(x[i], loc=0.0, scale=sigma[i]))
               
    return np.exp(np.sum(res) + np.log(C_G)) 