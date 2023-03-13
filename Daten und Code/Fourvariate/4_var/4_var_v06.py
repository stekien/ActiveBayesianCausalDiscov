# -*- coding: utf-8 -*-
"""
Created on Sat Aug 27 09:49:14 2022
@author: Stefan Kienle (stefan.kienle@tum.de)
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

X2_int = [1]
X1_int = [f_1(x) + rng.normal(loc=0.0, scale=np.sqrt(0.12), size=None) for x in X2_int] 
X3_int = [f_3(x, y) + rng.normal(loc=0.0, scale=np.sqrt(0.08), size=None) for x,y in zip(X1_int,X2_int)] 
X4_int = [f_4(x) + rng.normal(loc=0.0, scale=np.sqrt(0.1), size=None) for x in X3_int]

x = np.array([X1_int[0],X2_int[0],X3_int[0],X4_int[0]])

G_obj_test =  {"graph":G_lst_params['graph'][1],"data":G_lst_params['data'],"sigma":G_lst_params['sigma'][1],"gamma":G_lst_params['gamma'][1],"C_G":G_lst_params['C_G'][1]} 

test_fct = probDj_G(x,G_obj_test,0)
          
            
def sample_int_j_givenG(x, G_obj, j):
    
    n = 100
    
    #identify source node
    G = G_obj['graph']
    data = G_obj["data"]
    sigma = G_obj["sigma"]
    gamma = G_obj["gamma"]
    #C_G = G_obj['C_G']
    
    rows, columns = G.shape
    indices = list(range(rows)) 
            
    dim = np.sum(G, axis=1)  
    #dim = [int(x) for x in dim]
    sample = {}
    
    for i in indices:
        sample[str(i)] = list(range(n))
    # set intervention value fix
    sample[str(j)] = list(itertools.repeat(x, n))
    
    source_nodes = np.where(dim == 0)
    source_nodes = list(source_nodes[0])
    #print(source_nodes)
    
    for i in source_nodes:
        sample[str(i)] = list(np.random.normal(loc=0.0, scale=sigma[i], size=n))
    
    first_order = np.where(dim == 1)
    first_order = list(first_order[0])

    second_order = np.where(dim == 2)
    second_order = list(second_order[0])
    
    third_order = np.where(dim == 3)
    third_order = list(third_order[0])
    
    for k in range(n):
        
        for i in first_order:
            if i != j:
                index = [int(x*y) for x,y in zip(G[i,:].tolist(),indices) if x > 0]
                #print('index: ', index, '\n')
                 
                covariates = data[index,:]
                 
                target = data[i,:]
                 
                kernel = ConstantKernel(1, constant_value_bounds="fixed") * Matern(length_scale=gamma[i], length_scale_bounds="fixed", nu=2.5)
                 
                gp = GaussianProcessRegressor(
                    kernel=kernel, alpha=sigma[i]
                )
                # fit the model to the training data
                gp.fit(covariates.T, target)
                
                sam = []
                for l in index:
                    sam.append(sample[str(l)][k])
                 
                mu_int, std_sig_int = gp.predict(np.array(sam).reshape(1,-1), return_std=True)
                 
                sample[str(i)][k] = np.random.normal(loc=mu_int, scale=np.sqrt(sigma[i]**2 + std_sig_int**2), size=1)[0] 
                
        for i in second_order:
            if i != j:
                index = [int(x*y) for x,y in zip(G[i,:].tolist(),indices) if x > 0]
                #print('index: ', index, '\n')
                 
                covariates = data[index,:]
                 
                target = data[i,:]
                 
                kernel = ConstantKernel(1, constant_value_bounds="fixed") * Matern(length_scale=gamma[i], length_scale_bounds="fixed", nu=2.5)
                 
                gp = GaussianProcessRegressor(
                    kernel=kernel, alpha=sigma[i]
                )
                # fit the model to the training data
                gp.fit(covariates.T, target)
                
                sam = []
                for l in index:
                    sam.append(sample[str(l)][k])
                 
                mu_int, std_sig_int = gp.predict(np.array(sam).reshape(1,-1), return_std=True)
                 
                sample[str(i)][k] = np.random.normal(loc=mu_int, scale=np.sqrt(sigma[i]**2 + std_sig_int**2), size=1)[0] 
                
        for i in third_order:
            if i != j:
                index = [int(x*y) for x,y in zip(G[i,:].tolist(),indices) if x > 0]
                #print('index: ', index, '\n')
                 
                covariates = data[index,:]
                 
                target = data[i,:]
                 
                kernel = ConstantKernel(1, constant_value_bounds="fixed") * Matern(length_scale=gamma[i], length_scale_bounds="fixed", nu=2.5)
                 
                gp = GaussianProcessRegressor(
                    kernel=kernel, alpha=sigma[i]
                )
                # fit the model to the training data
                gp.fit(covariates.T, target)
                
                sam = []
                for l in index:
                    sam.append(sample[str(l)][k])
                 
                mu_int, std_sig_int = gp.predict(np.array(sam).reshape(1,-1), return_std=True)
                 
                sample[str(i)][k] = np.random.normal(loc=mu_int, scale=np.sqrt(sigma[i]**2 + std_sig_int**2), size=1)[0] 
                
    return sample
    
    
test_sample = sample_int_j_givenG(1, G_obj_test, 0) 


def probG_Dj(x,G,G_lst_,j):
    
    summands = list(range(len(G_lst_['graph'])))
    
    for i in range(len(G_lst_['graph'])):
        G_obj = {"graph":G_lst_['graph'][i],"data":G_lst_['data'],"sigma":G_lst_['sigma'][i],"gamma":G_lst_['gamma'][i],"C_G":G_lst_['C_G'][i]}
        summands[i] = probDj_G(x,G_obj,j)
        
    nominator = probDj_G(x,G,j)

    return nominator/np.sum(summands)

test_fct2 = probG_Dj(x,G_obj_test,G_lst_params,1)

### The below code can be used to compute the objectives in the first iteration of 
### the procedure. But caution it takes very long.
def g_j(u, G_lst_, j):
    
    summands = list(range(len(G_lst_['graph'])))
    
    #def mc_calc(p):
        #x_sam = np.array([int_sample[str(0)][p],int_sample[str(1)][p],int_sample[str(2)][p],int_sample[str(3)][p]])
        #mc_su = probG_Dj(x_sam,G,G_lst_,j)
        #return mc_su
        
    
    for i in range(len(G_lst_['graph'])):
        G = {"graph":G_lst_['graph'][i],"data":G_lst_['data'],"sigma":G_lst_['sigma'][i],"gamma":G_lst_['gamma'][i],"C_G":G_lst_['C_G'][i]}
        int_sample = sample_int_j_givenG(u, G, j)
        mc_sum = list(range(len(int_sample[str(0)])))
           
        for l in range(len(int_sample[str(0)])):
            x_sam = np.array([int_sample[str(0)][l],int_sample[str(1)][l],int_sample[str(2)][l],int_sample[str(3)][l]])
            mc_sum[l] = probG_Dj(x_sam,G,G_lst_,j)
            
        summands[i] = (1/len(int_sample[str(0)]))*np.sum(mc_sum)
        
    return np.sum(summands)


#test_fct_g = g_j(0,G_lst_params,1)

def f_j(i,u, G_lst_, j):
    
    #summands = list(range(len(G_lst_['graph'])))
        
    G = {"graph":G_lst_['graph'][i],"data":G_lst_['data'],"sigma":G_lst_['sigma'][i],"gamma":G_lst_['gamma'][i],"C_G":G_lst_['C_G'][i]}
    int_sample = sample_int_j_givenG(u, G, j)
    mc_sum = list(range(len(int_sample[str(0)])))
           
    for l in range(len(int_sample[str(0)])):
        x_sam = np.array([int_sample[str(0)][l],int_sample[str(1)][l],int_sample[str(2)][l],int_sample[str(3)][l]])
        mc_sum[l] = probG_Dj(x_sam,G,G_lst_,j)
            
    summand = (1/len(int_sample[str(0)]))*np.sum(mc_sum)
        
    return summand


#f_h = lambda h: f_j(h,0,G_lst_params,1)

def f_h(h):
    return f_j(h,0,G_lst_params,1)

if __name__ == "__main__":
    nprocs= 10
    
    # printthenumberofcores
    print("Number of workers equals %d"% nprocs)
    # createa poolofworkers
    pool= Pool(processes=nprocs)
    # createan arrayof10 integers, from1 to10
    a = range(len(G_lst_params['graph']))
    
    summands = pool.map(f_h, a)
    #print(result)
