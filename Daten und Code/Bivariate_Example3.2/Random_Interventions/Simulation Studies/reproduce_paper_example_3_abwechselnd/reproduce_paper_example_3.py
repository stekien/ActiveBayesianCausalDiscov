# -*- coding: utf-8 -*-
"""
Created on Tue May 24 16:30:14 2022

@author: Stefan Kienle (stefan.kienle@tum.de)
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel, Matern
from scipy.stats import norm, uniform#,chi2, gamma, lognorm
import matplotlib.pyplot as plt
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
# plot params
params = {
    "legend.fontsize": 13.5,
    "axes.titlesize": 20,
    "figure.figsize": (4, 4),
    "figure.dpi": 100,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "lines.linewidth": 3,
    "lines.markeredgewidth": 1.5,
    "lines.markersize": 10,
    "lines.marker": "o",
    "patch.edgecolor": "black",
}

plt.rcParams.update(params)
plt.style.use("seaborn")
# 
#from joblib import Parallel, delayed
def pobD_givenG(X, Y, gamma, sigma_x, sigma_y):
    
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=gamma, length_scale_bounds="fixed")
       
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=sigma_y 
    )
    # fit the model to the training data
    gaussian_process.fit(X, Y)
    
    res = gaussian_process.log_marginal_likelihood_value_ + np.sum(np.log(norm.pdf(X.reshape(-1,), scale=sigma_x))) 
          
    return np.exp(res)


def pobD_givenG_int(X, Y, gamma, sigma_x, sigma_y, X_interv_x=[], Y_interv_x=[], X_interv_y=[], Y_interv_y=[]):
    
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(length_scale=gamma, length_scale_bounds="fixed")
       
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=sigma_y 
    )
    # fit the model to the training data
    try:
        X_GPr = np.concatenate([X, X_interv_x])
        Y_GPr = np.concatenate([Y, Y_interv_x])
    except:
        X_GPr = X
        Y_GPr = Y
    #print(X_GPr)
    gaussian_process.fit(X_GPr, Y_GPr)
    
    try:
        res = gaussian_process.log_marginal_likelihood_value_ + np.sum(np.log(norm.pdf(X.reshape(-1,), scale=sigma_x))) + np.sum(np.log(norm.pdf(X_interv_y.reshape(-1,), scale=sigma_x)))
    except:
        res = gaussian_process.log_marginal_likelihood_value_ + np.sum(np.log(norm.pdf(X.reshape(-1,), scale=sigma_x))) 

    return np.exp(res)

# methods to perform interventions
def rnd_interv_x(X_train, Y_train):
    x_interv = np.random.uniform(-3,3,[1,1])
    #x_interv = np.random.uniform(min(X_train),max(X_train),[1,1])
    y_dox = np.array([2 * np.tanh(x_interv) + rng.normal(loc=0.0, scale=noise_std, size=1)]).reshape(1,)
    X_train_1 = np.concatenate([X_train, x_interv])  
    Y_train_1 = np.concatenate([Y_train, y_dox])
    return X_train_1, Y_train_1

def rnd_interv_y(X_train, Y_train):
    y_doy = np.random.uniform(-2.1,2.1,[1,1]).reshape(1,)
    #y_doy = np.random.uniform(min(y_train),max(y_train),[1,1]).reshape(1,)
    x_doy = rng.normal(loc=0.0, scale=1, size=1).reshape(1,1)
    X_train_2 = np.concatenate([X_train, x_doy])  
    Y_train_2 = np.concatenate([Y_train, y_doy]) 
    return X_train_2, Y_train_2

# set parameters for the two graphs respectively 
gamma_g1 = 1.75
sigma_x_g1 = 1
sigma_y_g1 = np.sqrt(0.1)

gamma_g2 = 1.75
sigma_x_g2 = 1
sigma_y_g2 = np.sqrt(0.1)


result_0 = np.zeros((10000,2))
result_1 = np.zeros((10000,2))
result_2 = np.zeros((10000,2))
result_3 = np.zeros((10000,2))
result_4 = np.zeros((10000,2))
result_5 = np.zeros((10000,2))
result_6 = np.zeros((10000,2))
result_7 = np.zeros((10000,2))
result_8 = np.zeros((10000,2))
result_9 = np.zeros((10000,2))
result_10 = np.zeros((10000,2))

probG_givenD_isproportional_xtoy_1 = []
probG_givenD_isproportional_ytox_1 = []  

for j in range(10000):
    np.random.seed(seed=j)
    
    # data is generated synthetically in this script
    rng = np.random.RandomState(j)
    # generate data sequence which will be predicted in the later graphs
    X_predict = np.linspace(start=-3, stop=3, num=1_000).reshape(-1, 1)
    y_predict = np.linspace(start=-3, stop=3, num=1_000).reshape(-1, 1)
    
    # generate (large) sample of the underlying distribution
    X = rng.normal(loc = 0.0, scale = 1, size=(1000)).reshape(-1, 1)
    y = (2 * np.tanh(X)).reshape(1000,)
    
    # sample (size) draws from the large sample, i.e. define training data
    training_indices = rng.choice(np.arange(y.size), size=5, replace=False)
    X_train, y_train = X[training_indices], y[training_indices]
    
    # add stochasticity to Y
    noise_std = np.sqrt(0.1)
    y_train_noisy = y_train + rng.normal(loc=0.0, scale=noise_std, size=y_train.shape)
    
    # initialise interventional data
    # intervention on x
    X_interv_x = np.random.uniform(-3,3,[1,1])
    Y_interv_x = np.array([2 * np.tanh(X_interv_x[0]) + rng.normal(loc=0.0, scale=noise_std, size=1)]).reshape(1,)
    
    #X_interv_x, Y_interv_x = rnd_interv_x(X_interv_x, Y_interv_x)
    
    # intervention on y
    X_interv_y = rng.normal(loc=0.0, scale=1, size=1).reshape(1,1)
    Y_interv_y = np.random.uniform(-2.1,2.1,[1,1]).reshape(1,)
    
    probG_givenD_isproportional_xtoy = []
    probG_givenD_isproportional_ytox = []  
    probG_givenD_isproportional_xtoy.append(pobD_givenG(X_train, y_train_noisy, gamma_g1, sigma_x_g1, sigma_y_g1))
    probG_givenD_isproportional_ytox.append(pobD_givenG(y_train_noisy.reshape(-1,1), X_train.reshape(-1,),gamma_g2, sigma_x_g2, sigma_y_g2))
    
    # first intervention
    probG_givenD_isproportional_xtoy.append(pobD_givenG_int(X_train, y_train_noisy, gamma_g1, sigma_x_g1, sigma_y_g1, X_interv_x = X_interv_x, Y_interv_x = Y_interv_x))
    probG_givenD_isproportional_ytox.append(pobD_givenG_int(y_train_noisy.reshape(-1,1), X_train.reshape(-1,),gamma_g2, sigma_x_g2, sigma_y_g2, X_interv_y =X_interv_x, Y_interv_y =Y_interv_x))
    
    # second intervention
    probG_givenD_isproportional_xtoy.append(pobD_givenG_int(X_train, y_train_noisy, gamma_g1, sigma_x_g1, sigma_y_g1, X_interv_x, Y_interv_x, X_interv_y, Y_interv_y))
    probG_givenD_isproportional_ytox.append(pobD_givenG_int(y_train_noisy.reshape(-1,1), X_train.reshape(-1,),gamma_g2, sigma_x_g2, sigma_y_g2,X_interv_y, Y_interv_y, X_interv_x, Y_interv_x))
    
    # remaining interventions
    for i in range(0,4):
        
        # only interventions on x  
        X_interv_x, Y_interv_x = rnd_interv_x(X_interv_x, Y_interv_x)
        probG_givenD_isproportional_xtoy.append(pobD_givenG_int(X_train, y_train_noisy, gamma_g1, sigma_x_g1, sigma_y_g1, X_interv_x, Y_interv_x, X_interv_y, Y_interv_y))
        probG_givenD_isproportional_ytox.append(pobD_givenG_int(y_train_noisy.reshape(-1,1), X_train.reshape(-1,),gamma_g2, sigma_x_g2, sigma_y_g2,X_interv_y, Y_interv_y, X_interv_x, Y_interv_x))
        #probG_givenD_isproportional_xtoy.append(pobD_givenG_int(X_train, y_train_noisy, gamma_g1, sigma_x_g1, sigma_y_g1, X_interv_x = X_interv_x, Y_interv_x = Y_interv_x))
        #probG_givenD_isproportional_ytox.append(pobD_givenG_int(y_train_noisy.reshape(-1,1), X_train.reshape(-1,),gamma_g2, sigma_x_g2, sigma_y_g2, X_interv_y =X_interv_x, Y_interv_y =Y_interv_x))
    
        # only interventions on y 
        X_interv_y , Y_interv_y  = rnd_interv_y(X_interv_y, Y_interv_y)
        probG_givenD_isproportional_xtoy.append(pobD_givenG_int(X_train, y_train_noisy, gamma_g1, sigma_x_g1, sigma_y_g1, X_interv_x, Y_interv_x, X_interv_y, Y_interv_y))
        probG_givenD_isproportional_ytox.append(pobD_givenG_int(y_train_noisy.reshape(-1,1), X_train.reshape(-1,),gamma_g2, sigma_x_g2, sigma_y_g2,X_interv_y, Y_interv_y, X_interv_x, Y_interv_x))
        
        
    ProbGxtoy_givenD = [x / (x + y) for x,y in zip(probG_givenD_isproportional_xtoy, probG_givenD_isproportional_ytox)]   
    ProbGytox_givenD = [y / (x + y) for x,y in zip(probG_givenD_isproportional_xtoy, probG_givenD_isproportional_ytox)]  
    
    result_0[j,0] = ProbGxtoy_givenD[0]
    result_0[j,1] = ProbGytox_givenD[0]
    
    result_1[j,0] = ProbGxtoy_givenD[1]
    result_1[j,1] = ProbGytox_givenD[1]
    
    result_2[j,0] = ProbGxtoy_givenD[2]
    result_2[j,1] = ProbGytox_givenD[2]
    
    result_3[j,0] = ProbGxtoy_givenD[3]
    result_3[j,1] = ProbGytox_givenD[3]
    
    result_4[j,0] = ProbGxtoy_givenD[4]
    result_4[j,1] = ProbGytox_givenD[4]
    
    result_5[j,0] = ProbGxtoy_givenD[5]
    result_5[j,1] = ProbGytox_givenD[5]
    
    result_6[j,0] = ProbGxtoy_givenD[6]
    result_6[j,1] = ProbGytox_givenD[6]
    
    result_7[j,0] = ProbGxtoy_givenD[7]
    result_7[j,1] = ProbGytox_givenD[7]
    
    result_8[j,0] = ProbGxtoy_givenD[8]
    result_8[j,1] = ProbGytox_givenD[8]
    
    result_9[j,0] = ProbGxtoy_givenD[9]
    result_9[j,1] = ProbGytox_givenD[9]
    
    result_10[j,0] = ProbGxtoy_givenD[10]
    result_10[j,1] = ProbGytox_givenD[10]
    
    probG_givenD_isproportional_xtoy_1.append(pobD_givenG(np.concatenate([X_train, X_interv_x]), np.concatenate([y_train_noisy, Y_interv_x]), gamma_g1, sigma_x_g1, sigma_y_g1))
    probG_givenD_isproportional_ytox_1.append(pobD_givenG(np.concatenate([y_train_noisy, Y_interv_x]).reshape(-1,1), np.concatenate([X_train, X_interv_x]).reshape(-1,),gamma_g2, sigma_x_g2, sigma_y_g2))
    
def evaluate(result):
    params = {
        "legend.fontsize": 25,
        "axes.titlesize": 60,
        "figure.figsize": (12, 12),
        "figure.dpi": 100,
        "axes.labelsize": 60,
        "xtick.labelsize": 40,
        "ytick.labelsize": 40,
        "lines.linewidth": 5,
        "lines.markeredgewidth": 2.5,
        "lines.markersize": 10,
        "lines.marker": "o",
        "patch.edgecolor": "black",
        "text.usetex": True,
        "font.family": "serif"
    }
    plt.style.use('seaborn')  
    plt.rcParams.update(params)
    fig, ax = plt.subplots()
    #plt.xticks(list(np.arange(0, 1, step=0.2)))
    plt.hist(result[:,0], bins=100, density=True)
    ax.tick_params(axis = 'x',labelrotation = 45)
    ax.set_xticks(np.arange(0, 1.2, step=0.2))
    plt.xlabel(r"$P(G_1|\textbf{D})$")
    plt.ylabel("$Frequency$")
    _ = plt.title("Histogram of " + r"$P(G_1|\textbf{D})$")
    plt.show()
    
    success_95 = np.sum((result[:,0] >= 0.95)) / 10000
    success_80 = np.sum((result[:,0] >= 0.95)) / 10000
    failure_50 = np.sum((result[:,0] <= 0.5)) / 10000
    
    mn = np.mean(result[:,0])
    var = np.var(result[:,0])
    
    quantiles = np.quantile(result[:,0], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

    return success_95, success_80, failure_50, mn, var, quantiles

success_95_0, success_80_0, failure_50_0, mn_0, var_0, quantiles_0 = evaluate(result_0)
success_95_1, success_80_1, failure_50_1, mn_1, var_1, quantiles_1 = evaluate(result_1)
success_95_2, success_80_2, failure_50_2, mn_2, var_2, quantiles_2 = evaluate(result_2)
success_95_3, success_80_3, failure_50_3, mn_3, var_3, quantiles_3 = evaluate(result_3)
success_95_4, success_80_4, failure_50_4, mn_4, var_4, quantiles_4 = evaluate(result_4)
success_95_5, success_80_5, failure_50_5, mn_5, var_5, quantiles_5 = evaluate(result_5)
success_95_6, success_80_6, failure_50_6, mn_6, var_6, quantiles_6 = evaluate(result_6)
success_95_7, success_80_7, failure_50_7, mn_7, var_7, quantiles_7 = evaluate(result_7)
success_95_8, success_80_8, failure_50_8, mn_8, var_8, quantiles_8 = evaluate(result_8)
success_95_9, success_80_9, failure_50_9, mn_9, var_9, quantiles_9 = evaluate(result_9)
success_95_10, success_80_10, failure_50_10, mn_10, var_10, quantiles_10 = evaluate(result_10)


ProbGxtoy_givenD = [x / (x + y) for x,y in zip(probG_givenD_isproportional_xtoy_1, probG_givenD_isproportional_ytox_1)]   


