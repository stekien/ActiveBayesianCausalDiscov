# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 10:25:53 2022

@author: Stefan Kienle (stefan.kienle@tum.de)
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel, Matern
from scipy.stats import norm, uniform#,chi2, gamma, lognorm
import matplotlib.pyplot as plt
import matplotlib as mpl
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
# 
#from joblib import Parallel, delayed

# plot params
params = {
    "legend.fontsize": 13.5,
    "figure.figsize": (8, 10),
    "figure.dpi": 100,
    "axes.labelsize": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "lines.linewidth": 3,
    "lines.markeredgewidth": 1.5,
    "lines.markersize": 5,
    "lines.marker": "o",
    "patch.edgecolor": "black",
}

plt.rcParams.update(params)
plt.style.use("seaborn")
#mpl.rcParams['text.usetex'] = True


np.random.seed(seed=13)
nuu = 1.5

# data is generated synthetically in this script
rng = np.random.RandomState(13)
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

#plt.style.use('seaborn')
plt.scatter(X_train, y_train_noisy, color="black", label="$Observations$")
plt.xlabel("$X$")
plt.ylabel("$Y$")
plt.legend()
plt.show()

def pobD_givenG(X, Y, gamma, sigma_x, sigma_y):
    
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=gamma, length_scale_bounds="fixed", nu=nuu)
       
    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, alpha=sigma_y 
    )
    # fit the model to the training data
    gaussian_process.fit(X, Y)
    
    res = gaussian_process.log_marginal_likelihood_value_ + np.sum(np.log(norm.pdf(X.reshape(-1,), scale=sigma_x))) 
          
    return np.exp(res)


def pobD_givenG_int(X, Y, gamma, sigma_x, sigma_y, X_interv_x=[], Y_interv_x=[], X_interv_y=[], Y_interv_y=[]):
    
    kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=gamma, length_scale_bounds="fixed", nu=nuu)
       
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

# initialise interventional data
# intervention on x
X_interv_x = np.random.uniform(-3,3,[1,1])
Y_interv_x = np.array([2 * np.tanh(X_interv_x[0]) + rng.normal(loc=0.0, scale=noise_std, size=1)]).reshape(1,)

#X_interv_x, Y_interv_x = rnd_interv_x(X_interv_x, Y_interv_x)

# intervention on y
X_interv_y = rng.normal(loc=0.0, scale=1, size=1).reshape(-1,1)
Y_interv_y = np.random.uniform(-2.1,2.1,[1,1]).reshape(1,)

#X_interv_y , Y_interv_y  = rnd_interv_y(X_interv_y, Y_interv_y)   

# set parameters for the two graphs respectively 
gamma_g1 = 1.75
sigma_x_g1 = 1
sigma_y_g1 = np.sqrt(0.1)

gamma_g2 = 1.75
sigma_x_g2 = 1
sigma_y_g2 = np.sqrt(0.1)

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

x = range(0,len(ProbGytox_givenD))

params = {
    "legend.fontsize": 25,
    "axes.titlesize": 30,
    "figure.figsize": (12, 14),
    "figure.dpi": 100,
    "axes.labelsize": 40,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "lines.linewidth": 5,
    "lines.markeredgewidth": 2.5,
    "lines.markersize": 10,
    "lines.marker": "o",
    "patch.edgecolor": "black",
    "text.usetex": True,
    "font.family": "serif"
}
#plt.style.use('seaborn')
plt.rcParams.update(params)
plt.subplot(1, 2, 1)
plt.ylim(0, 1)
plt.xlim(0,10)
plt.xticks(list(range(11)))
plt.plot(x, ProbGxtoy_givenD, '-.', label=r"$P(G_1|\textbf{D})$", color="green")
plt.legend()
#plt.xlabel("Intervention")
plt.ylabel("Probability")
plt.xlabel("Intervention")
_ = plt.title("$G_1$ : " + r"$(X_1 \rightarrow X_2)$")

plt.subplot(1, 2, 2)
plt.rcParams.update(params)
plt.ylim(0, 1)
plt.xlim(0,10)
plt.xticks(list(range(11)))
plt.plot(x, ProbGytox_givenD, '-.', label=r"$P(G_2|\textbf{D})$", color="red")
plt.legend()
plt.xlabel("Intervention")
#plt.ylabel("$Probability$")
_ = plt.title("$G_2$ : " + r"$(X_1 \leftarrow X_2)$")
plt.show()

# plot params
params = {
    "legend.fontsize": 25,
    "axes.titlesize": 40,
    "figure.figsize": (12, 14),
    "figure.dpi": 100,
    "axes.labelsize": 40,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "lines.linewidth": 5,
    "lines.markeredgewidth": 7,
    "lines.markersize": 20,
    "lines.marker": "o",
    "patch.edgecolor": "black",
    "text.usetex": True,
    "font.family": "serif"
}
plt.rcParams.update(params)
plt.scatter(X_train, y_train_noisy, color="black", label="Initial obs.")
try:
    plt.scatter(X_interv_x, Y_interv_x, color="blue", label=r"$do(X_1=x_1)$")
except:
    print('No interventions on X.')
try:    
    plt.scatter(X_interv_y, Y_interv_y, color="green", label=r"$do(X_2=x_2)$")
except:
    print('No interventions on Y.')
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
_ = plt.title("Initial and Intervention Observations")
plt.legend()
plt.show()


def plot_xtoy(fitted_GP_object, X_predict, X, Y):
    # plot params
    params = {
        "legend.fontsize": 25,
        "axes.titlesize": 40,
        "figure.figsize": (12, 14),
        "figure.dpi": 100,
        "axes.labelsize": 40,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "lines.linewidth": 5,
        "lines.markeredgewidth": 5,
        "lines.markersize": 10,
        "lines.marker": "o",
        "patch.edgecolor": "black",
        "text.usetex": True,
        "font.family": "serif"
    }
    plt.rcParams.update(params)
    mean_prediction, std_prediction = fitted_GP_object.predict(X_predict, return_std=True)
    plt.errorbar(
        X,
        Y,
        noise_std,
        linestyle="None",
        color="tab:blue",
        marker=".",
        markersize=10,
        label="Observations",
    )
    plt.plot(X_predict, mean_prediction, label="Mean prediction", markersize=1)
    plt.fill_between(
        X_predict.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"95\% confidence interval",
    )
    plt.legend(loc = 'lower right')
    plt.xlabel("$X_1$")
    plt.ylabel("$X_2$")
    _ = plt.title(r"$G_1 : (X_1 \rightarrow X_2)$")
    plt.show()
    return

def plot_ytox(fitted_GP_object, X_predict, X, Y):
    # plot params
    params = {
        "legend.fontsize": 25,
        "axes.titlesize": 40,
        "figure.figsize": (12, 14),
        "figure.dpi": 100,
        "axes.labelsize": 40,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "lines.linewidth": 5,
        "lines.markeredgewidth": 5,
        "lines.markersize": 10,
        "lines.marker": "o",
        "patch.edgecolor": "black",
        "text.usetex": True,
        "font.family": "serif"
    }
    plt.rcParams.update(params)
    mean_prediction, std_prediction = fitted_GP_object.predict(X_predict, return_std=True)
    plt.errorbar(
        X,
        Y,
        noise_std,
        linestyle="None",
        color="tab:blue",
        marker=".",
        markersize=10,
        label="Observations",
    )
    plt.plot(X_predict, mean_prediction, label="Mean prediction", markersize=1)
    plt.fill_between(
        X_predict.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        color="tab:orange",
        alpha=0.5,
        label=r"95\% confidence interval",
    )
    plt.legend(loc = 'lower right')
    plt.xlabel("$X_2$")
    plt.ylabel("$X_1$")
    _ = plt.title(r"$G_2 : (X_1 \leftarrow X_2)$")
    plt.show()
    return



# check what perfect fit would be
k = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=gamma_g1, length_scale_bounds="fixed", nu=nuu)
k1 = ConstantKernel(1.0) * RBF(length_scale=gamma_g1)       
gp = GaussianProcessRegressor(
     kernel=k, alpha=sigma_y_g1 
)
# fit the model to the training data
try:
    X_GPr = np.concatenate([X_train, X_interv_x])
    Y_GPr = np.concatenate([y_train_noisy, Y_interv_x])
except:
    X_GPr = X_train
    Y_GPr = y_train_noisy
gp.fit(X_GPr, Y_GPr)
try:
    plot_xtoy(gp, X_predict.reshape(-1, 1), np.concatenate([X_train, X_interv_x]),  np.concatenate([y_train_noisy, Y_interv_x]))
except:
    plot_xtoy(gp, X_predict.reshape(-1, 1), X_train, y_train_noisy)
"""
gp1 = GaussianProcessRegressor(
     kernel=k1, alpha=sigma_y_g1,n_restarts_optimizer=9 
)
# fit the model to the training data
gp1.fit(np.concatenate([X_train, X_interv_x]), np.concatenate([y_train_noisy, Y_interv_x]))
plot_xtoy(gp1, X_predict.reshape(-1, 1), np.concatenate([X_train, X_interv_x]),  np.concatenate([y_train_noisy, Y_interv_x]))

#gaussian_process.fit(Y_train.reshape(-1,1), X_train.reshape(-1,))
#plot_ytox(gaussian_process, y_predict, Y_train.reshape(-1,1), X_train.reshape(-1,))
"""
k2 = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=gamma_g2, length_scale_bounds="fixed", nu=nuu)
       
gp2 = GaussianProcessRegressor(
     kernel=k2, alpha=sigma_x_g2 
)
# fit the model to the training data
try:
    X_GPr2 = np.concatenate([y_train_noisy, Y_interv_y]).reshape(-1,1)
    Y_GPr2 = np.concatenate([X_train, X_interv_y]).reshape(-1,)
except:
    X_GPr2 = y_train_noisy.reshape(-1,1)
    Y_GPr2 = X_train.reshape(-1,)
     
gp2.fit(X_GPr2, Y_GPr2)
try:
    plot_ytox(gp2, y_predict, np.concatenate([y_train_noisy, Y_interv_y]).reshape(-1,1), np.concatenate([X_train, X_interv_y]).reshape(-1,))
except:
    plot_ytox(gp2, y_predict, y_train_noisy.reshape(-1,1), X_train.reshape(-1,))

