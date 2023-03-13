# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:31:06 2022
@author: Stefan Kienle (stefan.kienle@tum.de)
"""
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel, Matern
from scipy.stats import norm, uniform#,chi2, gamma, lognorm
import matplotlib.pyplot as plt
from skopt import gp_minimize
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
# 
from joblib import Parallel, delayed
rng = np.random.RandomState(714)

class Bivariate(object):
    
    def __init__(self, X1_obs, X2_obs, gamma_g1, sigma_x1_g1, sigma_x2_g1, gamma_g2, sigma_x1_g2, sigma_x2_g2, sigma_x1_g3, sigma_x2_g3, nu_matern, const_ker, fun):
        
        self.X1_obs = X1_obs
        self.X2_obs = X2_obs
        self.gamma_g1 = gamma_g1
        self.sigma_x1_g1 = sigma_x1_g1
        self.sigma_x2_g1 = sigma_x2_g1
        self.gamma_g2 = gamma_g2
        self.sigma_x1_g2 = sigma_x1_g2
        self.sigma_x2_g2 = sigma_x2_g2
        self.sigma_x1_g3 = sigma_x1_g3
        self.sigma_x2_g3 = sigma_x2_g3
        self.nu_matern = nu_matern
        self.const_ker = const_ker
        self.fun = fun
        self.C_G1 = []
        self.C_G2 = []
        self.C_G3 = []
        self.IntProb_G1 = [1]
        self.IntProb_G2 = [1]
        self.IntProb_G3 = [1]
        
        
    def pobD_givenG1(self):
        
        kernel = ConstantKernel(self.const_ker, constant_value_bounds="fixed") * Matern(length_scale=self.gamma_g1, length_scale_bounds="fixed", nu=self.nu_matern)
           
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, alpha=self.sigma_x2_g1 
        )
        # fit the model to the training data
        gaussian_process.fit(self.X1_obs, self.X2_obs)
        
        res = gaussian_process.log_marginal_likelihood_value_ + np.sum(np.log(norm.pdf(self.X1_obs.reshape(-1,), scale=self.sigma_x1_g1)))
        
        self.C_G1.append(np.exp(res))
              
        return #np.exp(res)
    
    
    def pobD_givenG2(self):
        
        kernel = ConstantKernel(self.const_ker, constant_value_bounds="fixed") * Matern(length_scale=self.gamma_g2, length_scale_bounds="fixed", nu=self.nu_matern)
           
        gaussian_process = GaussianProcessRegressor(
            kernel=kernel, alpha=self.sigma_x1_g2 
        )
        # fit the model to the training data
        gaussian_process.fit(self.X2_obs.reshape(-1,1), self.X1_obs.reshape(-1,))
        
        res = gaussian_process.log_marginal_likelihood_value_ + np.sum(np.log(norm.pdf(self.X2_obs.reshape(-1,), scale=self.sigma_x2_g2))) 
        
        self.C_G2.append(np.exp(res))
              
        return #np.exp(res)
    
    
    def pobD_givenG3(self):
        
        res = np.sum(np.log(norm.pdf(self.X2_obs.reshape(-1,), scale=self.sigma_x2_g3))) + np.sum(np.log(norm.pdf(self.X1_obs.reshape(-1,), scale=self.sigma_x1_g3)))
        
        self.C_G3.append(np.exp(res))
              
        return #np.exp(res) 
    
    
    def pobInt_X1_givenF_G1(self, X1_interv_x1=[], X2_interv_x1=[]):
        
        kernel = ConstantKernel(self.const_ker, constant_value_bounds="fixed") * Matern(length_scale=self.gamma_g1, length_scale_bounds="fixed", nu=self.nu_matern)
           
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.sigma_x2_g1 
        )
        # fit the model to the training data
        try:
            X1_GPr = np.concatenate([self.X1_obs, X1_interv_x1[0:-1]])
            X2_GPr = np.concatenate([self.X2_obs, X2_interv_x1[0:-1]])
            gp.fit(X1_GPr.reshape(-1,1), X2_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(X1_interv_x1[-1].reshape(-1,1), return_std=True)
            res = norm.pdf(X2_interv_x1[-1].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int[0]**2))
            res = res[0]
        except:
            try:
                X1_GPr = self.X1_obs
                X2_GPr = self.X2_obs
                gp.fit(X1_GPr.reshape(-1,1), X2_GPr.reshape(-1,))
                mu_int, std_sig_int = gp.predict(X1_interv_x1[-1].reshape(-1,1), return_std=True)
                res = norm.pdf(X2_interv_x1[-1].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int[0]**2))
                res = res[0]
            except:  
                print('ErrorG1: No Intervention on X1 Data provided. Function returned 1.\n')
                res = 1
            
        self.IntProb_G1.append(res)

        return #res
    
    def pobInt_X1_given_G2(self, X1_interv_x1=[], X2_interv_x1=[]):

        try:
            res = norm.pdf(X2_interv_x1[-1].reshape(-1,), loc=0, scale=self.sigma_x2_g2)
            res = res[0]
        except:
            print('ErrorG2: No Intervention on X1 Data provided. Function returned 1.\n')
            res = 1
            
        self.IntProb_G2.append(res)

        return #res
    
    def pobInt_X1_given_G3(self, X1_interv_x1=[], X2_interv_x1=[]):

        try:
            res = norm.pdf(X2_interv_x1[-1].reshape(-1,), loc=0, scale=self.sigma_x2_g3)
            res = res[0]
        except:
            print('ErrorG3: No Intervention on X1 Data provided. Function returned 1.\n')
            res = 1
            
        self.IntProb_G3.append(res)

        return #res
    
    def pobInt_X2_givenF_G2(self, X1_interv_x2=[], X2_interv_x2=[]):
        
        kernel = ConstantKernel(self.const_ker, constant_value_bounds="fixed") * Matern(length_scale=self.gamma_g2, length_scale_bounds="fixed", nu=self.nu_matern)
           
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.sigma_x1_g2 
        )
        try:
            X1_GPr = np.concatenate([self.X1_obs, X1_interv_x2[0:-1]])
            X2_GPr = np.concatenate([self.X2_obs, X2_interv_x2[0:-1]])
            gp.fit(X2_GPr.reshape(-1,1), X1_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(X2_interv_x2[-1].reshape(-1,1), return_std=True)
            res = norm.pdf(X1_interv_x2[-1].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int[0]**2))
            res = res[0]
        except:
            try:
                X1_GPr = self.X1_obs
                X2_GPr = self.X2_obs
                gp.fit(X2_GPr.reshape(-1,1), X1_GPr.reshape(-1,))
                mu_int, std_sig_int = gp.predict(X2_interv_x2[-1].reshape(-1,1), return_std=True)
                res = norm.pdf(X1_interv_x2[-1].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int[0]**2))
                res = res[0]
            except:
                print('ErrorG2: No Intervention on X2 Data provided. Function returned 1.\n')
                res = 1
            
        self.IntProb_G2.append(res)

        return #res
    
    def pobInt_X2_given_G1(self, X1_interv_x2=[], X2_interv_x2=[]):

        try:
            res = norm.pdf(X1_interv_x2[-1].reshape(-1,), loc=0, scale=self.sigma_x1_g1)
            res = res[0]
        except:
            print('ErrorG1: No Intervention on X2 Data provided. Function returned 1.\n')
            res = 1
            
        self.IntProb_G1.append(res)

        return #res
    
    def pobInt_X2_given_G3(self, X1_interv_x2=[], X2_interv_x2=[]):

        try:
            res = norm.pdf(X1_interv_x2[-1].reshape(-1,), loc=0, scale=self.sigma_x1_g3)
            res = res[0]
        except:
            print('ErrorG3: No Intervention on X2 Data provided. Function returned 1.\n')
            res = 1
            
        self.IntProb_G3.append(res)

        return
    
    def f_int_x1(self, x, X1_interv_x1=[], X2_interv_x1=[]):
        
        n = 500
        kernel = ConstantKernel(self.const_ker, constant_value_bounds="fixed") * Matern(length_scale=self.gamma_g1, length_scale_bounds="fixed", nu=self.nu_matern)
           
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.sigma_x2_g1 
        )
        # fit the model to the training data
        try:
            X1_GPr = np.concatenate([self.X1_obs, X1_interv_x1])
            X2_GPr = np.concatenate([self.X2_obs, X2_interv_x1])
            gp.fit(X1_GPr.reshape(-1,1), X2_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(x.reshape(-1,1), return_std=True)
            sample_interv_g1 = np.random.normal(loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2), size=n) 
        except:
            X1_GPr = self.X1_obs
            X2_GPr = self.X2_obs
            gp.fit(X1_GPr.reshape(-1,1), X2_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(x.reshape(-1,1), return_std=True)
            sample_interv_g1 = np.random.normal(loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2), size=n) 
            
        # G_1
        def calc1(j):      
            res = -np.log(norm.pdf(sample_interv_g1[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2))*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g2)*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3)) 
            return res
        
        warnings.filterwarnings('ignore')
        summe1 = Parallel(n_jobs=6, batch_size=128)(delayed(calc1)(i) for i in range(n))  
        summe1 = (1/n) * np.sum(summe1)
        
        # G_2
        sample_interv_g2 = np.random.normal(loc=0.0, scale=self.sigma_x2_g2, size=n)
        def calc2(j):
            res = -np.log(norm.pdf(sample_interv_g2[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2))*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g2)*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3)) 
            return res
        
        warnings.filterwarnings('ignore')
        summe2 = Parallel(n_jobs=6, batch_size=128)(delayed(calc2)(i) for i in range(n)) 
        summe2 = (1/n) * np.sum(summe2)
        
        # G_3
        sample_interv_g3 = np.random.normal(loc=0.0, scale=self.sigma_x2_g3, size=n)
        def calc3(j):
            res = -np.log(norm.pdf(sample_interv_g3[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2))*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g2)*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3)) 
            return res
        
        warnings.filterwarnings('ignore')
        summe3 = Parallel(n_jobs=6, batch_size=128)(delayed(calc3)(i) for i in range(n)) 
        summe3 = (1/n) * np.sum(summe3)
        
        #first part of sum
        summe4 = -np.log(np.sqrt(2*np.pi*(self.sigma_x2_g1**2 + std_sig_int[0]**2)))
        
        #print('f_int_x1:', summe1 ,summe2,summe3 , summe4)
        res = (summe1 + summe2 + summe3 + summe4)#[0]
        
        return res
    
    def g_int_x1(self, x, X1_interv_x1=[], X2_interv_x1=[]):
        
        n = 5000
        kernel = ConstantKernel(self.const_ker, constant_value_bounds="fixed") * Matern(length_scale=self.gamma_g1, length_scale_bounds="fixed", nu=self.nu_matern)
           
        gp = GaussianProcessRegressor(
            kernel=kernel, alpha=self.sigma_x2_g1 
        )
        # fit the model to the training data
        try:
            X1_GPr = np.concatenate([self.X1_obs, X1_interv_x1])
            X2_GPr = np.concatenate([self.X2_obs, X2_interv_x1])
            gp.fit(X1_GPr.reshape(-1,1), X2_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(x.reshape(-1,1), return_std=True)
            sample_interv_g1 = np.random.normal(loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2), size=n) 
        except:
            X1_GPr = self.X1_obs
            X2_GPr = self.X2_obs
            gp.fit(X1_GPr.reshape(-1,1), X2_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(x.reshape(-1,1), return_std=True)
            sample_interv_g1 = np.random.normal(loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2), size=n) 
            
        # G_1
        def calc1(j):      
            res = np.log(norm.pdf(sample_interv_g1[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2))*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3)) - np.log(norm.pdf(sample_interv_g1[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2))*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g2)*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3)) 
            return res
        
        warnings.filterwarnings('ignore')
        summe1 = Parallel(n_jobs=6, batch_size=128)(delayed(calc1)(i) for i in range(n))  
        summe1 = (1/n) * np.sum(summe1)
        
        # G_2
        sample_interv_g2 = np.random.normal(loc=0.0, scale=self.sigma_x2_g2, size=n)
        def calc2(j):
            res = np.log(norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g2)*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3)) - np.log(norm.pdf(sample_interv_g2[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2))*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g2)*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3)) 
            return res
        
        warnings.filterwarnings('ignore')
        summe2 = Parallel(n_jobs=6, batch_size=128)(delayed(calc2)(i) for i in range(n)) 
        summe2 = (1/n) * np.sum(summe2)
        
        # G_3
        sample_interv_g3 = np.random.normal(loc=0.0, scale=self.sigma_x2_g3, size=n)
        def calc3(j):
            res = np.log(norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3)) -np.log(norm.pdf(sample_interv_g3[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2))*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g2)*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x2_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3)) 
            return res
        
        warnings.filterwarnings('ignore')
        summe3 = Parallel(n_jobs=6, batch_size=128)(delayed(calc3)(i) for i in range(n)) 
        summe3 = (1/n) * np.sum(summe3)
        
        #print('f_int_x1:', summe1 ,summe2,summe3 , summe4)
        res = (summe1 + summe2 + summe3)#[0]
        
        return res
    

    def f_int_x2(self, y, X1_interv_x2=[], X2_interv_x2=[]):
        
        n = 500
        k = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=self.gamma_g2, length_scale_bounds="fixed", nu=self.nu_matern)
           
        gp = GaussianProcessRegressor(
            kernel=k, alpha=self.sigma_x1_g2
        )
        # fit the model to the training data
        # fit the model to the training data
        try:
            X1_GPr = np.concatenate([self.X1_obs, X1_interv_x2])
            X2_GPr = np.concatenate([self.X2_obs, X2_interv_x2])
            gp.fit(X2_GPr.reshape(-1,1), X1_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(y.reshape(-1,1), return_std=True)
            sample_interv_g2 = np.random.normal(loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2), size=n) 
        except:
            X1_GPr = self.X1_obs
            X2_GPr = self.X2_obs
            gp.fit(X2_GPr.reshape(-1,1), X1_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(y.reshape(-1,1), return_std=True)
            sample_interv_g2 = np.random.normal(loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2), size=n) 
        
        # G_1
        sample_interv_g1 = np.random.normal(loc=0.0, scale=self.sigma_x1_g1, size=n)            
        def calc1(j): 
            res = -np.log(norm.pdf(sample_interv_g1[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2))*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g1)*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3))
            return res
        
        warnings.filterwarnings('ignore')
        summe1 = Parallel(n_jobs=6, batch_size=128)(delayed(calc1)(i) for i in range(n))  
        summe1 = (1/n) * np.sum(summe1)
        
        # G_2 cont.       
        def calc2(j):
            res = -np.log(norm.pdf(sample_interv_g2[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2))*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g1)*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3))
            return res
        
        warnings.filterwarnings('ignore')
        summe2 = Parallel(n_jobs=6, batch_size=128)(delayed(calc2)(i) for i in range(n)) 
        summe2 = (1/n) * np.sum(summe2)
        
        # G_3
        sample_interv_g3 = np.random.normal(loc=0.0, scale=self.sigma_x1_g3, size=n) 
        def calc3(j): 
            res = -np.log(norm.pdf(sample_interv_g3[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2))*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g1)*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3))
            return res
        
        warnings.filterwarnings('ignore')
        summe3 = Parallel(n_jobs=6, batch_size=128)(delayed(calc3)(i) for i in range(n))  
        summe3 = (1/n) * np.sum(summe3)
        
        #first part of sum
        summe4 = -np.log(np.sqrt(2*np.pi*(self.sigma_x1_g2**2 + std_sig_int[0]**2)))
        #print('f_int_x2:', summe1 ,summe2,summe3 , summe4)
        res = (summe1 + summe2 + summe3 + summe4)#[0]
        
        return res
    
    def g_int_x2(self, y, X1_interv_x2=[], X2_interv_x2=[]):
        
        n = 5000
        k = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(length_scale=self.gamma_g2, length_scale_bounds="fixed", nu=self.nu_matern)
           
        gp = GaussianProcessRegressor(
            kernel=k, alpha=self.sigma_x1_g2
        )
        # fit the model to the training data
        # fit the model to the training data
        try:
            X1_GPr = np.concatenate([self.X1_obs, X1_interv_x2])
            X2_GPr = np.concatenate([self.X2_obs, X2_interv_x2])
            gp.fit(X2_GPr.reshape(-1,1), X1_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(y.reshape(-1,1), return_std=True)
            sample_interv_g2 = np.random.normal(loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2), size=n) 
        except:
            X1_GPr = self.X1_obs
            X2_GPr = self.X2_obs
            gp.fit(X2_GPr.reshape(-1,1), X1_GPr.reshape(-1,))
            mu_int, std_sig_int = gp.predict(y.reshape(-1,1), return_std=True)
            sample_interv_g2 = np.random.normal(loc=mu_int, scale=np.sqrt(self.sigma_x2_g1**2 + std_sig_int**2), size=n) 
        
        # G_1
        sample_interv_g1 = np.random.normal(loc=0.0, scale=self.sigma_x1_g1, size=n)            
        def calc1(j): 
            res = np.log(norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g1)*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3)) -np.log(norm.pdf(sample_interv_g1[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2))*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g1)*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g1[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3))
            return res
        
        warnings.filterwarnings('ignore')
        summe1 = Parallel(n_jobs=6, batch_size=128)(delayed(calc1)(i) for i in range(n))  
        summe1 = (1/n) * np.sum(summe1)
        
        # G_2 cont.       
        def calc2(j):
            res = np.log(norm.pdf(sample_interv_g2[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2))*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3)) -np.log(norm.pdf(sample_interv_g2[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2))*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g1)*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g2[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3))
            return res
        
        warnings.filterwarnings('ignore')
        summe2 = Parallel(n_jobs=6, batch_size=128)(delayed(calc2)(i) for i in range(n)) 
        summe2 = (1/n) * np.sum(summe2)
        
        # G_3
        sample_interv_g3 = np.random.normal(loc=0.0, scale=self.sigma_x1_g3, size=n) 
        def calc3(j): 
            res = np.log(norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3)) - np.log(norm.pdf(sample_interv_g3[j].reshape(-1,), loc=mu_int, scale=np.sqrt(self.sigma_x1_g2**2 + std_sig_int**2))*self.C_G2[0]*np.prod(self.IntProb_G2)*(1/3) + norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g1)*self.C_G1[0]*np.prod(self.IntProb_G1)*(1/3) + norm.pdf(sample_interv_g3[j].reshape(-1,), loc=0.0, scale=self.sigma_x1_g3)*self.C_G3[0]*np.prod(self.IntProb_G3)*(1/3))
            return res
        
        warnings.filterwarnings('ignore')
        summe3 = Parallel(n_jobs=6, batch_size=128)(delayed(calc3)(i) for i in range(n))  
        summe3 = (1/n) * np.sum(summe3)
        
        res = (summe1 + summe2 + summe3)#[0]
        
        return res
    
    def optimal_int(self, num_interv, interv_x, interv_y):
        
        probG_givenD_isproportional_g1 = []
        probG_givenD_isproportional_g2 = []
        probG_givenD_isproportional_g3 = []
        self.pobD_givenG1()
        self.pobD_givenG2()
        self.pobD_givenG3()
        print('C_G1: ',self.C_G1)
        print('C_G2: ',self.C_G2)
        print('C_G3: ',self.C_G3)
        probG_givenD_isproportional_g1.append(self.C_G1[0])
        probG_givenD_isproportional_g2.append(self.C_G2[0])
        probG_givenD_isproportional_g3.append(self.C_G3[0])
        
        for t in range(num_interv):

            try:
                f_x = lambda h: -self.f_int_x1(x=np.array(h), X1_interv_x1=X1_interv_x1, X2_interv_x1=X2_interv_x1)

                resx = gp_minimize(f_x,                  # the function to minimize
                                  [(-3.0, 3.0)],      # the bounds on each dimension of x
                                  acq_func="gp_hedge",      # the acquisition function
                                  n_calls=15,         # the number of evaluations of f
                                  n_random_starts=5,
                                  n_jobs=6)  # the number of random initialization points
                                  #noise=0.1**2,       # the noise level (optional)
                                  #random_state=9417)   # the random seed
                                  
                g_x1 = -self.g_int_x1(x = np.array(resx.x[0]), X1_interv_x1=X1_interv_x1, X2_interv_x1=X2_interv_x1)
                
                fun_g_x1 = lambda h: -self.g_int_x1(x = np.array(h), X1_interv_x1=X1_interv_x1, X2_interv_x1=X2_interv_x1)
                                  
                f_y = lambda x: -self.f_int_x2(y=np.array(x), X1_interv_x2=X1_interv_x2, X2_interv_x2=X2_interv_x2)

                resy = gp_minimize(f_y,                  # the function to minimize
                                  [(-3.0, 3.0)],      # the bounds on each dimension of x
                                  acq_func="gp_hedge",      # the acquisition function
                                  n_calls=15,         # the number of evaluations of f
                                  n_random_starts=5,
                                  n_jobs=6)  # the number of random initialization points
                                  #noise=0.1**2,       # the noise level (optional)
                                  #random_state=9417)   # the random seed
                                  
                g_x2 = -self.g_int_x2(y = np.array(resy.x[0]), X1_interv_x2=X1_interv_x2, X2_interv_x2=X2_interv_x2)
                
                fun_g_x2 = lambda h: -self.g_int_x2(y = np.array(h), X1_interv_x2=X1_interv_x2, X2_interv_x2=X2_interv_x2)
                print('Case 1')
            except:
                try:
                    f_x = lambda h: -self.f_int_x1(x=np.array(h), X1_interv_x1=X1_interv_x1, X2_interv_x1=X2_interv_x1)
            
                    resx = gp_minimize(f_x,                  # the function to minimize
                                      [(-3.0, 3.0)],      # the bounds on each dimension of x
                                      acq_func="gp_hedge",      # the acquisition function
                                      n_calls=15,         # the number of evaluations of f
                                      n_random_starts=5,
                                      n_jobs=6)  # the number of random initialization points
                                      #noise=0.1**2,       # the noise level (optional)
                                      #random_state=9417)   # the random seed
                                      
                    g_x1 = -self.g_int_x1(x = np.array(resx.x[0]), X1_interv_x1=X1_interv_x1, X2_interv_x1=X2_interv_x1)
                    
                    fun_g_x1 = lambda h: -self.g_int_x1(x = np.array(h), X1_interv_x1=X1_interv_x1, X2_interv_x1=X2_interv_x1)
                                      
                    f_y = lambda x: -self.f_int_x2(y=np.array(x))
            
                    resy = gp_minimize(f_y,                  # the function to minimize
                                      [(-3.0, 3.0)],      # the bounds on each dimension of x
                                      acq_func="gp_hedge",      # the acquisition function
                                      n_calls=15,         # the number of evaluations of f
                                      n_random_starts=5,
                                      n_jobs=6)  # the number of random initialization points
                                      #noise=0.1**2,       # the noise level (optional)
                                      #random_state=9417)   # the random seed
                                      
                    g_x2 = -self.g_int_x2(y = np.array(resy.x[0]))
                    
                    fun_g_x2 = lambda h: -self.g_int_x2(y = np.array(h))
                                      
                    print('Case 2')
                except:
                   try:
                       f_x = lambda h: -self.f_int_x1(x=np.array(h))

                       resx = gp_minimize(f_x,                  # the function to minimize
                                         [(-3.0, 3.0)],      # the bounds on each dimension of x
                                         acq_func="gp_hedge",      # the acquisition function
                                         n_calls=15,         # the number of evaluations of f
                                         n_random_starts=5,
                                         n_jobs=6)  # the number of random initialization points
                                         #noise=0.1**2,       # the noise level (optional)
                                         #random_state=9417)   # the random seed
                                         
                       g_x1 = -self.g_int_x1(x = np.array(resx.x[0]))
                       
                       fun_g_x1 = lambda h: -self.g_int_x1(x = np.array(h))
                                         
                       f_y = lambda x: -self.f_int_x2(y=np.array(x), X1_interv_x2=X1_interv_x2, X2_interv_x2=X2_interv_x2)

                       resy = gp_minimize(f_y,                  # the function to minimize
                                         [(-3.0, 3.0)],      # the bounds on each dimension of x
                                         acq_func="gp_hedge",      # the acquisition function
                                         n_calls=15,         # the number of evaluations of f
                                         n_random_starts=5,
                                         n_jobs=6)  # the number of random initialization points
                                         #noise=0.1**2,       # the noise level (optional)
                                         #random_state=9417)   # the random seed
                                         
                       g_x2 = -self.g_int_x2(y = np.array(resy.x[0]), X1_interv_x2=X1_interv_x2, X2_interv_x2=X2_interv_x2)
                       
                       fun_g_x2 = lambda h: -self.g_int_x2(y = np.array(h), X1_interv_x2=X1_interv_x2, X2_interv_x2=X2_interv_x2)
                                         
                       print('Case 3')
                   except:
                      f_x = lambda h: -self.f_int_x1(x=np.array(h))
                      #print(f_x(1))

                      resx = gp_minimize(f_x,                  # the function to minimize
                                        [(-3.0, 3.0)],      # the bounds on each dimension of x
                                        acq_func="gp_hedge",      # the acquisition function
                                        n_calls=15,         # the number of evaluations of f
                                        n_random_starts=5,
                                        n_jobs=6)  # the number of random initialization points
                                        #noise=0.1**2,       # the noise level (optional)
                                        #random_state=9417)   # the random seed
                                        
                      g_x1 = -self.g_int_x1(x = np.array(resx.x[0]))
                      
                      fun_g_x1 = lambda h: -self.g_int_x1(x = np.array(h))
                                        
                      f_y = lambda h: -self.f_int_x2(y=np.array(h))

                      resy = gp_minimize(f_y,                  # the function to minimize
                                        [(-3.0, 3.0)],      # the bounds on each dimension of x
                                        acq_func="gp_hedge",      # the acquisition function
                                        n_calls=15,         # the number of evaluations of f
                                        n_random_starts=5,
                                        n_jobs=6)  # the number of random initialization points
                                        #noise=0.1**2,       # the noise level (optional)
                                        #random_state=9417)   # the random seed
                                        
                      g_x2 = -self.g_int_x2(y = np.array(resy.x[0]))
                      
                      fun_g_x2 = lambda h: -self.g_int_x2(y = np.array(h))
                                        
                      print('Case 4')
            
            
            params = {
                "legend.fontsize": 32,
                "axes.titlesize": 45,
                "figure.figsize": (12, 14),
                "figure.dpi": 100,
                "axes.labelsize": 40,
                "xtick.labelsize": 28,
                "ytick.labelsize": 28,
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
            
            x = np.linspace(-3, 3, 100).reshape(-1, 1)
            fx = [f_x(x_i) for x_i in x]
            gx = [fun_g_x1(x_i) for x_i in x]
            #plt.plot(x, fx, "r--", label=r"$\tilde{g}_1(x_1^1)$")
            #plt.plot(x, gx, "g--", label=r"$g_1(x_1^1)$")
            #plt.legend()
            #plt.xlabel("Intervention value $x_1^1$")
            #plt.ylabel("Negative Information Gain")
            #_ = plt.title("Objective for $D_1$")
            #plt.show()
            
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel("Intervention value $x_1^1$")
            ax1.set_ylabel(r"$-\tilde{g}_1(x_1^1)$")
            ax1.plot(x, fx, "r--", label=r"$-\tilde{g}_1(x_1^1)$")
            #ax1.legend(loc=0)
            ax1.tick_params(axis='y')
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
            color = 'tab:green'
            ax2.set_ylabel("Negative Information Gain, -$g_1(x_1^1)$")  # we already handled the x-label with ax1
            ax2.plot(x, gx, "g--", label=r"$-g_1(x_1^1)$")
            #ax2.legend(loc=0)
            ax2.tick_params(axis='y')
            
            
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            fig.legend(bbox_to_anchor=(0.5, 0.5, 0.4, 0.5))
            _ = plt.title("Objective(s) for $do(X_1=x_1^1)$")
            plt.show()
            
        
            
            fy = [f_y(x_i) for x_i in x]
            gy = [fun_g_x2(x_i) for x_i in x]
            
            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel("Intervention value $x_2^2$")
            ax1.set_ylabel(r"$-\tilde{g}_2(x_2^2)$")
            ax1.plot(x, fy, "r--", label=r"$-\tilde{g}_2(x_2^2)$")
            #ax1.legend(loc=0)
            ax1.tick_params(axis='y')
            
            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            
            color = 'tab:green'
            ax2.set_ylabel("Negative Information Gain, -$g_2(x_2^2)$")  # we already handled the x-label with ax1
            ax2.plot(x, gy, "g--", label=r"$-g_2(x_2^2)$")
            #ax2.legend(loc=0)
            ax2.tick_params(axis='y')
            
            
            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            fig.legend(bbox_to_anchor=(0.5, 0.5, 0.4, 0.5))
            _ = plt.title("Objective(s) for $do(X_2=x_2^2)$")
            plt.show()
            #plt.plot(x, fy, "r--", label=r"$\tilde{g}_2(x_2^2)$")
            #plt.plot(x, gy, "g--", label=r"$g_2(x_2^2)$")
            #plt.legend()
            #plt.xlabel("Intervention value $x_2^2$")
            #plt.ylabel("Negative Information Gain")
           # _ = plt.title("Objective for $D_2$")
            #plt.show()
            
            
            
            # calculate function values for original functions
            #g_x1 = self.g_int_x1(x = np.array(resx.x[0]), X1_interv_x1=X1_interv_x1, X2_interv_x1=X2_interv_x1)
            #g_x2 = self.g_int_x2(y = np.array(resy.x[0]), X1_interv_x2=X1_interv_x2, X2_interv_x2=X2_interv_x2)

            #print('Max_interv_x1: ', resx.fun, 'Max_interv_x2: ',resy.fun)
            print('Max_interv_x1: ', g_x1, 'Max_interv_x2: ',g_x2 , "Intervention on X1:", (g_x2 > g_x1))
            # (resy.fun < resx.fun)
            if (g_x2 < g_x1):
                try:
                    X1_interv_x2, X2_interv_x2 = interv_y(np.array(resy.x[0]).reshape(1,), X1_interv_x2, X2_interv_x2)
                    self.pobInt_X2_givenF_G2(X1_interv_x2, X2_interv_x2)
                    self.pobInt_X2_given_G1(X1_interv_x2, X2_interv_x2)
                    self.pobInt_X2_given_G3(X1_interv_x2, X2_interv_x2)
                    try:
                        print('X1_interv_x1: ',X1_interv_x1, 'X2_interv_x1: ',X2_interv_x1, 'X1_interv_x2: ',X1_interv_x2, 'X2_interv_x2: ',X2_interv_x2)
                        probG_givenD_isproportional_g1.append(self.C_G1[0] * np.prod(self.IntProb_G1))
                        probG_givenD_isproportional_g2.append(self.C_G2[0] * np.prod(self.IntProb_G2))
                        probG_givenD_isproportional_g3.append(self.C_G3[0] * np.prod(self.IntProb_G3))
                    except:
                        print('X1_interv_x2: ',X1_interv_x2, 'X2_interv_x2: ',X2_interv_x2)
                        probG_givenD_isproportional_g1.append(self.C_G1[0] * np.prod(self.IntProb_G1))
                        probG_givenD_isproportional_g2.append(self.C_G2[0] * np.prod(self.IntProb_G2))
                        probG_givenD_isproportional_g3.append(self.C_G3[0] * np.prod(self.IntProb_G3))
                except:
                    # intervention on y
                    X1_interv_x2 = rng.normal(loc=0.0, scale=1, size=1).reshape(-1,1)
                    X2_interv_x2 = np.array(resy.x[0]).reshape(1,)
                    self.pobInt_X2_givenF_G2(X1_interv_x2, X2_interv_x2)
                    self.pobInt_X2_given_G1(X1_interv_x2, X2_interv_x2)
                    self.pobInt_X2_given_G3(X1_interv_x2, X2_interv_x2)
                    print('IntProb_G1: ',self.IntProb_G1)
                    print('IntProb_G2: ',self.IntProb_G2)
                    print('IntProb_G3: ',self.IntProb_G3)
                    print('X1_interv_x2: ',X1_interv_x2, 'X2_interv_x2: ',X2_interv_x2)
                    # second intervention
                    probG_givenD_isproportional_g1.append(self.C_G1[0] * np.prod(self.IntProb_G1))
                    probG_givenD_isproportional_g2.append(self.C_G2[0] * np.prod(self.IntProb_G2))
                    probG_givenD_isproportional_g3.append(self.C_G3[0] * np.prod(self.IntProb_G3))
            else:
                try:
                    X1_interv_x1, X2_interv_x1 = interv_x(np.array(resx.x[0]).reshape(-1, 1), X1_interv_x1, X2_interv_x1)
                    self.pobInt_X1_givenF_G1(X1_interv_x1, X2_interv_x1)
                    self.pobInt_X1_given_G2(X1_interv_x1, X2_interv_x1)
                    self.pobInt_X1_given_G3(X1_interv_x1, X2_interv_x1)
                    try:
                        print('X1_interv_x1: ',X1_interv_x1, 'X2_interv_x1: ',X2_interv_x1, 'X1_interv_x2: ',X1_interv_x2, 'X2_interv_x2: ',X2_interv_x2)
                        probG_givenD_isproportional_g1.append(self.C_G1[0] * np.prod(self.IntProb_G1))
                        probG_givenD_isproportional_g2.append(self.C_G2[0] * np.prod(self.IntProb_G2))
                        probG_givenD_isproportional_g3.append(self.C_G3[0] * np.prod(self.IntProb_G3))
                    except:
                        print('X1_interv_x1: ',X1_interv_x1, 'X2_interv_x1: ',X2_interv_x1)
                        probG_givenD_isproportional_g1.append(self.C_G1[0] * np.prod(self.IntProb_G1))
                        probG_givenD_isproportional_g2.append(self.C_G2[0] * np.prod(self.IntProb_G2))
                        probG_givenD_isproportional_g3.append(self.C_G3[0] * np.prod(self.IntProb_G3))
                except:
                    # initialise intervention on x
                    X1_interv_x1 = np.array(resx.x[0]).reshape(-1, 1)
                    X2_interv_x1 = np.array([self.fun(X1_interv_x1[0]) + rng.normal(loc=0.0, scale=np.sqrt(0.1), size=1)]).reshape(1,)
                    print('X1_interv_x1: ',X1_interv_x1, 'X2_interv_x1: ',X2_interv_x1)
                    # first intervention
                    self.pobInt_X1_givenF_G1(X1_interv_x1, X2_interv_x1)
                    self.pobInt_X1_given_G2(X1_interv_x1, X2_interv_x1)
                    self.pobInt_X1_given_G3(X1_interv_x1, X2_interv_x1)
                    probG_givenD_isproportional_g1.append(self.C_G1[0] * np.prod(self.IntProb_G1))
                    probG_givenD_isproportional_g2.append(self.C_G2[0] * np.prod(self.IntProb_G2))
                    probG_givenD_isproportional_g3.append(self.C_G3[0] * np.prod(self.IntProb_G3))
                        
        try:
            X1_interv_x1_out = X1_interv_x1
            X2_interv_x1_out = X2_interv_x1
        except: 
            X1_interv_x1_out = []
            X2_interv_x1_out = []
            
        try:
            X1_interv_x2_out = X1_interv_x2
            X2_interv_x2_out = X2_interv_x2
        except: 
            X1_interv_x2_out = []
            X2_interv_x2_out = []
                        
                        
        return probG_givenD_isproportional_g1, probG_givenD_isproportional_g2, probG_givenD_isproportional_g3, X1_interv_x1_out, X2_interv_x1_out, X1_interv_x2_out, X2_interv_x2_out

# global parameters
num_interv = 5
nuu = 2.5
# set parameters for the two graphs respectively | np.sqrt(0.1)
gamma_g1 = 1.75
sigma_x1_g1 = 1
sigma_x2_g1 = np.sqrt(0.1)

gamma_g2 = 1.75
sigma_x1_g2 = np.sqrt(0.1)
sigma_x2_g2 = 1

sigma_x1_g3 = 1
sigma_x2_g3 = 1

def f(x):
    y = 2 * np.tanh(x) 
    #y = 0.5*x#**2
    return y

# generate data sequence which will be predicted in the later graphs
X_predict = np.linspace(start=-3, stop=3, num=1_000).reshape(-1, 1)
y_predict = np.linspace(start=-3, stop=3, num=1_000).reshape(-1, 1)

# generate (large) sample of the underlying distribution
X = rng.normal(loc = 0.0, scale = 1, size=(1000)).reshape(-1, 1)
y = np.zeros(len(X))
for h in range(len(X)):
    y[h] = f(X[h])
y = y.reshape(1000,)

# plot params
params = {
    "legend.fontsize": 23,
    "axes.titlesize": 28,
    "figure.figsize": (12, 10),
    "figure.dpi": 100,
    "axes.labelsize": 40,
    "xtick.labelsize": 23,
    "ytick.labelsize": 23,
    "lines.linewidth": 5,
    "lines.markeredgewidth": 2.5,
    "lines.markersize": 10,
    "lines.marker": "o",
    "patch.edgecolor": "black",
}

plt.rcParams.update(params)
plt.style.use("seaborn")
#plt.style.use("seaborn-deep")

# generate initial observations 
num_initial_obs = 5
noise_std = np.sqrt(0.1)

X1_obs = rng.normal(loc=0.0, scale=1, size=num_initial_obs).tolist()
X2_obs = [f(x) + rng.normal(loc=0.0, scale=noise_std, size=None) for x in X1_obs] 

X1_obs = np.array(X1_obs).reshape(-1, 1)
X2_obs = np.array(X2_obs).reshape(-1,)

# plot data 
#plt.style.use('seaborn')
plt.scatter(X, y, label=r"$f(x)$", linestyle="dotted")
#plt.rcParams.update(params)
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
_ = plt.title("True generative process")
plt.style.use('seaborn')
plt.scatter(X1_obs, X2_obs, label=r"$noisy_initial_obs$")
plt.show()

###############################################################################
#   Interventions
###############################################################################
# methods to perform interventions
def interv_x(x, X_train, Y_train):
    x_interv = x
    y_dox = np.array([f(x_interv) + rng.normal(loc=0.0, scale=noise_std, size=1)]).reshape(1,)
    X_train_1 = np.concatenate([X_train, x_interv])  
    Y_train_1 = np.concatenate([Y_train, y_dox])
    return X_train_1, Y_train_1

def interv_y(y, X_train, Y_train):
    y_doy = y
    #y_doy = rng.uniform(min(y_train),max(y_train),[1,1]).reshape(1,)
    x_doy = rng.normal(loc=0.0, scale=1, size=1).reshape(1,1)
    X_train_2 = np.concatenate([X_train, x_doy])  
    Y_train_2 = np.concatenate([Y_train, y_doy]) 
    return X_train_2, Y_train_2

# build Bivariate model
model = Bivariate(X1_obs, X2_obs, gamma_g1, sigma_x1_g1, sigma_x2_g1, gamma_g2, sigma_x1_g2, sigma_x2_g2, sigma_x1_g3, sigma_x2_g3, nu_matern = nuu, const_ker = 1.0, fun = f)  

# performe the proposed approach
probG_givenD_isproportional_g1, probG_givenD_isproportional_g2, probG_givenD_isproportional_g3, X1_interv_x1, X2_interv_x1, X1_interv_x2, X2_interv_x2 = model.optimal_int(num_interv, interv_x, interv_y)  


# calculate the posterior probabilities assuming a flat prior for the graphs
ProbGxtoy_givenD = [x / (x + y + z) for x,y,z in zip(probG_givenD_isproportional_g1, probG_givenD_isproportional_g2,probG_givenD_isproportional_g3)]   
ProbGytox_givenD = [y / (x + y + z) for x,y,z in zip(probG_givenD_isproportional_g1, probG_givenD_isproportional_g2,probG_givenD_isproportional_g3)]   
ProbGg3_givenD = [z / (x + y + z) for x,y,z in zip(probG_givenD_isproportional_g1, probG_givenD_isproportional_g2,probG_givenD_isproportional_g3)]  


###############################################################################
#         Plots
###############################################################################
x = range(0,len(ProbGytox_givenD))
# plot params
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
plt.subplot(1, 3, 1)
plt.ylim(0, 1)
plt.xlim(0,num_interv)
plt.xticks(list(range(num_interv + 1)))
plt.plot(x, ProbGxtoy_givenD, '-.', label=r"$P(G_1|\textbf{D})$", color="green")
plt.legend()
#plt.xlabel("Intervention")
plt.ylabel("Probability")
_ = plt.title("$G_1$ : " + r"$(X_1 \rightarrow X_2)$")

plt.subplot(1, 3, 2)
plt.rcParams.update(params)
plt.ylim(0, 1)
plt.xlim(0,num_interv)
plt.xticks(list(range(num_interv + 1)))
plt.plot(x, ProbGytox_givenD, '-.', label=r"$P(G_2|\textbf{D})$", color="red")
plt.legend()
plt.xlabel("Intervention")
#plt.ylabel("$Probability$")
_ = plt.title("$G_2$ : " + r"$(X_1 \leftarrow X_2)$")

plt.subplot(1, 3, 3)
plt.rcParams.update(params)
plt.ylim(0, 1)
plt.xlim(0,num_interv)
plt.xticks(list(range(num_interv + 1)))
plt.plot(x, ProbGg3_givenD, '-.', label=r"$P(G_3|\textbf{D})$", color="red")
plt.legend()
#plt.xlabel("Intervention")
#plt.ylabel("$Probability$")
_ = plt.title("$G_3$ : " + r"$(X_1$" + "  " +  r"$X_2)$")
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
plt.scatter(X1_obs, X2_obs, color="black", label="Initial obs.")
try:
    plt.scatter(X1_interv_x1, X2_interv_x1, color="blue", label=r"$do(X_1=x_1)$")
except:
    print('No interventions on X.')
try:    
    plt.scatter(X1_interv_x2, X2_interv_x2, color="green", label=r"$do(X_2=x_2)$")
except:
    print('No interventions on Y.')
plt.xlabel("$X_1$")
plt.ylabel("$X_2$")
_ = plt.title("Initial and Intervention Observations")
plt.legend()
plt.show()

##############################################################################
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
#k1 = ConstantKernel(1.0) * RBF(length_scale=gamma_g1)       
gp = GaussianProcessRegressor(
     kernel=k, alpha=sigma_x2_g1 
)
# fit the model to the training data
try:
    X_GPr = np.concatenate([X1_obs, X1_interv_x1])
    Y_GPr = np.concatenate([X2_obs, X2_interv_x1])
except:
    X_GPr = X1_obs
    Y_GPr = X2_obs
gp.fit(X_GPr, Y_GPr)
try:
    plot_xtoy(gp, X_predict.reshape(-1, 1), np.concatenate([X1_obs, X1_interv_x1]),  np.concatenate([X2_obs, X2_interv_x1]))
except:
    plot_xtoy(gp, X_predict.reshape(-1, 1), X1_obs, X2_obs)
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
     kernel=k2, alpha=sigma_x1_g2 
)
# fit the model to the training data
try:
    X_GPr2 = np.concatenate([X2_obs, X2_interv_x2]).reshape(-1,1)
    Y_GPr2 = np.concatenate([X1_obs, X1_interv_x2]).reshape(-1,)
except:
    X_GPr2 = X2_obs.reshape(-1,1)
    Y_GPr2 = X1_obs.reshape(-1,)
     
gp2.fit(X_GPr2, Y_GPr2)
try:
    plot_ytox(gp2, y_predict, np.concatenate([X2_obs, X2_interv_x2]).reshape(-1,1), np.concatenate([X1_obs, X1_interv_x2]).reshape(-1,))
except:
    plot_ytox(gp2, y_predict, X2_obs.reshape(-1,1), X1_obs.reshape(-1,))
