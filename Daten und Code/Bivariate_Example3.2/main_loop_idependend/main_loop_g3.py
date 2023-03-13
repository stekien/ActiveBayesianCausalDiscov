# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:17:24 2022

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
# ignore warnings
import warnings
warnings.filterwarnings('ignore')
rng = np.random#.RandomState(12)

class Bivariate(object):
    
    def __init__(self, X1_obs, X2_obs, gamma_g1, sigma_x1_g1, sigma_x2_g1, gamma_g2, sigma_x1_g2, sigma_x2_g2, sigma_x1_g3, sigma_x2_g3, nu_matern, const_ker):
        
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
            
            """
            params = {
                "legend.fontsize": 30,
                "axes.titlesize": 45,
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
            
            x = np.linspace(-3, 3, 200).reshape(-1, 1)
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
            _ = plt.title("Objective(s) for $D_1$")
            plt.show()
            
        
            
            fy = [f_y(x_i) for x_i in x]
            gy = [fun_g_x1(x_i) for x_i in x]
            
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
            _ = plt.title("Objective(s) for $D_2$")
            plt.show()
            #plt.plot(x, fy, "r--", label=r"$\tilde{g}_2(x_2^2)$")
            #plt.plot(x, gy, "g--", label=r"$g_2(x_2^2)$")
            #plt.legend()
            #plt.xlabel("Intervention value $x_2^2$")
            #plt.ylabel("Negative Information Gain")
           # _ = plt.title("Objective for $D_2$")
            #plt.show()
            """
            
            
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
                    X2_interv_x1 = rng.normal(loc=0.0, scale=1, size=1).reshape(1,)
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



###############################################################################
#   Interventions
###############################################################################
# methods to perform interventions
def interv_x(x, X_train, Y_train):
    x_interv = x
    y_dox = rng.normal(loc=0.0, scale=1, size=1).reshape(1,)
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
    

def gfunk(j):
    print("Worker %s calculating for seed %d"% (current_process().pid, j))
    
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
    
    # generate data sequence which will be predicted in the later graphs
    #X_predict = np.linspace(start=-3, stop=3, num=1_000).reshape(-1, 1)
    X_predict = np.linspace(start=-3, stop=3, num=100).reshape(-1, 1)
    #y_predict = np.linspace(start=-3, stop=3, num=1_000).reshape(-1, 1)
    
    np.random.seed(seed=j)
    
    # data is generated synthetically in this script
    rng = np.random.RandomState(j)
    # generate (large) sample of the underlying distribution
    
    # generate initial observations 
    num_initial_obs = 5
    noise_std = 1
    
    X1_obs = rng.normal(loc=0.0, scale=1, size=num_initial_obs).tolist()
    X2_obs = [rng.normal(loc=0.0, scale=noise_std, size=None) for x in X1_obs] 
    
    X1_obs = np.array(X1_obs).reshape(-1, 1)
    X2_obs = np.array(X2_obs).reshape(-1,)
    
    model = Bivariate(X1_obs, X2_obs, gamma_g1, sigma_x1_g1, sigma_x2_g1, gamma_g2, sigma_x1_g2, sigma_x2_g2, sigma_x1_g3, sigma_x2_g3, nu_matern = nuu, const_ker = 1.0)  
   
    probG_givenD_isproportional_g1, probG_givenD_isproportional_g2, probG_givenD_isproportional_g3, X1_interv_x1, X2_interv_x1, X1_interv_x2, X2_interv_x2 = model.optimal_int(num_interv, interv_x, interv_y)  


    ProbGg1_givenD = [x / (x + y + z) for x,y,z in zip(probG_givenD_isproportional_g1, probG_givenD_isproportional_g2,probG_givenD_isproportional_g3)]   
    ProbGg2_givenD = [y / (x + y + z) for x,y,z in zip(probG_givenD_isproportional_g1, probG_givenD_isproportional_g2,probG_givenD_isproportional_g3)]   
    ProbGg3_givenD = [z / (x + y + z) for x,y,z in zip(probG_givenD_isproportional_g1, probG_givenD_isproportional_g2,probG_givenD_isproportional_g3)]  
    """
    result_0[j,0] = ProbGxtoy_givenD[0]
    result_0[j,1] = ProbGytox_givenD[0]
    result_0[j,2] = ProbGg3_givenD[0]
    
    result_1[j,0] = ProbGxtoy_givenD[1]
    result_1[j,1] = ProbGytox_givenD[1]
    result_1[j,2] = ProbGg3_givenD[1]
    
    result_2[j,0] = ProbGxtoy_givenD[2]
    result_2[j,1] = ProbGytox_givenD[2]
    result_2[j,2] = ProbGg3_givenD[2]
    
    result_3[j,0] = ProbGxtoy_givenD[3]
    result_3[j,1] = ProbGytox_givenD[3]
    result_3[j,2] = ProbGg3_givenD[3]
    
    result_4[j,0] = ProbGxtoy_givenD[4]
    result_4[j,1] = ProbGytox_givenD[4]
    result_4[j,2] = ProbGg3_givenD[4]
    
    result_5[j,0] = ProbGxtoy_givenD[num_interv]
    result_5[j,1] = ProbGytox_givenD[num_interv]
    result_5[j,2] = ProbGg3_givenD[num_interv]
    """    
    #####################################
    # Confidence Interval Part
    #####################################
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
    mean_prediction, std_prediction = gp.predict(X_predict, return_std=True)
    lower_g1 = mean_prediction - 1.96 * std_prediction
    upper_g1 = mean_prediction + 1.96 * std_prediction
    truth_g1 = (2 * np.tanh(X_predict))
    #oolb = np.sum([(x-y<0) for x,y in zip(truth_g1, lower_g1)])
    #ooub = np.sum([(x-y<0) for x,y in zip(upper_g1, truth_g1)])
    #confidenceIntervalViolationsPercent = (oolb + ooub) / len(X_predict)
    
    oolb = [(x-y<0).astype(int) for x,y in zip(truth_g1, lower_g1)]
    oolb = [np.asscalar(x) for x in oolb]
    ooub = [(x-y<0).astype(int) for x,y in zip(upper_g1, truth_g1)]
    ooub = [np.asscalar(x) for x in ooub]
    confidenceIntervalViolationsPercent = [x+y for x,y in zip(oolb, ooub)] 
    
    return {"j":j,"results":{"ProbGg1_givenD":ProbGg1_givenD, "ProbGg2_givenD":ProbGg2_givenD, "ProbGg3_givenD":ProbGg3_givenD, "confidenceIntervalViolationsPercent":confidenceIntervalViolationsPercent}}


k1 = 1000
if __name__ == "__main__":
    nprocs= 6
    
    # printthenumberofcores
    print("Number of workers equals %d"% nprocs)
    # createa poolofworkers
    pool= Pool(processes=nprocs)
    # createan arrayof10 integers, from1 to10
    a = range(k1)
    
    result= pool.map(gfunk, a)
    print(result)
    
    #total = reduce(lambda x,y: x+y, result)
    
    #print("The sum of the square of the first 10 integers is %d"% total)

"""
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
plt.style.use("seaborn")
plt.rcParams.update(params)
# 


result_0 = np.zeros((k1,3))
result_1 = np.zeros((k1,3))
result_2 = np.zeros((k1,3))
result_3 = np.zeros((k1,3))
result_4 = np.zeros((k1,3))
result_5 = np.zeros((k1,3))
#confidenceIntervalViolationsPercent = np.zeros(k1)
confidenceIntervalViolationsPercent = np.zeros(100)

for h in result:
    #print(h["j"])
    
    dum = h["results"]
    
    result_0[h["j"],0] = dum["ProbGg1_givenD"][0]
    result_0[h["j"],1] = dum["ProbGg2_givenD"][0]
    result_0[h["j"],2] = dum["ProbGg3_givenD"][0]
    
    result_1[h["j"],0] = dum["ProbGg1_givenD"][1]
    result_1[h["j"],1] = dum["ProbGg2_givenD"][1]
    result_1[h["j"],2] = dum["ProbGg3_givenD"][1]
    
    result_2[h["j"],0] = dum["ProbGg1_givenD"][2]
    result_2[h["j"],1] = dum["ProbGg2_givenD"][2]
    result_2[h["j"],2] = dum["ProbGg3_givenD"][2]
    
    result_3[h["j"],0] = dum["ProbGg1_givenD"][3]
    result_3[h["j"],1] = dum["ProbGg2_givenD"][3]
    result_3[h["j"],2] = dum["ProbGg3_givenD"][3]
    
    result_4[h["j"],0] = dum["ProbGg1_givenD"][4]
    result_4[h["j"],1] = dum["ProbGg2_givenD"][4]
    result_4[h["j"],2] = dum["ProbGg3_givenD"][4]
    
    result_5[h["j"],0] = dum["ProbGg1_givenD"][5]
    result_5[h["j"],1] = dum["ProbGg2_givenD"][5]
    result_5[h["j"],2] = dum["ProbGg3_givenD"][5]
    
    #confidenceIntervalViolationsPercent[h["j"]] = dum["confidenceIntervalViolationsPercent"]
    confidenceIntervalViolationsPercent = confidenceIntervalViolationsPercent + dum["confidenceIntervalViolationsPercent"]
    #print(len(dum["confidenceIntervalViolationsPercent"]))
    
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
    
    success_95 = np.sum((result[:,0] >= 0.95)) / 1000
    success_80 = np.sum((result[:,0] >= 0.8)) / 1000
    failure_50 = np.sum((result[:,0] <= 0.5)) / 1000
    
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


def evaluate1(result):    
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
    plt.hist(result[:,1], bins=100, density=True)
    ax.tick_params(axis = 'x',labelrotation = 45)
    ax.set_xticks(np.arange(0, 1.2, step=0.2))
    plt.xlabel(r"$P(G_2|\textbf{D})$")
    plt.ylabel("$Frequency$")
    _ = plt.title("Histogram of " + r"$P(G_2|\textbf{D})$")
    plt.show()
    
    success_95 = np.sum((result[:,1] >= 0.95)) / 1000
    success_80 = np.sum((result[:,1] >= 0.8)) / 1000
    failure_50 = np.sum((result[:,1] <= 0.5)) / 1000
    
    mn = np.mean(result[:,1])
    var = np.var(result[:,1])
    
    quantiles = np.quantile(result[:,1], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

    return success_95, success_80, failure_50, mn, var, quantiles

success_95_01, success_80_01, failure_50_01, mn_0, var_01, quantiles_01 = evaluate1(result_0)
success_95_11, success_80_11, failure_50_11, mn_11, var_11, quantiles_11 = evaluate1(result_1)
success_95_21, success_80_21, failure_50_21, mn_21, var_21, quantiles_21 = evaluate1(result_2)
success_95_31, success_80_31, failure_50_31, mn_31, var_31, quantiles_31 = evaluate1(result_3)
success_95_41, success_80_41, failure_50_41, mn_41, var_41, quantiles_41 = evaluate1(result_4)
success_95_51, success_80_51, failure_50_51, mn_51, var_51, quantiles_51 = evaluate1(result_5)


def evaluate2(result):    
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
    plt.hist(result[:,2], bins=100, density=True)
    ax.tick_params(axis = 'x',labelrotation = 45)
    ax.set_xticks(np.arange(0, 1.2, step=0.2))
    plt.xlabel(r"$P(G_3|\textbf{D})$")
    plt.ylabel("$Frequency$")
    _ = plt.title("Histogram of " + r"$P(G_3|\textbf{D})$")
    plt.show()
    
    success_95 = np.sum((result[:,2] >= 0.95)) / 1000
    success_80 = np.sum((result[:,2] >= 0.8)) / 1000
    failure_50 = np.sum((result[:,2] <= 0.5)) / 1000
    
    mn = np.mean(result[:,2])
    var = np.var(result[:,2])
    
    quantiles = np.quantile(result[:,2], [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])

    return success_95, success_80, failure_50, mn, var, quantiles

success_95_02, success_80_02, failure_50_02, mn_02, var_02, quantiles_02 = evaluate2(result_0)
success_95_12, success_80_12, failure_50_12, mn_12, var_12, quantiles_12 = evaluate2(result_1)
success_95_22, success_80_22, failure_50_22, mn_22, var_22, quantiles_22 = evaluate2(result_2)
success_95_32, success_80_32, failure_50_32, mn_32, var_32, quantiles_32 = evaluate2(result_3)
success_95_42, success_80_42, failure_50_42, mn_42, var_42, quantiles_42 = evaluate2(result_4)
success_95_52, success_80_52, failure_50_52, mn_52, var_52, quantiles_52 = evaluate2(result_5)


#plt.hist(confidenceIntervalViolationsPercent, bins=100, density=True)
#plt.xlabel("Percentage of confidence bound violations")
#plt.ylabel("$Frequency$")
#_ = plt.title("Percentage of confidence bound (0.95) violations of 1000 tested points")
#plt.show()


params = {
    "legend.fontsize": 25,
    "axes.titlesize": 40,
    "figure.figsize": (12, 12),
    "figure.dpi": 100,
    "axes.labelsize": 40,
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
plt.plot(np.linspace(start=-3, stop=3, num=100).reshape(-1, 1), confidenceIntervalViolationsPercent/1000)
plt.xlabel("$X_1$")
plt.ylabel("Violations in Percent")
_ = plt.title("Empirical 95\% Confidence Bound violations of $\hat{f}^{(2)}$")
plt.show()
"""