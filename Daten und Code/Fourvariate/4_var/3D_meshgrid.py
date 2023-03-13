# -*- coding: utf-8 -*-
"""
Created on Tue Oct 25 08:44:18 2022

@author: Stefan
"""
import numpy as np

x_ = np.linspace(0., 1., 10)
y_ = np.linspace(1., 2., 20)
z_ = np.linspace(3., 4., 30)

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

assert np.all(x[:,0,0] == x_)
assert np.all(y[0,:,0] == y_)
assert np.all(z[0,0,:] == z_)
