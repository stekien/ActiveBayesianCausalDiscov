# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 15:04:06 2022

@author: Stefan
"""
import numpy as np
import matplotlib.pyplot as plt


t = np.linspace(-2.5,2.5,10000)
a = 2*np.tanh(t)
b = 2*t
c = t


params = {
    "legend.fontsize": 30,
    "axes.titlesize": 65,
    "figure.figsize": (12, 14),
    "figure.dpi": 100,
    "axes.labelsize": 40,
    "xtick.labelsize": 40,
    "ytick.labelsize": 40,
    "lines.linewidth": 5,
    "lines.markeredgewidth": 2.5,
    "lines.markersize": 5,
    "lines.marker": "o",
    "patch.edgecolor": "black",
    "text.usetex": True,
    "font.family": "serif"
}
plt.style.use('seaborn')
plt.rcParams.update(params)
plt.plot(t, b, "g--", label=r"$f(x)=2x$")
plt.plot(t, c, "r--", label=r"$f(x)=x$")
plt.plot(t, a, "b", label=r"$f(x)=2\tanh(x)$")
plt.legend()
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.ylim((-2.5,2.5))
_ = plt.title("Underlying Functional Relation")
#plt.plot(t, a, t, b, t, c)
plt.show()