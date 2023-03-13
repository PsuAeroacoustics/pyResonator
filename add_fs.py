import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import resonator as res
import os
from scipy.signal import butter
from scipy.optimize import differential_evolution,NonlinearConstraint
import sys
from time import time
import h5py
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()),'pyPostAcs'))
import pyPostAcsFun as fun

#%%

cmap = plt.cm.Spectral.reversed()
fontName = 'Times New Roman'
fontSize = 14
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%

res_data = {}
with h5py.File(os.path.join(os.getcwd(),'opt_4res_elem_no_fs.h5'),'r') as f:
    for k,v in f.items():
        res_data = {**res_data,**{k:v[()]}}

#%%

fs1  = res.fs(t = 1e-3,r = 0.5e-3,phi = 0.073)
fs1.set_Z(f[1:])
Z_tot = np.zeros(len(f[1:]))
for k,v in helm_dict.items():
    Z_tot = Z_tot+N/len(L)*A_ratio*(v.Z+fs1.get_Z())**-1
Z_tot = Z_tot**-1

alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2
