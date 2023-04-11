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

fname = 'res_opt.h5'

res_data = {}
with h5py.File(os.path.join(os.path.dirname(__file__),fname),'r') as f:
    for k,v in f.items():
        res_data = {**res_data,**{k:v[()]}}
f = res_data['f']
A_ratio = .3**2/2**2

res_temp_data = {}
for i in range(len(res_data['res_opt'])):
    res_temp_data = {**res_temp_data,**{f'helm{i}':res.resonator(a_n = res_data['res_opt'][i,1],a_c = res_data['res_opt'][i,3],L_n =res_data['res_opt'][i,2], L_c = res_data['res_opt'][i,4])}}
    res_temp_data[f'helm{i}'].set_Z(f,model = 'k',rad = False,interior = False,loss = True,table = True)

fs1  = res.fs(t = 1e-3,r = 0.5e-3,phi = 0.073)
fs1.set_Z(f)

Z_tot = np.zeros(len(f))
for i,k in enumerate(res_temp_data.keys()):    
    Z_tot = Z_tot+np.round(res_data['res_opt'][:,0])[i]/len(res_data['res_opt'])*A_ratio*(res_temp_data[k].Z+fs1.Z)**-1
Z_tot = Z_tot**-1
alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2


res_data = {}
N = 25
A_ratio = .3**2/2**2
# A_ratio = 0.07339
L = np.array([ 3.751, 2.701, 2.110, 1.731, 1.350])*0.0254

df = 1
f_max = 3e3
# f = (np.arange(int(f_max/df))*df)[1:]
fs1  = res.fs(t = 1e-3,r = 0.5e-3,phi = 0.073)
fs1.set_Z(f)
Z_tot = np.zeros(len(f))
for i,n in enumerate(L):
    res_data = {**res_data,**{f'helm{i}':res.resonator(a_n = 0.003809,a_c = 0.003809,L_n =n/2, L_c = n/2)}}
    res_data[f'helm{i}'].set_Z(f,model = 'k',rad = False,interior = False,loss = True,table = True)

for k,v in res_data.items():
    Z_tot = Z_tot+N/len(L)*A_ratio*(v.Z+fs1.get_Z())**-1
Z_tot = Z_tot**-1

alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2

res_data['Z_tot'] = Z_tot
res_data['alpha'] = alpha

save_dir = os.path.join(os.getcwd(),'res_opt_fs.h5')
if os.path.exists(save_dir):
    os.remove(save_dir)

with h5py.File(save_dir, 'a') as h5_f:
    for k, v in res_data.items():
        h5_f.create_dataset(k, shape=np.shape(v), data=v)




#%%

fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].plot(f,np.real(Z_tot))
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([500,3e3])
ax[0].set_ylim([0,10])
ax[0].grid()

ax[1].tick_params(axis = 'x', labelsize=0)
ax[1].plot(f,np.imag(Z_tot))
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([500,3e3])
ax[1].set_ylim([-5,5])
ax[1].grid()

ax[-1].plot(f,alpha)
ax[-1].set_ylabel(r'$Absorption, \ \alpha$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([500,3e3])
ax[-1].set_ylim([0, 1])

