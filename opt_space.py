from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import resonator as res
import os
import re
from scipy.interpolate import interp1d,interp2d,RectBivariateSpline
from scipy import signal
import sys
sys.path.insert(0, '/Users/danielweitsman/codes/github/DanWeitsman/pyPostAcs')
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

# Frequency array vector
df = 1
f_max = 3e3
f = np.arange(int(f_max/df))*df

# Density [kg/m^2]
rho = 1.125
# SoS [m/s]
a0 = 340

#%% Uniform depth liner variation in chamber length vs radius

# Number of resonators
N = 25
# number of parameter variations
n = 10
# maximum length [m]
L_max =  12*0.0254
# Area of sample (2"x2")
A_sample = 2**2*0.0254**2
# maximum radius of each resonator [m]
a_max  = np.sqrt(A_sample/(np.pi*N))

L = (np.arange(n)/n*L_max)[1:]
a = (np.arange(n)/n*a_max)[1:]

L2_err = np.empty((len(L),len(a))) 

# target normalized complex impedance
# Z_targ = np.zeros(len(f[1:]))+1j*np.zeros(len(f[1:]))
alpha_targ = np.ones(len(f[1:]))

# frequency weighting function

# b,a = signal.butter(4,  500 ,btype = 'hp',fs = len(f))

n,d = signal.butter(4,  [1400,1500] ,btype = 'bp',fs = 2*df*len(f))
f2,y,h,phase = fun.filt_response(n,d,fs = df*len(f),N = 2*len(f) ,plot=False)
W = np.abs(h)[1:]

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.plot(f[1:],W)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Weighting Function, W')
ax.grid()

for i,L_itr in enumerate(L):
    for ii,a_itr in enumerate(a):
        res_temp = res.resonator(a_n = a_itr,a_c = a_itr,L_n = 1e-10, L_c = L_itr)
        res_temp.set_Z(f[1:],model = 'ZKTL',rad = False,interior = False,loss = True,table = False)
        Z_tot = (N*np.pi*a_itr**2/A_sample*(res_temp.Z)**-1)**-1
        alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2
        L2_err[i,ii] = np.sum(W*(alpha_targ-alpha)**2)/np.sum(W)

        # L2_err[i,ii] = np.sqrt(np.real(np.sum(W*(Z_targ-Z_tot)*np.conj(Z_targ-Z_tot))))

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)
plt.subplots_adjust(left = 0.15)

# plt.subplots_adjust(bottom = 0.15)

levels = np.linspace(np.min(L2_err), np.max(L2_err), 50)
h = ax.contourf(L, a, L2_err.transpose(),cmap = cmap,levels = levels)
# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel('Length [m]')
ax.set_ylabel('Radius [m]')
cbar = fig.colorbar(h)
cbar.ax.set_ylabel('$L_2$')
# cbar.ax.set_ylim([0,5e3])

#%% Uniform radius - 2 different depth resonators

# Number of resonators
N = 24
# number of parameter variations
n = 10
# maximum length [m]
L_max =  12*0.0254
L = (np.arange(n)/n*L_max)[1:]
a_c = a[2]

L2_err = np.empty((len(L),len(L))) 
# n,d = signal.butter(4,  500 ,btype = 'hp',fs = len(f))

n,d = signal.butter(4,  [1000,2000] ,btype = 'bp',fs = 2*df*len(f))
f2,y,h1,phase = fun.filt_response(n,d,fs = df*len(f),N = 2*len(f) ,plot=False)

# n,d = signal.butter(4,  [1400,1500] ,btype = 'bp',fs = 2*df*len(f))
# f2,y,h2,phase = fun.filt_response(n,d,fs = df*len(f),N = 2*len(f) ,plot=False)
# W = (np.abs(h1)+np.abs(h2))[1:]
W = (np.abs(h1))[1:]
fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.plot(f[1:],W)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Weighting Function, W')
ax.grid()


helm_dict = {}
for i,L1_itr in enumerate(L):
    helm_dict = {**helm_dict,**{f'helm1{i}':res.resonator(a_n = a_c,a_c = a_c,L_n =1e-10, L_c = L1_itr)}}
    helm_dict[f'helm1{i}'].set_Z(f[1:],model = 'k',rad = False,interior = False,loss = True,table = True)
    for ii,L2_itr in enumerate(L):
        helm_dict = {**helm_dict,**{f'helm2{i}':res.resonator(a_n = a_c,a_c = a_c,L_n =1e-10, L_c = L2_itr)}}
        helm_dict[f'helm2{i}'].set_Z(f[1:],model = 'k',rad = False,interior = False,loss = True,table = True)
        
        Z_tot = (N/2*np.pi*a_c**2/A_sample*(helm_dict[f'helm2{i}'].Z)**-1+N/2*np.pi*a_c**2/A_sample*(helm_dict[f'helm1{i}'].Z)**-1)**-1
        alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2
        L2_err[i,ii] = np.sum(W*(1-alpha)**2)/np.sum(W)



fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
levels = np.linspace(np.min(L2_err), np.max(L2_err), 50)
h = ax.contourf(L, L, L2_err.transpose(),cmap = cmap,levels = levels)
ax.set_xlabel('Length of 1st resonator [m]')
ax.set_ylabel('Length of 2nd resonator [m]')
cbar = fig.colorbar(h)
cbar.ax.set_ylabel('$L_2$')


Z_tot = (N/2*np.pi*a_c**2/A_sample*(helm_dict[f'helm27'].Z)**-1+N/2*np.pi*a_c**2/A_sample*(helm_dict[f'helm11'].Z)**-1)**-1
alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.plot(f[1:],alpha)
ax.set_ylabel(r'$Absorption, \ \alpha$')
ax.set_xlabel('Frequency [Hz]')
ax.grid()
ax.set_xlim([500,3e3])
ax.set_ylim([0, 1])

#%%

# Number of resonators
N = 25
# number of parameter variations
n = 10
# maximum length [m]
L_c = L[1]
# Area of sample (2"x2")
A_sample = 2**2*0.0254**2

L_n = (np.arange(n)/n*.9*L_c)[1:]
a_n = (np.arange(n)/n*.9*a_c)[1:]

L2_err = np.empty((len(L_n),len(a_n))) 

# target normalized complex impedance
# Z_targ = np.zeros(len(f[1:]))+1j*np.zeros(len(f[1:]))
alpha_targ = np.ones(len(f[1:]))

# frequency weighting function

n,d = signal.butter(4,  [900,1000] ,btype = 'bp',fs = 2*df*len(f))
f2,y,h,phase = fun.filt_response(n,d,fs = df*len(f),N = 2*len(f) ,plot=False)
W = np.abs(h)[1:]

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.plot(f[1:],W)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel('Weighting Function, W')
ax.grid()

for i,L_itr in enumerate(L_n):
    for ii,a_itr in enumerate(a_n):
        res_temp = res.resonator(a_n = a_itr,a_c = a_c,L_n = L_itr, L_c = L_c)
        res_temp.set_Z(f[1:],model = 'ZKTL',rad = False,interior = False,loss = True,table = False)
        Z_tot = (N*np.pi*a_itr**2/A_sample*(res_temp.Z)**-1)**-1
        alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2
        L2_err[i,ii] = np.sum(W*(alpha_targ-alpha)**2)/np.sum(W)

        # L2_err[i,ii] = np.sqrt(np.real(np.sum(W*(Z_targ-Z_tot)*np.conj(Z_targ-Z_tot))))

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)
plt.subplots_adjust(left = 0.15)
levels = np.linspace(np.min(L2_err), np.max(L2_err), 50)
h = ax.contourf(L_n/L_c, a_n/a_c, L2_err.transpose(),cmap = cmap,levels = levels)
ax.set_xlabel('$L_n/L_c$')
ax.set_ylabel('$r_n/r_c$')
cbar = fig.colorbar(h)
cbar.ax.set_ylabel('$L_2$')
