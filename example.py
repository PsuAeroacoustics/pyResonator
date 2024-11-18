#!/usr/bin/env python3

import numpy as np
import resonator as res
import matplotlib.pyplot as plt

#%%
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Times New Roman"]
plt.rcParams['font.size'] = 12

#%%

def init_res(f,a_n,L_n,a_c,L_c):
    '''
    This function creates and returns a resonator object with its complex impedance evaluated
    '''
    res_temp = res.resonator(a_n = a_n,L_n =L_n,a_c = a_c, L_c = L_c)
    res_temp.set_Z(f,model = 'k',rad = False,interior = False,loss = False,table = False)
    return res_temp

def init_fs(f,t_fs,r_fs,phi_fs,SPL,M,Z_cav):
    '''
    This function creates and returns a resonator object with its complex impedance evaluated
    '''
    fs_temp = res.fs(t = t_fs,r = r_fs,phi = phi_fs)
    fs_temp.set_Z(f,M = M,SPL = SPL,model = '2P',Z_cav = Z_cav)
    return fs_temp

#%%
df = 1
f_max = 3e3
# frequency array
f = np.arange(1,int(f_max/df))*df

# Density [kg/m^2]
rho = 1.125
# sos [m/s]
sos = 340

# Open area ratio of resonator inlet to entire test sampe of the LaRC NIT facility.
A_ratio = .3**2/2**2
# A_ratio = 1
#%% Uniform depth

# Number of resonators
N = 25
# length of resonator [m]
L = 0.08575
# radius of resonator [m]
r_cav = 0.00381

res1 = res.resonator(a_n = r_cav,L_n = L/2,a_c = r_cav,L_c =  L/2)
res1.set_Z(f)
Z_tot = (N*A_ratio*res1.Z**-1)**-1
react_approx = -(N*A_ratio*np.tan((2*np.pi*f)/sos*L))**-1

fig,ax = plt.subplots(3,1, figsize = (5,6.5))
ax[0].set_xticklabels([])
ax[0].plot(f,np.real(Z_tot))
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([500,3e3])
ax[0].set_ylim([0,5])
ax[0].grid()

ax[1].set_xticklabels([])
ax[1].plot(f,np.imag(Z_tot))
ax[1].plot(f,react_approx)
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([500,3e3])
ax[1].set_ylim([-10,5])
ax[1].legend(['SAIM','-cot(kL)'])
ax[1].grid()

ax[-1].plot(f,res1.alpha)
ax[-1].set_ylabel(r'$Absorption, \ \alpha$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([500,3e3])
ax[-1].set_ylim([0, 1])
plt.show()

#%% Variable depth without facesheet

# Number of resonators
N = 25
# length of each type of resonator comprising the sample [m]
L = np.array([ 3.751, 2.701, 2.110, 1.731, 1.350])/39.37
# radius of cavities (assumed to be the same for the neck and chamber) [m]
r_cav = 0.00381
Z_res = np.array([init_res(f = f,a_n = r_cav,L_n = L[i]/2,a_c = r_cav,L_c =  L[i]/2).Z for i in range(len(L))])

Z_tot = np.sum(N/len(L)*A_ratio*(Z_res)**-1,axis = 0)**-1
alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2

fig,ax = plt.subplots(3,1, figsize = (5,6.5))
ax[0].set_xticklabels([])
ax[0].plot(f,np.real(Z_tot))
ax[0].set_yticks(np.arange(11)[::2])
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([500,3e3])
ax[0].set_ylim([0,10])
ax[0].grid()

ax[1].set_xticklabels([])
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
plt.show()

#%% Variale depth with facesheet

# thickness of facesheet [m]
t_fs = 0.8128e-3
# radius of the perforations [m]
r_fs = 0.7366e-3/2
# porosity of the facesheet 
phi_fs = 0.073

Z_fs = np.array([init_fs(f = f,t_fs = t_fs,r_fs = r_fs,phi_fs = phi_fs,SPL = 120,M = 0,Z_cav = Z_res[i]).Z for i in range(len(L))])

Z_tot = np.sum(N/len(L)*A_ratio*(Z_res+Z_fs)**-1,axis = 0)**-1
alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2

fig,ax = plt.subplots(3,1, figsize = (5,6.5))
ax[0].set_xticklabels([])
ax[0].plot(f,np.real(Z_tot))
ax[0].set_yticks(np.arange(11)[::2])
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([500,3e3])
ax[0].set_ylim([0,10])
ax[0].grid()

ax[1].set_xticklabels([])
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
plt.show()