from xml.parsers.expat import model
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import resonator as res
import os
import re
from scipy.interpolate import interp2d,RectBivariateSpline
# Frequency array vector
df = 1
f_max = 3e3
f = np.arange(int(f_max/df))*df

# Density [kg/m^2]
rho = 1.125
# SoS [m/s]
a0 = 340

# sigma_i = (np.pi*helm1.a_n**2)/0.00258064
# Open area ratio of resonator inlet to entire test sampe of the LaRC NIT facility.
A_ratio = .3**2/2**2

#%% Single uniform depth resonator located at center of sample (NU1)

a_n ,a_c,L_n , L_c = 0.006286,0.006286,0.04445/2,0.04445/2
helm1 = res.resonator(a_n = a_n,a_c = a_c,L_n =L_n, L_c = L_c)
helm1.set_Z(f[1:],model = 'WG',rad = True,loss = False,interior = True)
Z_tot = (A_ratio*helm1.Z**-1)**-1

helm1.set_Z(f[1:],model = 'Kirchoff',rad = True,interior = False)
Z_tot_2 = (A_ratio*helm1.Z**-1)**-1

fig,ax = plt.subplots(1,2, figsize = (8,4.5))
plt.subplots_adjust(bottom = 0.15)
ax[0].plot(f[1:],np.real(Z_tot),c = 'black')
ax[0].plot(f[1:],np.real(Z_tot_2),c = 'r',linestyle = '--')
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([1400,2200])
ax[0].set_ylim([0,1.5])
ax[0].grid()
ax[0].set_xlabel('Frequency [Hz]')

ax[1].plot(f[1:],np.imag(Z_tot),c = 'black')
ax[1].plot(f[1:],np.imag(Z_tot_2),c = 'r',linestyle = '--')
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([1400,2200])
ax[1].set_ylim([-5,5])
ax[1].grid()
ax[1].set_xlabel('Frequency [Hz]')
ax[-1].legend(['NU1','NU4'])

#%%


f_dir = os.path.join(os.getcwd(),'Tijdeman_gamma')
f_name = ['re_gamma.txt','imag_gamma.txt']

data_temp = np.empty((len(f_name),36,11))
for i,n in enumerate(f_name):
    with open(os.path.join(f_dir,n)) as f:
        data_temp[i] = np.array(re.split("\t|\n",f.read())).reshape((36,11)).astype(float)
s = data_temp[0,1:,0]
ka = data_temp[0,0,1:]

data = data_temp[0,1:,1:]+1j*data_temp[1,1:,1:]
del data_temp

f_re_gamma = RectBivariateSpline(x = s,y = ka, z = np.real(data))
f_imag_gamma = RectBivariateSpline(x = s,y = ka, z = np.imag(data))

gamma = f_re_gamma(x = 2.2,y = np.arange(10)*0.05)+1j*f_imag_gamma(x = 2.2,y = np.arange(10)*0.05)

#%%


a_n ,a_c,L_n , L_c = 0.005388,0.005388,0.0857504/2,0.0857504/2
helm1 = res.resonator(a_n = a_n,a_c = a_c,L_n =L_n, L_c = L_c)
helm1.set_Z(f[1:],model = 'k',rad = False,interior = False,loss = True)
Z_tot = (25*A_ratio*(helm1.Z)**-1)**-1
Z_tot_2 = -(25*A_ratio*np.tan(2*np.pi*f[1:]/a0*0.0857504))**-1

fig,ax = plt.subplots(1,2, figsize = (8,4.5))
plt.subplots_adjust(wspace = 0.3)

# plt.subplots_adjust(bottom = 0.15)
ax[0].plot(f[1:],np.real(Z_tot),c = 'black')
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([400,3e3])
ax[0].set_ylim([0,5])
ax[0].grid()
ax[0].set_xlabel('Frequency [Hz]')

ax[1].plot(f[1:],np.imag(Z_tot),c = 'black')
# ax[1].plot(f[1:],Z_tot_2)
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([400,3e3])
ax[1].set_ylim([-10,5])
ax[1].grid()
ax[1].set_xlabel('Frequency [Hz]')
# ax[-1].legend(['NU1','NU4'])

#%%

# Number of resonators
N = 25
L = np.array([ 3.751, 2.701, 2.110, 1.731, 1.350])*0.0254
# L = np.array([2.701])*0.0254
helm_dict = {}
for i,n in enumerate(L):
    helm_dict = {**helm_dict,**{f'helm{i}':res.resonator(a_n = 0.003809,a_c = 0.003809,L_n =n/2, L_c = n/2)}}
    helm_dict[f'helm{i}'].set_Z(f[1:],model = 'k',rad = False,interior = False,loss = True,table = True)

Z_tot = np.zeros(len(f[1:]))
for k,v in helm_dict.items():
    Z_tot = Z_tot+N/len(L)*A_ratio*(v.Z)**-1
Z_tot = Z_tot**-1

# Br = 0.5
# Bi = 8/(3*np.pi)
# a_rad = 0.005388
# k = 2*np.pi*f[1:]/a0
# Z_rad = (Br*(k*a_rad)**2+1j*Bi*(k*a_rad))*np.pi*0.005388**2
# # Z_rad = 1j*k*a_rad/(1+1j*k*a_rad)
# # Z_rad = 1j*k*a_rad*(Bi*(1-1.25*np.sqrt(0.02)))

# Z_tot = Z_rad+Z_tot
alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2

fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].plot(f[1:],np.real(Z_tot))
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([500,3e3])
ax[0].set_ylim([0,10])
ax[0].grid()

ax[1].tick_params(axis = 'x', labelsize=0)
ax[1].plot(f[1:],np.imag(Z_tot))
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([500,3e3])
ax[1].set_ylim([-5,5])
ax[1].grid()

ax[-1].plot(f[1:],alpha)
ax[-1].set_ylabel(r'$Absorption, \ \alpha$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([500,3e3])
ax[-1].set_ylim([0, 1])

#%%

fs1  = res.fs(t = 1e-3,r = 0.5e-3,phi = 0.073)
fs1.set_Z(f)

Z_tot = fs1.get_Z()+Z_tot
alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2

fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].plot(f[1:],np.real(Z_tot))
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([500,3e3])
ax[0].set_ylim([0,10])
ax[0].grid()

ax[1].tick_params(axis = 'x', labelsize=0)
ax[1].plot(f[1:],np.imag(Z_tot))
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([500,3e3])
ax[1].set_ylim([-5,5])
ax[1].grid()

ax[-1].plot(f[1:],alpha)
ax[-1].set_ylabel(r'$Absorption, \ \alpha$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([500,3e3])
ax[-1].set_ylim([0, 1])


#%% Optimization space - compare different radii and lengths / compare different neck/chamber radii and lengths

N = 25
n = 10
A_ratio_max = 0.8
A_ratio = (np.arange(n)/(n-1)*A_ratio_max)[1:]

A_s = 2**2*0.0254**2
a_n = np.sqrt(A_ratio*A_s/np.pi)

helm_dict = {}
for i,n in enumerate(A_ratio):
    helm_dict = {**helm_dict,**{f'helm{i}':res.resonator(a_n = a_n[i],a_c =a_n[i],L_n =L_n, L_c  = L_n)}}
    helm_dict[f'helm{i}'].set_Z(f[1:],model = 'k',rad = False,interior = False,loss = True,table = False)

Z_tot = np.zeros(len(f[1:]))
for i,k in enumerate(helm_dict):
    Z_tot = Z_tot+N/len(A_ratio)*A_ratio[i]*(helm_dict[k].Z)**-1
Z_tot = Z_tot**-1


#%%

# Total number of resonators
N = 25
# number of different types of resonators
n = 10

# ratio of length of neck to cavity (ranges from 0-0.75)
L_ratio = np.arange(n)/(n-1)*0.75
# ratio of radius of neck to cavity (ranges from 0-0.75)
a_ratio = np.arange(n)/(n-1)*0.75

A_s 
a_n = 0.003809
L_n = 3.751

helm_dict = {}
for i in n:
    helm_dict = {**helm_dict,**{f'helm{i}':res.resonator(a_n = a_n,a_c = a_ratio**-1*a_n,L_n =L_n, L_c  =L_ratio**-1*L_n)}}
    helm_dict[f'helm{i}'].set_Z(f[1:],model = 'k',rad = False,interior = False,loss = True,table = False)

Z_tot = np.zeros(len(f[1:]))
for k,v in helm_dict.items():
    Z_tot = Z_tot+N/len(L)*A_ratio*(v.Z)**-1
Z_tot = Z_tot**-1


