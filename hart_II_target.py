import os
from pyparsing import line
import h5py
import numpy as np
import sys
sys.path.insert(0, '/Users/danielweitsman/codes/github/DanWeitsman/pyPostAcs')
import pyPostAcsFun as fun
import matplotlib.pyplot as plt
from scipy.signal import welch  
from scipy.fft import fft,ifft
sys.path.insert(0, '/Users/danielweitsman/codes/github/OpenWopWop/pyWopwop')
import wopwop
import matplotlib.colors as mcolors
import resonator as res
from scipy.optimize import minimize, LinearConstraint,NonlinearConstraint
import csv
import re
from scipy import signal
#%%

work_dir = '/Users/danielweitsman/Documents/research/prediction/hart_II/hart_vtu'
f_data = 'id10427_data.csv'

with open(os.path.join(work_dir,f_data)) as f:

    raw_data = re.split( ",|\n",f.read())
    header = raw_data[:12]
    data = np.array(raw_data[12:-1]).reshape(int(len(raw_data[12:-1])/len(header)),len(header)).transpose().astype(float)

data_dict = {n:data[i] for i,n in enumerate(header)}

#%%
# non-dimentinal rotor radius
r = np.mean(np.linalg.norm((data_dict['"Points:0"'],data_dict['"Points:1"'],data_dict['"Points:2"']),axis = 0))

# rotational rate [rad/s] - "A comprehensive rotary-wing data base for code validation: the HART II international workshop" - V.D Wall
omega = 109
# Bo-105 radius [m]
R = 2
# Bo-105 chord [m]
c = .121
# Bo-105 root cut-out [m]
e = 0.44
# Bo-105 max thickness [m]
t = 0.12*c
# Estimated blade volume [m^2]
V0 = 0.7*(R-e)*t*c
# local total velocity
U = np.linalg.norm((data_dict['"u_p"'],data_dict['"u_t"']),axis = 0)*omega*R
# density [kg/m3]
rho = 1.125
# SoS [m/s]
a0 = 340

#%%

N = len(data_dict['"dC_T_dot"'])
fs = N/(omega/(2*np.pi))**-1
dt = fs**-1
df = (N*dt)**-1

# Xm = fft(data_dict['"dC_T_dot"'])*dt
# #   Computes double-sided PSD
# Sxx = np.conj(Xm) * Xm*df
# #   Computes single-sided PSD
# Gxx = Sxx[:int(N/2)]
# Gxx[1:-1] = 2 * Gxx[1:-1]

# ax.stem(f,np.real(Gxx))
# ax.set_xscale('log')
# ax.set_ylabel($'\partial C_T/ \partial \psi ,\dB '$)
#%
f, Xm, Sxx, Gxx = fun.PSD(data_dict['"dC_T_dot"'],fs = fs)
f, Xm, Sxx, Gxx = fun.PSD(data_dict['"dC_T_dot"'],fs = fs)
fc = 500
Xm[int(fc/df):int(N/2)+1] = 0
Xm[int(N/2)+1:-int(fc/df)+1] = 0
dC_T_dot_filt = ifft(Xm)/dt


# fc = [400,1.5e3]

# Xm[int(fc[0]/df):int(fc[-1]/df)] = 0
# Xm[-int(fc[-1]/df)+1:-int(fc[0]/df)+1] = 0
# dC_T_dot_filt = ifft(Xm)/dt

# Xm[Xm != 0] = 1+1j
b,a = signal.butter(4, [500,1500] ,'bandstop',fs = fs)
b,a = signal.butter(4, fc ,'lp',fs = fs)
fun.filt_response(b,a,fs,N-1 ,plot=True)

dC_T_dot_filt = signal.lfilter(b, a, data_dict['"dC_T_dot"'])

f, Xm, Sxx, Gxx_l = fun.PSD(data_dict['"l"'],fs = fs)
f, Xm, Sxx, Gxx_dCT = fun.PSD(data_dict['"dC_T_dot"'],fs = fs)

f, Xm, Sxx, Gxx_l = fun.PSD(data_dict['"l"'],fs = fs)
f, Xm, Sxx, Gxx_dCT_filt = fun.PSD(dC_T_dot_filt,fs = fs)


fig,ax = plt.subplots(1,1,figsize = (8,6))
plt.subplots_adjust(hspace = 0.35,bottom = 0.1)
ax.plot(data_dict['"dC_T_dot"'])
ax.plot(dC_T_dot_filt)
ax.set_ylabel('$ \partial C_T / \partial \psi$')
ax.set_xlabel('$\psi [\circ]$')
ax.grid()
ax.legend(['Original','Filtered'])


fig,ax = plt.subplots(1,2,figsize = (8,6))
plt.subplots_adjust(wspace = 0.3,bottom = 0.1)
ax[0].plot(data_dict['"l"'])
ax[0].set_ylabel('$  C_T$')
ax[0].set_xlabel('$\psi [\circ]$')
ax[0].grid()

ax[1].plot(f,10*np.log10(np.real(Gxx_l)*df))
ax[1].set_ylabel('$PSD \ [dB]$')
ax[1].set_xlabel('$Frequency \ [Hz]$')
ax[1].grid()


fig,ax = plt.subplots(1,2,figsize = (8,6))
plt.subplots_adjust(wspace = 0.3,bottom = 0.1)
ax[0].plot(data_dict['"dC_T_dot"'])
ax[0].set_ylabel('$ \partial C_T / \partial \psi$')
ax[0].set_xlabel('$\psi [\circ]$')
ax[0].grid()

ax[1].plot(f,10*np.log10(np.real(Gxx_dCT)*df))
ax[1].set_ylabel('$PSD \ [dB]$')
ax[1].set_xlabel('$Frequency \ [Hz]$')
ax[1].grid()

#%%

df = 1
f_max = 3e3
f = np.arange(int(f_max/df))*df
f_target = np.arange(-3,3)+fc

L = a0/fc*0.25
a_n ,a_c,L_n , L_c = L/6,L/4,L/2,L/2
helm1 = res.resonator(a_n = a_n,a_c = a_c,L_n =L_n, L_c = L_c)
# helm1.minimize_Z(fc,lb=[0,0,0,0,0,0],ub=[c/2,c/2,R/2,R/2,V0,1])
helm1.minimize_Z(fc,lb=[0,0,0,0,0,0,0],ub=[c/4,c/4,R,R,R,V0,1])
helm1.set_Z(f[1:],model = 'WG',rad = True)

a_n ,a_c,L_n , L_c = 0.006286,0.006286,1e-10,0.04445
helm1 = res.resonator(a_n = a_n,a_c = a_c,L_n =L_n, L_c = L_c)
# helm1.minimize_Z(fc,lb=[0,0,0,0,0,0],ub=[c/2,c/2,R/2,R/2,V0,1])
helm1.set_Z(f[1:],model = 'WG',rad = True,interior = False)

helm2 = res.resonator(a_n = a_n,a_c = a_c,L_n =L_n, L_c = L_c)
# helm1.minimize_Z(fc,lb=[0,0,0,0,0,0],ub=[c/2,c/2,R/2,R/2,V0,1])
helm2.set_Z(f[1:],model = 'Kirchoff',rad = False,interior = True)

fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].plot(f[1:],np.real(helm1.Z)/(rho*a0))
# ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([500,3e3])
ax[0].set_ylim([0,.15])
ax[0].grid()

ax[1].tick_params(axis = 'x', labelsize=0)
ax[1].plot(f[1:],np.imag(helm1.Z)/(rho*a0))
# ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([500,3e3])
ax[1].set_ylim([-5,5])

ax[1].grid()

ax[-1].plot(f[1:],helm1.alpha)
ax[-1].set_ylabel(r'$\alpha$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([500,3e3])
# ax[-1].set_xlim([100,4e3])
# ax[-1].set_ylim([-180, 180])
ax[-1].legend(['Waveguide','Basic Acoustic Element','FEM'])

ax[-1].semilogx(f[1:],np.angle(helm1.Z)*180/np.pi)
ax[-1].set_ylabel('$Phase \ [\circ]$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([100,4e3])
ax[-1].set_ylim([-180, 180])
ax[-1].legend(['Waveguide','Basic Acoustic Element','FEM'])



#%%

#ratio of the resonator intlet to that of the entire sample
# sigma_i = (np.pi*helm1.a_n**2)/0.00258064
sigma_i = .35**2/2**2

# smeered impedance
# Z_tot = (2*sigma_i*(helm1.Z/(rho*a0))**-1)**-1
Z_tot = np.sum((sigma_i*(helm1.Z/(rho*a0))**-1,sigma_i*(helm2.Z/(rho*a0))**-1),axis = 0)**-1
# Z_tot_2 = -(25*sigma_i*np.tan(2*np.pi*f[1:]/a0*0.0857504))**-1

fig,ax = plt.subplots(3,1, figsize = (6.4,4.5))
plt.subplots_adjust(bottom = 0.15)
ax[0].tick_params(axis = 'x', labelsize=0)
ax[0].plot(f[1:],np.real(Z_tot))
# ax[0].set_ylabel('$Re[Y] \ [Pa \ s/m^3]$')
ax[0].set_ylabel(r'$Resistance, \ \overline{\theta}$')
ax[0].set_xlim([1400,2200])
ax[0].set_ylim([0,.15])
ax[0].grid()

ax[1].tick_params(axis = 'x', labelsize=0)
ax[1].plot(f[1:],np.imag(Z_tot))
# ax[1].plot(f[1:],Z_tot_2)

# ax[1].set_ylabel('$Im[Y] \ [Pa \ s/m^3]$')
ax[1].set_ylabel(r'$Reactance, \ \overline{\chi}$')
ax[1].set_xlim([1400,2200])
ax[1].set_ylim([-5,5])

ax[1].grid()

ax[-1].plot(f[1:],helm1.alpha)
ax[-1].set_ylabel(r'$\alpha$')
ax[-1].set_xlabel('Frequency [Hz]')
ax[-1].grid()
ax[-1].set_xlim([1400,2200])
# ax[-1].set_xlim([100,4e3])
# ax[-1].set_ylim([-180, 180])
ax[-1].legend(['Waveguide','Basic Acoustic Element','FEM'])

