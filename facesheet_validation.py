import numpy as np
import matplotlib.pyplot as plt
import resonator as res
import os

#%%

cmap = plt.cm.Spectral.reversed()
fontName = 'Times New Roman'
fontSize = 14
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%

df = 1
f_max = 5e3
f = np.arange(1,f_max/df+df)*df

t,r,phi = 1e-3 ,0.5e-3,0.07
A_s = np.pi*r**2/phi


fs1  = res.fs(t = 1e-3,r = 0.5e-3,phi = 0.07)
fs1.set_Z(f)

L = 60e-3
# Z = fs1.get_Z()-1j*fs1.rho*fs1.c*np.arctan(fs1.w/fs1.c*L)
Z = fs1.get_Z()+1.4*1.125/(1j*fs1.w*A_s*L)

alpha = 1 - abs((Z/(fs1.c*fs1.rho)-1)/(Z/(fs1.c*fs1.rho)+1))**2

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.plot(f,alpha)
ax.set_ylabel(r'$Absorption, \ \alpha$')
ax.set_xlabel('Frequency [Hz]')
ax.grid()
ax.set_xlim([f[0],f[-1]])
ax.set_ylim([0, 1])
plt.savefig(os.path.join(os.getcwd(),'fs_val.png'),format = 'png')
