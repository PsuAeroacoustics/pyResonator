import numpy as np
import resonator as res
import matplotlib.pyplot as plt
import os
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize,least_squares

#%%
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ["Times New Roman"]
plt.rcParams['font.size'] = 12

#%%

validation_dir = os.path.join(os.getcwd(),'validation_data')
validation_data_fname = ['qf130_ReZ', 'qf130_ImZ','qf130_ReGamma', 'qf130_ImGamma','m1_ReZ','m1_ImZ','mfoam_85in_Re_Z','mfoam_85in_Im_Z','mfoam_170in_Re_Z','mfoam_170in_Im_Z']
qf130_Re_Z, qf130_Im_Z,qf130_Re_Gamma, qf130_Im_Gamma,m1_Re_Z,m1_Im_Z ,mfoam_85in_Re_Z,mfoam_85in_Im_Z,mfoam_170in_Re_Z,mfoam_170in_Im_Z= list(map(lambda x: np.loadtxt(os.path.join(validation_dir,f'{x}.csv'),delimiter=",", dtype=str)[1:].astype(float),validation_data_fname))


#%%
df = 10
f_max = 6.4e3
# frequency array
f = np.arange(1,int(f_max/df))*df

# density [kg/m^3]
rho =  1.215
# speed of sound [m/s]
c = np.sqrt(1.4*287.05*(273.15+18))
# Prandtl number 
Pr = 0.71


#%% Filtros Qf-130 porous ceramic - Wilson, D. Keith. "Simple, relaxational models for the acoustical properties of porous media." 

# thickness
t = 7.58e-2
# porosity - ratio of the volume of air in the pores to the total volume of the sample 
phi = 0.432
# flow resistivity [MKS rayls/m]
sigma = 4.45e4
# torousity
q = np.sqrt(1.7)

# Delancy-Bazley empirical model - if porosity is provided will use the Biot-Allard impedance model otherwise will use the Delancy-Bazley empirical model
Qf_130_DB = res.resonator(t =t,sigma = sigma,c = c,rho =rho,Pr = Pr)
Qf_130_DB.set_Z(f)
# Biot-Allard 4-parameter model
Qf_130_BA = res.resonator(t =t,sigma = sigma,phi = phi, q= q,c = c,rho =rho,Pr = Pr)
Qf_130_BA.set_Z(f)


fig,ax = plt.subplots(2,1, figsize = (5,6.5))
fig.subplots_adjust(left = 0.15)
ax[0].plot(f,np.real(Qf_130_DB.Z_c)/(rho*c))
ax[0].plot(f,np.real(Qf_130_BA.Z_c)/(rho*c))
ax[0].scatter(qf130_Re_Z[:,0],qf130_Re_Z[:,-1],c = 'black')
ax[0].set_ylabel(r'$Re[Z_c/\rho c]$')
ax[1].plot(f,np.imag(Qf_130_DB.Z_c)/(rho*c))
ax[1].plot(f,np.imag(Qf_130_BA.Z_c)/(rho*c))
ax[1].scatter(qf130_Im_Z[:,0],qf130_Im_Z[:,-1],c = 'black')
ax[1].set_ylabel(r'$Im[Z_c/\rho c]$')
ax[1].set_xlabel(r'$Frequency \ [Hz]$')
ax[1].legend(['Delany-Bazley','Biot-Allard','Measured'])
ax[0].set_ylim([0,10])
ax[1].set_ylim([-10,0])
for i in range(2):
    # ax[i].set_ylim([0.1,100])
    ax[i].set_xlim([50,f_max])
    ax[i].set_xscale('log')
    # ax[i].set_yscale('log')
    ax[i].grid()
ax[0].set_xticklabels([])

fig,ax = plt.subplots(2,1, figsize = (5,6.5))
fig.subplots_adjust(left = 0.15)
ax[0].plot(f,np.real(Qf_130_DB.Gamma))
ax[0].plot(f,np.real(Qf_130_BA.Gamma))
ax[0].scatter(qf130_Re_Gamma[:,0],qf130_Re_Gamma[:,-1]*100,c = 'black')
ax[0].set_ylabel(r'$Re[\Gamma] \ [m^{-1}]$')
ax[1].plot(f,np.imag(Qf_130_DB.Gamma))
ax[1].plot(f,np.imag(Qf_130_BA.Gamma))
ax[1].scatter(qf130_Im_Gamma[:,0],qf130_Im_Gamma[:,-1]*100,c = 'black')
ax[1].set_ylabel(r'$Im[\Gamma] \ [m^{-1}]$')
ax[1].set_xlabel(r'$Frequency \ [Hz]$')
ax[1].legend(['Delany-Bazley','Biot-Allard','Measured'])
# ax[0].set_ylim([0,10])
# ax[1].set_ylim([-10,0])
for i in range(2):
    # ax[i].set_ylim([0.1,100])
    ax[i].set_xlim([50,f_max])
    ax[i].set_xscale('log')
    # ax[i].set_yscale('log')
    ax[i].grid()
ax[0].set_xticklabels([])


#%%
# Coors porous ceramic #100: Champoux, Yvan, and Michael R. Stinson. "On acoustical models for sound propagation in rigid frame porous materials and the influence of shape factors.

# thickness
t = 7.59e-2
# porosity - ratio of the volume of air in the pores to the total volume of the sample 
phi = 0.436
# flow resistivity [MKS rayls/m]
sigma = 380000
# torousity
q = np.sqrt(2)

# Delancy-Bazley empirical model - if porosity is provided will use the Biot-Allard impedance model otherwise will use the Delancy-Bazley empirical model
C_100_DB = res.resonator(t =t,sigma = sigma,c = c,rho =rho,Pr = Pr)
C_100_DB.set_Z(f)
# Biot-Allard 4-parameter model
C_100_BA = res.resonator(q=q,s_b = 1.37,t =t,sigma = sigma,phi = phi,c =c,rho =rho,Pr = Pr)
C_100_BA.set_Z(f)

fig,ax = plt.subplots(2,1, figsize = (5,6.5))
ax[0].set_xticklabels([])
ax[0].plot(f,np.real(C_100_DB.Z_c)/(rho*c))
ax[0].plot(f,np.real(C_100_BA.Z_c)/(rho*c))
ax[0].set_ylabel(r'$Re[R/(\rho c)]$')
ax[1].plot(f,np.imag(C_100_DB.Z_c)/(rho*c))
ax[1].plot(f,np.imag(C_100_BA.Z_c)/(rho*c))
ax[1].set_ylabel(r'$Im[R/(\rho c)]$')
ax[1].legend(['Delancy-Bazley','Biot-Allard'])
# ax[0].set_ylim([0,10])
# ax[1].set_ylim([-10,0])
for i in range(2):
    ax[i].set_xlim([50,5e3])
    ax[i].set_xscale('log')
    ax[i].grid()

fig,ax = plt.subplots(2,1, figsize = (5,6.5))
ax[0].set_xticklabels([])
ax[0].plot(f,np.real(C_100_DB.Gamma*2*np.pi*f/c/100))
ax[0].plot(f,np.real(C_100_BA.Gamma*2*np.pi*f/c/100))
ax[0].set_ylabel(r'$Re[\Gamma c/\omega]-1$')
ax[1].plot(f,np.imag(C_100_DB.Gamma*2*np.pi*f/c/100))
ax[1].plot(f,np.imag(C_100_BA.Gamma*2*np.pi*f/c/100))
ax[1].set_ylabel(r'$Im[\Gamma c/\omega]$')
ax[1].legend(['Delancy-Bazley','Biot-Allard'])
# ax[0].set_ylim([0,.4])
# ax[1].set_ylim([0,1.5])
for i in range(2):
    ax[i].set_xlim([10,5e3])
    ax[i].set_xscale('log')
    ax[i].grid()


#%% single-layer fiber metal sheets #1: Bo, Zhang, and Chen Tianning. "Calculation of sound absorption characteristics of porous sintered fiber metal."

# thickness
t = 23.31e-3
# porosity - ratio of the volume of air in the pores to the total volume of the sample 
phi = .9094
# flow resistivity [MKS rayls/m]
sigma = 18980
# torousity
q = np.sqrt(1.513)
s_b = 1.0142

# # Delancy-Bazley empirical model - if porosity is provided will use the Biot-Allard impedance model otherwise will use the Delancy-Bazley empirical model
M_1_DB = res.resonator(t =t,sigma = sigma,c = c,rho =rho)
M_1_DB.set_Z(f)
# Biot-Allard 4-parameter model
M_1_BA = res.resonator(q=q,s_b =s_b,t =t,sigma = sigma,phi = phi,c =c,rho =rho,Pr = Pr)
M_1_BA.set_Z(f)

M_1_BA_tuned = res.resonator(q=q,s_b =s_b,t =t,sigma = sigma,phi = phi,c =c,rho =rho,Pr = Pr)
M_1_BA_tuned.tune_params(f,m1_Re_Z,m1_Im_Z,bnds = ([1e3,0.2,0.2,.8],[5e6,4,4,1]))


fig,ax = plt.subplots(2,1, figsize = (5,6.5))
fig.subplots_adjust(left = 0.15)
ax[0].plot(f,np.real(M_1_DB.Z)/(rho*c))
ax[0].plot(f,np.real(M_1_BA.Z)/(rho*c))
ax[0].plot(f,np.real(M_1_BA_tuned.Z)/(rho*c))
ax[0].scatter(m1_Re_Z[:,0],m1_Re_Z[:,-1],c = 'black')
ax[0].set_ylabel(r'$Re[Z/\rho c]$')
ax[1].plot(f,np.imag(M_1_DB.Z)/(rho*c))
ax[1].plot(f,np.imag(M_1_BA.Z)/(rho*c))
ax[1].plot(f,np.imag(M_1_BA_tuned.Z)/(rho*c))
ax[1].scatter(m1_Im_Z[:,0],m1_Im_Z[:,-1],c = 'black')
ax[1].set_ylabel(r'$Im[Z/\rho c]$')
ax[1].set_xlabel(r'$Frequency \ [Hz]$')
ax[1].legend(['Delany-Bazley','Biot-Allard','Biot-Allard Tuned','Measured'])
for i in range(2):
    ax[i].set_ylim([-5,5])
    ax[i].set_xlim([1,6.4e3])
    ax[i].grid()
ax[0].set_xticklabels([])


#%% cobalt metallic foam (ASTM F90) - Sutliff, Daniel L., Michael G. Jones, and Thomas C. Hartley. "High-speed turbofan noise reduction using foam-metal liner over-the-rotor." 

t = 21.59e-3
mfoam = res.resonator(q=q,s_b =s_b,t =t,sigma = sigma,phi = phi,Pr = 0.71,c =c,rho =rho)
mfoam.tune_params(f,mfoam_85in_Re_Z,mfoam_85in_Im_Z,bnds = ([1e3,0.5,0.5,.2],[1e5,3,3,1]))

mfoam.t = 43.18e-3
mfoam.set_Z(f)

fig,ax = plt.subplots(2,1, figsize = (5,6.5))
fig.subplots_adjust(left = 0.15)
ax[0].plot(f,np.real(mfoam.Z)/(rho*c))
ax[0].scatter(mfoam_170in_Re_Z[:,0],mfoam_170in_Re_Z[:,-1],c = 'black')
ax[0].set_ylabel(r'$Re[Z/\rho c]$')
ax[1].plot(f,np.imag(mfoam.Z)/(rho*c))
ax[1].scatter(mfoam_170in_Im_Z[:,0],mfoam_170in_Im_Z[:,-1],c = 'black')

ax[1].set_ylabel(r'$Im[Z/\rho c]$')
ax[1].set_xlabel(r'$Frequency \ [Hz]$')
ax[1].legend(['Predicted','Measured'])
for i in range(2):
    ax[i].set_ylim([-5,3])
    ax[i].set_xlim([1,3.5e3])
    ax[i].grid()
ax[0].set_xticklabels([])

