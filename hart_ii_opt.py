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

def init_res(f,a_n,L_n,a_c,L_c):
    '''
    This function creates and returns a resonator object with its complex impedance evaluated
    '''
    res_temp = res.resonator(a_n = a_n,L_n =L_n,a_c = a_c, L_c = L_c)
    res_temp.set_Z(f,model = 'k',rad = False,interior = False,loss = True,table = False)
    return res_temp

def init_fs(f,t,r,phi):
    '''
    This function creates and returns a resonator object with its complex impedance evaluated
    '''
    fs_temp = res.fs(t,r,phi)
    fs_temp.set_Z(f)
    return fs_temp


def smeared_Z(f,res_params,**kwargs):
    '''
    This function computes the smeared impedance and absorption for a sample that consists of multiple resonator cavities that have different geometries. 
    
    Parameters:
    f: frequency array [Hz]
    res_params: a nested array whose elements corresponding to each resonator in the sample. Each of the nested arrays must be of size (5,) and specifies
    [[# of this resonator in the sample, a_n,L_n,a_c,L_c]]

    Return:
    Z_tot: total non-dimensionalized (rho*c0) complex impedance of the sample (inverse of the total admittance)
    alpha: total absorption of the sample
    '''
    Z = np.array([init_res(f,a_n = x[1],L_n = x[2],a_c = x[3],L_c = x[4]).Z for x in res_params]).transpose()

    if kwargs['facesheet']:
        Z_fs = init_fs(f,t = kwargs['t'],r = kwargs['r'],phi = kwargs['N']*kwargs['r']**2/res_params[0,1]**2).Z
        Z_tot = np.sum(np.round(res_params[:,0])/A_s*np.pi*res_params[:,1]**2*(Z+np.expand_dims(Z_fs,axis = -1))**-1,axis = -1).transpose()**-1

    else:
        Z_tot = np.sum(np.round(res_params[:,0])/A_s*np.pi*res_params[:,1]**2*Z**-1,axis = -1).transpose()**-1
    
    alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2

    return Z_tot, alpha

def opt_res(opt_in,facesheet=True):

    '''
    This is the opjective function for the global  optimizer. Essentially the opjective is to minimize the normalized L2 error of the acoustic absorption.  
    '''
    # reshapes the array of input parameters so that each row corresponds to a different type of resonator
    if facesheet:

        fs_in = opt_in[-3:]
        opt_in = opt_in[:-3].reshape(int(len(opt_in[:-3])/5),5,order = 'F')
        Z_tot, alpha = smeared_Z(f,opt_in,facesheet = facesheet,t = fs_in[-3],N = np.round(fs_in[-2]), r = fs_in[-1])

    else:
        opt_in = opt_in.reshape(int(len(opt_in)/5),5,order = 'F')
        Z_tot, alpha = smeared_Z(f,opt_in,facesheet = facesheet)
        
    L2_err = np.sum(W*(1-alpha)**2)/np.sum(W)

    print(L2_err)

    if np.isnan(L2_err):
        print(opt_in)
        raise Exception( 'Nan Encountered')
        # opt_in[:,1] = a0/(20*2*np.pi*f[-1])
        # opt_res(opt_in.flatten(order = 'F'))
        # print('NaN Encountered!!!')
    return L2_err
    
#%%

elements = 48
dt = 0.00015994586253613723
iterations = 5400
df = (iterations*dt)**-1

# Frequency array vector
f = np.arange(1,iterations)*df

# Density [kg/m^2]
rho = 1.125
# SoS [m/s]
a0 = 340

#%% Operating condition corresponding to the maximum BVI case of the Hart II program

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
V0 = 0.7*(R-e)/48*t*c*6

# Blade surface area over which the resonators would be applied.
A_s = 0.25*c*((R-e)/48)

#%%%
# Generates frequency weighting function for total L2 error. BVI is concentrated in mid-range frequencies, 
# therefore, the weighting function is the magnitude of the impulse response of a 4th order bandpass butterworth 
# filter with cutoff frequendies of 500Hz and 1500Hz.
n,d = butter(4,  [400,1500] ,btype = 'bp',fs = 2*df*len(f))
f2,y,h,phase = fun.filt_response(n,d,fs = df*len(f),N = 2*len(f) ,plot=False)
W = np.abs(h)

#%%

# number of different types of resonators within the sample
n = 4

a_max = a0/(2*np.pi*f[-1])

# set to true in order to include the face sheet in the optimization 
facesheet = False

# min and max dimensions of the resonator neck and cavity
r_min, r_max, L_min, L_max = a0/(2*np.pi*2000)*.05,a0/(2*np.pi*2000),a0/(2*np.pi*2000),a0/(2*np.pi*350)

if facesheet:
    opt_bounds = np.concatenate((np.array([[1,1000],[r_min, r_max],[L_min, L_max],[r_min, r_max],[L_min, L_max]]).repeat(n,axis = 0),np.array([[0.05*0.0254,0.5*0.0254],[1,100],[5e-4,5e-4]])))
else:
    opt_bounds = np.array([[1,1000],[r_min, r_max],[L_min, L_max],[r_min, r_max],[L_min, L_max]]).repeat(n,axis = 0)


def get_constraints(lb,ub):
    # constraints = NonlinearConstraint(fun = lambda x: [np.sum(np.round(x[:n])),np.sum(np.round(x[:n])*np.pi*x[n:n+n]**2)],lb =lb,ub = ub)
    if facesheet:
        constraints = NonlinearConstraint(fun = lambda x: [np.sum(np.round(x[:n])*np.pi*x[n:n+n]**2),np.sum(np.round(x[:n])*(np.pi*x[n:n+n]**2*x[2*n:2*n+n]+np.pi*x[3*n:3*n+n]**2*x[4*n:4*n+n]))]+list(x[n:n+n]/x[3*n:3*n+n])+list(np.pi*x[-1]**2*np.round(x[-2])/(np.pi*x[n:n+n]**2)),lb =lb,ub = ub)
    else:
        constraints = NonlinearConstraint(fun = lambda x: [np.sum(np.round(x[:n])*np.pi*x[n:n+n]**2),np.sum(np.round(x[:n])*(np.pi*x[n:n+n]**2*x[2*n:2*n+n]+np.pi*x[3*n:3*n+n]**2*x[4*n:4*n+n]))]+list(x[n:n+n]/x[3*n:3*n+n]),lb =lb,ub = ub)
    return constraints

start_t = time()
# mutation = 0.05,recombination = 0.5
if facesheet:
     opt_output = differential_evolution(opt_res,args = (facesheet,),bounds = opt_bounds,constraints =get_constraints([0,0]+[0]*n+[0.07]*n,[A_s,V0]+[1]*n+[.07]*n),polish=False,workers = -1,maxiter = int(1e10))
else:
    opt_output = differential_evolution(opt_res,args = (facesheet,),bounds = opt_bounds,constraints =get_constraints([0,0]+[0]*n,[A_s,V0]+[1]*n),polish=False,maxiter = int(1e10))

elapsed_t = time()-start_t
print(elapsed_t)

if facesheet:
    res_opt = opt_output.x[:-3].reshape(int(len(opt_output.x[:-3])/5),5,order = 'F')
    Z_tot, alpha = smeared_Z(f,res_opt,facesheet = facesheet,t = opt_output.x[-3],N = np.round(opt_output.x[-2]), r = opt_output.x[-1])

else:
    res_opt = opt_output.x.reshape(int(len(opt_output.x)/5),5,order = 'F')
    Z_tot, alpha = smeared_Z(f,res_opt,facesheet = facesheet)

print(res_opt)
#%%

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.plot(f,alpha)
ax.set_ylabel(r'$Absorption, \ \alpha$')
ax.set_xlabel('Frequency [Hz]')
ax.grid()
ax.set_xlim([f[0],f[-1]])
ax.set_ylim([0, 1])
plt.savefig(os.path.join(os.getcwd(),'opt_res.png'),format = 'png')

save_dir = os.path.join(os.getcwd(),'res_opt.h5')
if os.path.exists(save_dir):
    os.remove(save_dir)

with h5py.File(save_dir, 'a') as h5_f:
    for k, v in {'f': f, 'res_opt': res_opt,'Z_tot': Z_tot,'alpha':alpha}.items():
        h5_f.create_dataset(k, shape=np.shape(v), data=v)

