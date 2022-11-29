import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import resonator as res
import os
from scipy.signal import butter
from scipy.optimize import differential_evolution,NonlinearConstraint
import sys
from time import time
sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()),'pyPostAcs'))
import pyPostAcsFun as fun

#%%

def init_res(f,a_n,L_n,a_c,L_c):
    '''
    This function creates and returns a resonator object with its complex impedance evaluated
    '''
    res_temp = res.resonator(a_n = a_n,L_n =L_n,a_c = a_c, L_c = L_c)
    res_temp.set_Z(f,model = 'k',rad = False,interior = False,loss = True,table = False)
    return res_temp

def smeared_Z(f,res_params):
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

    Z = np.array([init_res(f,a_n = x[1],L_n = 1e-10,a_c = x[1],L_c = x[2]).Z for x in res_params]).transpose()
    Z_tot = np.sum(np.round(res_params[:,0])/A_s*np.pi*res_params[:,1]**2*Z**-1,axis = -1).transpose()**-1
    alpha = 1 - abs((Z_tot-1)/(Z_tot+1))**2

    return Z_tot, alpha

def opt_res(opt_in):

    '''
    This is the opjective function for the global  optimizer. Essentially the opjective is to minimize the normalized L2 error of the acoustic absorption.  
    '''
    # reshapes the array of input parameters so that each row corresponds to a different type of resonator
    opt_in = opt_in.reshape(int(len(opt_in)/3),3,order = 'F')
    
    # initializes an zero array for the total smeered admittance (inverse of impedance)
    Z_tot, alpha = smeared_Z(f,opt_in)
    L2_err = np.sum(W*(1-alpha)**2)/np.sum(W)

    print(L2_err)

    if np.isnan(L2_err):
        raise Exception('nan encountered')

    return L2_err

#%%

# Frequency array vector
df = 1
f_max = 3e3
f = np.arange(1,int(f_max/df))*df

# Density [kg/m^2]
rho = 1.125
# SoS [m/s]
a0 = 340

# Total area of the test sample that fits in the the LaRC NIT facility (2"x2").
A_s = 2*2*0.0254**2

# Generates frequency weighting function for total L2 error. BVI is concentrated in mid-range frequencies, 
# therefore, the weighting function is the magnitude of the impulse response of a 4th order bandpass butterworth 
# filter with cutoff frequendies of 500Hz and 1500Hz.
n,d = butter(4,  [500,1500] ,btype = 'bp',fs = 2*df*len(f))
f2,y,h,phase = fun.filt_response(n,d,fs = df*len(f),N = 2*len(f) ,plot=False)
W = np.abs(h)

#%%

# Total number of resonators in the test sample
N = 25
# number of different types of resonators within the sample
n = 3

# initial values of the resonators
a_n_init,L_n_init,a_c_init,L_c_init = 0.003809,0.003809*4,0.003809,0.003809*4
# Since the optimizer requires a 1D array consisting of the total number of parameters (a_n,L_n,a_c,L_c,# of respontors) x n, that array is generated here 
# opt_in = np.array([int(N/n),a_n_init,L_n_init,a_c_init,L_c_init]).repeat(n)
opt_in = np.array([int(N/n),a_c_init,L_c_init]).repeat(n)
opt_bounds = np.array([[1,25],[a0/(100*2*np.pi*f[-1]),a0/(2*np.pi*f[-1])],[a0/(2*np.pi*f[-1]),50*a0/(2*np.pi*f[-1])]]).repeat(n,axis = 0)

def get_constraints(lb,ub):
    constraints = NonlinearConstraint(fun = lambda x: [np.sum(np.round(x[:n])),np.sum(np.round(x[:n])*np.pi*x[n:n+n]**2)],lb =lb,ub = ub)
    return constraints

start_t = time()
res_optimal = differential_evolution(opt_res,bounds = opt_bounds,constraints =get_constraints([1,np.pi*(a0/(2*np.pi*f[-1]))**2],[N,A_s]),popsize=100, updating = 'immediate',polish=False,workers = 1)
elapsed_t = time()-start_t
print(elapsed_t)

res_optimal.x

/