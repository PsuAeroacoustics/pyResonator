import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import resonator as res
import os
import re
from scipy.signal import butter
from scipy.optimize import differential_evolution,NonlinearConstraint
import sys
sys.path.insert(0, '/Users/danielweitsman/codes/github/DanWeitsman/pyPostAcs')
import pyPostAcsFun as fun

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
n = 5

# initial values of the resonators
a_n_init,L_n_init,a_c_init,L_c_init = 0.003809,0.003809*4,0.003809,0.003809*4
# Since the optimizer requires a 1D array consisting of the total number of parameters (a_n,L_n,a_c,L_c,# of respontors) x n, that array is generated here 
# opt_in = np.array([int(N/n),a_n_init,L_n_init,a_c_init,L_c_init]).repeat(n)
opt_in = np.array([int(N/n),a_c_init,L_c_init]).repeat(n)
opt_bounds = np.array([[1,25],[a0/(100*2*np.pi*f[-1]),a0/(2*np.pi*f[-1])],[a0/(2*np.pi*f[-1]),50*a0/(2*np.pi*f[-1])]]).repeat(n,axis = 0)

def get_constraints(lb,ub):
    constraints = NonlinearConstraint(fun = lambda x: [np.sum(np.round(x[:n])),np.sum(np.round(x[:n])*np.pi*x[n:n+n]**2)]+list(x[n:n+n]),lb =lb,ub = ub)
    return constraints


# opt_bounds = np.array([[1,5],[1e-10,a_n_init*10],[1e-10,a_n_init*100],[1e-10,a_n_init*10],[1e-10,a_n_init*100]]).repeat(n,axis = 0)

L2_err_coll = []

def opt_res(opt_in):
    # reshapes the array of input parameters so that each row corresponds to a different type of resonator
    opt_in = abs(opt_in).reshape(int(len(opt_in)/3),3,order = 'F')
    # initializes an zero array for the total smeered admittance (inverse of impedance)
    beta_tot = np.zeros(len(f))
    for i in range(len(opt_in)):
        res_temp = res.resonator(a_n = opt_in[i,1],L_n =1e-10,a_c = opt_in[i,1], L_c = opt_in[i,2])
        res_temp.set_Z(f,model = 'k',rad = False,interior = False,loss = True,table = False)
        beta_tot = beta_tot+np.round(opt_in[i,0])/A_s*np.pi*res_temp.a_n**2*(res_temp.Z)**-1

    alpha = 1 - abs((beta_tot**-1-1)/(beta_tot**-1+1))**2
    L2_err = np.sum(W*(1-alpha)**2)/np.sum(W)
    print(L2_err)
    if np.isnan(L2_err):
        raise Exception('nan encountered')
    return L2_err

integ = np.ones(len(opt_in))*False
integ[:n] = True
res_optimal = differential_evolution(opt_res,bounds = opt_bounds,constraints =get_constraints([1,0]+list(np.ones(n)*a0/(100*2*np.pi*f[-1])),[N,A_s]+list(np.ones(n)*a0/(2*np.pi*f[-1]))),maxiter = 3,popsize=100,recombination=0.75,mutation = 0,updating = 'immediate',polish=True)

res_optimal.x