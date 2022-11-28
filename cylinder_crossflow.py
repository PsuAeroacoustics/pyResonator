import numpy as np
import matplotlib.pyplot as plt

#%%

M_h = 0.664
Re = 0.664*340*0.75*.1039/14.88e-6
Re = 1e5

St = .1848+8.6e-4*(Re/1.5e5)**4.6
# CL_rms = .52-0.06*np.log10(Re/1.6e3)**-2.6
CL_rms = 0.09+0.43*np.exp(-10**5*(Re/10**6)**10)

