import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import resonator as res
import os
from scipy.optimize import differential_evolution
import random
# import sys
# from time import time
# import h5py
# sys.path.insert(0, os.path.join(os.path.dirname(os.getcwd()),'pyPostAcs'))
# import pyPostAcsFun as fun

#%%

cmap = plt.cm.Spectral.reversed()
fontName = 'Times New Roman'
fontSize = 14
plt.rc('font',**{'family':'serif','serif':[fontName],'size':fontSize})
plt.rc('mathtext',**{'default':'regular'})
plt.rc('text',**{'usetex':False})
plt.rc('lines',**{'linewidth':2})

#%%

# add sample (Volume limits, # of resonators)
# add sub-object for each resonator (opening position,bend,)
class sample():
    def __init__(self,V_xlim,V_ylim,V_zlim):
        self.V_xlim = V_xlim
        self.V_ylim = V_ylim
        self.V_zlim = V_zlim

class resonator(sample):
    def __init__(self,L,n_pos):
        # super().__init__(V_xlim,V_ylim,V_zlim)   
        self.L = L
        self.n_pos = n_pos
    
res_in = np.array([15,12])
neck_pos = np.array(([1,1,2],[1, 2, 2]))

def set_bends(L,N_bends):
    bend = np.zeros(L)
    for i in range(N_bends):
        bend[random.randrange(1,L-1)] = random.choice((-2,-1,1,2))
    return bend

V_xlim, V_ylim,V_zlim = 2,2,2



resonators = {f'res{i}':resonator(L = res_in[i],n_pos = neck_pos[i]) for i in range(int(len(res_in)))}

def opt_fun(bends):

    for i,k in enumerate(resonators):
        if i == 0:
            resonators[k].bend = np.insert(np.round(bends[:resonators[list(resonators.keys())[i]].L-1]),0,0)
        else:
            resonators[k].bend = np.insert(np.round(bends[resonators[list(resonators.keys())[i-1]].L-1:resonators[list(resonators.keys())[i-1]].L+resonators[list(resonators.keys())[i]].L-1]),0,0)

    route_res()
    N_bends = np.count_nonzero(np.round(bends))
    N_intersect = count_intersect()
    N_out = count_out_bound()
    err =  N_intersect+N_out
    print(err)
    return err

def route_res(): 

    for k,v in resonators.items():
        # initiates position of the  resonator just arranged vertically to start
        v.res_pos = v.n_pos-np.concatenate((np.zeros((v.L,2)),np.expand_dims(np.arange(v.L),axis = -1)),axis = -1)

        bend_ind = np.squeeze(np.where(v.bend != 0))
        if bend_ind.shape ==():
            bend_ind = np.expand_dims(bend_ind,axis = -1) 

        for b in bend_ind:

            res_dir = np.squeeze(np.where(np.squeeze(np.diff(v.res_pos[b-1:b+1],axis = 0) !=0)))
            v.res_pos[b:,res_dir] = v.res_pos[b,res_dir]

            if abs(res_dir) == 0:
                if abs(v.bend[b]) == 1:
                    v.res_pos[b:,1] = v.res_pos[b,1] + v.bend[b]*np.arange(v.L-b)
                if abs(v.bend[b]) == 2:
                    v.res_pos[b:,2] = v.res_pos[b,2] + v.bend[b]/2*np.arange(v.L-b)

            if abs(res_dir) == 1:
                if abs(v.bend[b]) == 1:
                    v.res_pos[b:,0] = v.res_pos[b,0] + v.bend[b]*np.arange(v.L-b)
                if abs(v.bend[b]) == 2:
                    v.res_pos[b:,2] = v.res_pos[b,2] + v.bend[b]/2*np.arange(v.L-b)

            if abs(res_dir) == 2:
                if abs(v.bend[b]) == 1:
                    v.res_pos[b:,0] = v.res_pos[b,0] + v.bend[b]*np.arange(v.L-b)
                if abs(v.bend[b]) == 2:
                    v.res_pos[b:,1] = v.res_pos[b,1] + v.bend[b]/2*np.arange(v.L-b)

def count_intersect():
    intersect = np.zeros((V_xlim+1,V_ylim+1,V_zlim+1))
    for z in range(V_zlim+1):
        for y in range(V_ylim+1):
            for x in range(V_xlim+1):
                intersect[x,y,z] = np.sum([np.count_nonzero(np.sum(v.res_pos ==[x,y,z],axis = -1)==3) for k,v in resonators.items()])
    N_intersect = np.count_nonzero(intersect>1)
    return N_intersect

def count_out_bound():    
    N_out = np.sum([np.count_nonzero(np.sum(np.array((((v.res_pos[:,0] >V_xlim) | (v.res_pos[:,0] <0)) ,((v.res_pos[:,1] > V_ylim) | (v.res_pos[:,1] <0)) ,((v.res_pos[:,2] >V_zlim) | (v.res_pos[:,2] <0)) )),axis = 0) > 0)-1 for k,v in resonators.items()])
    return N_out

# out_bound = resonators['res0'].res_pos[np.sum(np.array((((resonators['res0'].res_pos[:,0] >= V_xlim) | (resonators['res0'].res_pos[:,0] <=0)) ,((resonators['res0'].res_pos[:,1] >= V_ylim) | (resonators['res0'].res_pos[:,1] <=0)) ,((resonators['res0'].res_pos[:,2] >= V_zlim) | (resonators['res0'].res_pos[:,2] <=0)) )),axis = 0) >0]


L_tot = np.sum([v.L for k,v in resonators.items()])
opt_bounds = [[-2,2]]*(L_tot-len(resonators))
opt_pack = differential_evolution(opt_fun,bounds = opt_bounds,polish=True,workers = 1)
opt_fun(opt_pack.x)
#%%

ax = plt.axes(projection='3d')
for k,v in resonators.items():
    ax.plot(v.res_pos[:,0], v.res_pos[:,1], v.res_pos[:,2])

# ax.scatter(np.squeeze(np.where(intersect>1))[0],np.squeeze(np.where(intersect>1))[1],np.squeeze(np.where(intersect>1))[2])
# ax.scatter(out_bound[:,0],out_bound[:,1],out_bound[:,2])

ax.plot([V_xlim,V_xlim],[0,V_ylim],[0,0],c = 'black',linewidth = 4)
ax.plot([V_xlim,V_xlim],[0,V_ylim],[V_zlim,V_zlim],c = 'black',linewidth = 4)
ax.plot([0,0],[0,V_ylim],[0,0],c = 'black',linewidth = 4)
ax.plot([0,0],[0,V_ylim],[V_zlim,V_zlim],c = 'black',linewidth = 4)

ax.plot([0,V_xlim],[V_ylim,V_ylim],[0,0],c = 'black',linewidth = 4)
ax.plot([0,V_xlim],[V_ylim,V_ylim],[V_zlim,V_zlim],c = 'black',linewidth = 4)
ax.plot([0,V_xlim],[0,0],[0,0],c = 'black',linewidth = 4)
ax.plot([0,V_xlim],[0,0],[V_zlim,V_zlim],c = 'black',linewidth = 4)

ax.plot([0,0],[0,0],[0,V_zlim],c = 'black',linewidth = 4)
ax.plot([V_xlim,V_xlim],[0,0],[0,V_zlim],c = 'black',linewidth = 4)
ax.plot([0,0],[V_ylim,V_ylim],[0,V_zlim],c = 'black',linewidth = 4)
ax.plot([V_xlim,V_xlim],[V_ylim,V_ylim],[0,V_zlim],c = 'black',linewidth = 4)

ax.grid()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
