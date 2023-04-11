import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush,heappop,heappushpop
from itertools import count
import time
from AnalyzeDegenGeom import AnalyzeDegenGeom
import os
import scipy.interpolate as interp
import h5py

#%%

class sample:
    def __init__(self,size = None, N_res = None,dx = None):
        self.size = size
        self.N_res = N_res
        self.occupied_nodes = deque()
        self.dx = dx

class resonator:
    def __init__(self,start_node = None,length = None,r = None):
        self.start_node =start_node
        self.length = length
        self.success = False
        self.r = r
    
    def get_path(self):
        if hasattr(self,'path'):
            path = np.flip(np.array([node.position for node in self.path]),axis = 0)
        # else:
        #     print("This resonator hasn't been routed")
            return path

class node(resonator):

    def __init__(self,f = None,parent = None, position = None,ind = None,count = None ,L = None, r = None):
        self.r = r
        self.parent = parent
        self.position = position
        self.ind = ind
        self.f = f
        self.count = count
        self.L = L

    def __lt__(self,other_node):
        return (self.f,self.count) < (other_node.f,other_node.count)

    def __gt__(self,other_node):
        return (self.f,self.count) > (other_node.f,other_node.count)

    def __eq__(self,other_node):
        # return tuple(self.position) == tuple(other_node.position)
        return np.linalg.norm(np.diff((self.position,other_node.position),axis = 0)) <= self.r+other_node.r


def trace_path(current_node,start_node):
    parent_set = deque()
    while current_node != start_node:
        current_node = current_node.parent
        parent_set.append(current_node) 
    return parent_set

def route(sample,resonator):

    open_set = []
    closed_set = deque()
    cnt = count()
    
    heappush(open_set,resonator.start_node)

    while len(open_set) > 0:

        current_node = heappop(open_set)
        current_node.r = resonator.r
        print(f'Current Node: {current_node.position} - Current Length: {current_node.L}')
        
        if current_node.L >= resonator.length:
            print('Path found!')
            resonator.path = trace_path(current_node,resonator.start_node)
            sample.occupied_nodes.extend(resonator.path)
            resonator.success = True
            break

        else:
            neighbors = current_node.position+stensile*liner.dx
            min_x,max_x,min_z,max_z = min_x_spline(neighbors[:,1:]),max_x_spline(neighbors[:,1:]),min_z_spline(neighbors[:,:2]),max_z_spline(neighbors[:,:2])
            min_x[np.isnan(min_x)] = np.inf
            max_x[np.isnan(max_x)] = -np.inf
            min_z[np.isnan(max_x)] = np.inf
            max_z[np.isnan(max_x)] = -np.inf

            bounds = ((neighbors[:,-1] >= max_z) | (neighbors[:,-1]<= min_z)) | ((neighbors[:,0] >= max_x) | (neighbors[:,0]<= min_x)) | ((neighbors[:,1] >= y_max) | (neighbors[:,1]<= y_min))

            # bounds = ((neighbors[:,0] <= sample.size[0]) | (neighbors[:,0] >= sample.size[1])) | ((neighbors[:,1] <= sample.size[2]) | (neighbors[:,1] >= sample.size[3])) | ((neighbors[:,2] <= sample.size[4]) | (neighbors[:,2] >= sample.size[5]))
            if np.any(bounds):
                neighbors = np.delete(neighbors,bounds,axis = 0)
            
            # if in closed set or already in open set remove from list 
            
            
            for i, neighbor in enumerate(neighbors):
                n = node(position = tuple(neighbor),r = current_node.r)
                if n in closed_set or n in sample.occupied_nodes:
                    neighbors = np.delete(neighbors,np.sum(neighbors==neighbor,axis = -1)==3,axis = 0)


            f = np.zeros(len(neighbors))
            if current_node != resonator.start_node:
                current_direction = np.diff((current_node.parent.position,current_node.position),axis = 0)
                bend_penelty_ind = np.sum((neighbors-current_node.position) ==  current_direction,axis = 1) != 3
                f[bend_penelty_ind] = f[bend_penelty_ind]+1
            
            for i,pos in enumerate(neighbors):
                temp_node = node(position=tuple(pos),parent=current_node,f = f[i], count = resonator.length-next(cnt))
                if current_node ==resonator.start_node:
                    temp_node.L = current_node.L+1*dx
                else:
                    temp_node.L = current_node.parent.L+2*dx

                # ind = np.squeeze(np.where([temp_node == existing_node for existing_node in open_set]))
                # if ind.size !=  0:
                #     if temp_node.f < open_set[ind].f:
                #         open_set[ind].f = temp_node.f
                #         # open_set[ind].remaining_dist = abs(L-L_current-L_remainder)
                #         open_set[ind].parent = temp_node.parent
                # else:
                heappush(open_set,temp_node)

            # print(f'{current_node.position} - {neighbors}')

            closed_set.append(current_node) 

    
#%%

dx = [0.1,.5,0.1]
bounds = np.array([-2.5,2.5,0,20,-1,1])
bound_range = bounds[1::2]- bounds[::2]
x,y,z = [np.arange(bound_range[i]/dx[i]+1)*dx[i]+bounds[::2][i] for i in range(3)]

# y,x,z
grid_coord = np.array(np.meshgrid(x,y,z))

grid = np.ones((grid_coord.shape[1:]+np.ones(3)).astype(int))
grid[0] = 0
grid[-1] = 0
grid[:,0] = 0
grid[:,-1] = 0
grid[:,:,0] = 0
grid[:,:,-1] = 0

grid_ind = np.array(np.meshgrid(np.arange(len(x)),np.arange(len(y)),np.arange(len(z))))

#%%

res_data = {}
with h5py.File(os.path.join(os.getcwd(),'res_opt.h5'),'r') as f:
    for k,v in f.items():
        # res_data['res_opt] = [# of each resonator, radius of neck, length of neck, radius of cavity, length of cavity] - for each resonator
        res_data = {**res_data,**{k:v[()]}}

r_max = np.max((np.max(res_data['res_opt'][:,1]),np.max(res_data['res_opt'][:,3])))
N_res = np.sum(np.round(res_data['res_opt'][:,0])) 
if N_res%2 != 0:
    N_res = N_res+1

L_tot = res_data['res_opt'][:,2]+res_data['res_opt'][:,-1]
r_tot =  np.max((res_data['res_opt'][:,1],res_data['res_opt'][:,3]),axis = 0)

N_res_x = 10
N_res_y = int(N_res/N_res_x)

res_spacing = (bounds[3]-bounds[2])/(N_res_y+1)
res_spacing_ind = int(np.floor(res_spacing/dx[1]))

# fix centering
y_res_ind = np.squeeze(np.where((y<bounds[3]) &(y>bounds[2])))[res_spacing_ind+1::res_spacing_ind][:N_res_y]
y_res = y[y_res_ind]

res_spacing = (bounds[1]-bounds[0])/(N_res_x+1)
res_spacing_ind = int(np.floor(res_spacing/dx[0]))
x_res_ind = np.squeeze(np.where((x<bounds[1]) &(x>bounds[0])))[res_spacing_ind-1::res_spacing_ind][:N_res_x]
x_res = x[x_res_ind]

res_coord = grid_coord[:,y_res_ind][:,:,x_res_ind][:,:,:,-1]
res_ind = grid_ind[:,y_res_ind][:,:,x_res_ind][:,:,:,-1]

#%%

N_type = len(res_data['res_opt'])
res_type =  np.tile(np.arange(4),int(N_res_y/N_type)+1)[:N_res_y]
res_type_full = np.zeros((N_res_x,N_res_y))

for i in range(N_res_x):
    res_type_full[i] = np.roll(res_type,-i)
res_type_full = res_type_full.astype(int)

liner = sample(size = [0,10,0,10,0,10],N_res = N_res,dx = dx)

res = {}
for iter_x in range(N_res_x):
    for iter_y in range(N_res_y):
        res_temp = resonator(start_node=node(position =res_coord[:,iter_y,iter_x],ind = res_ind[:,iter_y,iter_x],L = 0),length = L_tot[res_type_full[iter_x,iter_y]],r=r_tot[res_type_full[iter_x,iter_y]])
        res = {**res,**{f'res{iter_x*N_res_y+iter_y}':res_temp}}
liner.resonators = res

stensile = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])
[route(liner,v) for k,v in liner.resonators.items()]

neighbors = res['res0'].start_node.ind+stensile
neighbors = neighbors[grid[neighbors[:,0],neighbors[:,1],neighbors[:,-1]].astype(bool)]
grid_coord[:,neighbors[:,1],neighbors[:,0],neighbors[:,-1]]
np.take(grid_coord,neighbors)
