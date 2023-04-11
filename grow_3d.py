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

    

dataSorted, indHeader = AnalyzeDegenGeom(os.path.join(os.getcwd(),'bo105_DegenGeom.csv'))
surfNodes = np.float64(dataSorted['Component 1']['SURFACE_NODE'][1:, :3])
nXsecs = int(indHeader[0][1][1])
pntsPerXsec = int(indHeader[0][1][2])

surfNodes = surfNodes.reshape((pntsPerXsec,nXsecs,3) ,order = 'F')
nXsecs = 4
Xsec = 30
surfNodes = surfNodes[:,Xsec:Xsec+nXsecs]


min_z_surfNodes = surfNodes[:int(pntsPerXsec/2)+3]
max_z_surfNodes = surfNodes[int(pntsPerXsec/2)+2:]

min_z_surfNodes = min_z_surfNodes.reshape((len(min_z_surfNodes)*nXsecs,3) ,order = 'F')
max_z_surfNodes = max_z_surfNodes.reshape((len(max_z_surfNodes)*nXsecs,3) ,order = 'F')
# surfNodes = surfNodes.reshape((pntsPerXsec*nXsecs,3) ,order = 'F')

max_x_spline = interp.LinearNDInterpolator(points = max_z_surfNodes[:,1:],values = max_z_surfNodes[:,0])
min_x_spline = interp.LinearNDInterpolator(points = min_z_surfNodes[:,1:],values = min_z_surfNodes[:,0])

y_min = np.min(surfNodes[:,:,1])
y_max = np.max(surfNodes[:,:,1])

x_min = np.min(surfNodes[:,:,0])
x_max = np.max(surfNodes[:,:,0])
# max_y_spline = interp.LinearNDInterpolator(points = np.array((max_z_surfNodes[:,0],max_z_surfNodes[:,-1])).transpose(),values = max_z_surfNodes[:,1])
# min_y_spline = interp.LinearNDInterpolator(points = np.array((min_z_surfNodes[:,0],min_z_surfNodes[:,-1])).transpose(),values = min_z_surfNodes[:,1])

LENodes = np.float64(dataSorted['Component 1']['STICK_NODE'][1:, :3])
TENodes = np.float64(dataSorted['Component 1']['STICK_NODE'][1:, 3:6])
c = np.linalg.norm(abs(LENodes - TENodes), axis=1)

dx = np.mean(c)/100
# dx = (y_max-y_min)/100

max_bound = np.max(np.max(surfNodes,axis = 0),axis = 0)+2*dx
min_bound = np.min(np.min(surfNodes,axis = 0),axis = 0)-2*dx
bound_range = max_bound-min_bound
dx = bound_range/np.round(bound_range/dx)

x,y,z = [np.squeeze(np.arange(bound_range[i]/dx[i]+1)*dx[i]+min_bound[i]) for i in range(3)]
x2,y2 = np.meshgrid(x,y)

grid_coord = np.array(np.meshgrid(x,y,z))
grid = np.ones(grid_coord.shape[1:])
grid_ind = np.array(np.meshgrid(np.arange(len(x)),np.arange(len(y)),np.arange(len(z))))

max_z_spline = interp.griddata(points = max_z_surfNodes[:,:-1],values =max_z_surfNodes[:,-1],xi =(grid_coord[0],grid_coord[1]),fill_value=0,method = 'linear')
min_z_spline = interp.griddata(points = min_z_surfNodes[:,:-1],values =min_z_surfNodes[:,-1],xi =(grid_coord[0],grid_coord[1]), fill_value=0,method = 'linear')

# grid[(np.isnan(min_z_spline(x2,y2))) & (np.isnan(max_z_spline(x2,y2)))]= 0
grid[min_z_spline > grid_coord[-1]] = 0
grid[max_z_spline < grid_coord[-1]] = 0

# res_start = [(3,3,10),(5,5,10),(4,4,10),(9,6,10)]
# res_L = [100,25,50,50]

# res_start = [surfNodes[16,3],surfNodes[16,4],(surfNodes[16,28][0],surfNodes[16,28][1]+2*dx,np.float(max_z_spline((surfNodes[16,28][0]-dx,surfNodes[16,28][1])))),surfNodes[18,10],surfNodes[16,11],surfNodes[17,11],surfNodes[18,11]]

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

N_rows = 7
N_columns = int(N_res/N_rows)

# if N_columns >= np.floor((y_max-y_min)/(2*r_max)):
#     while N_columns >= np.floor((y_max-y_min)/(2*r_max)):
#         N_rows=N_rows+1
#         N_columns = int(N_res/N_rows)
#         print(N_columns)

res_spacing = (y_max-y_min)/(N_columns+1)
res_spacing_ind = int(np.floor(res_spacing/dx[1]))
y_res_ind = np.squeeze(np.where((y<y_max) &(y>y_min)))[res_spacing_ind-1::res_spacing_ind][:N_columns]
y_res = y[y_res_ind]
# grid_coord[1][y_res_ind][:,x_res_ind][:,:,0]-y_res

res_spacing_ind = int(np.floor(res_spacing/dx[0]))
x_res_ind = np.squeeze(np.where(x<(x_min+0.1*np.mean(c))))[-1]+np.squeeze(np.where((x<x_max) &(x>x_min)))[::res_spacing_ind][:N_rows]
x_res = x[x_res_ind]

x_res,y_res = np.meshgrid(x_res,y_res)
# z_res = interp.griddata(points = max_z_surfNodes[:,:-1],values =max_z_surfNodes[:,-1],xi =(x_res,y_res),method = 'linear')

z_res = max_z_spline[y_res_ind][:,x_res_ind][:,:,0]
# z_res_ind = [np.squeeze(np.where(z_res[iter_column,iter_row]<z))[0] for iter_column in range(N_columns) for iter_row in range(N_rows)]
# grid_coord[-1][x_res_ind][:,y_res_ind].shape
z_res_ind = np.zeros((N_columns,N_rows))
for iter_row in range(N_rows):
    for iter_column in range(N_columns):
        z_res_ind[iter_column,iter_row] = int(np.squeeze(np.where(z_res[iter_column,iter_row]<z))[0])

res_type= np.zeros(z_res.shape)
for i in range(N_rows-1):
    res_type[:,i+1] = np.roll(res_type[:,i],int(len(res_data['res_opt'])/2))
res_type = res_type.astype(int)
# es_start = [surfNodes[16,2],surfNodes[17,2],surfNodes[16,1],surfNodes[17,1]]
# res_L = [20*dx,20*dx,20*dx,20*dx]
# r = [dx/4,dx/8]

liner = sample(size = [0,10,0,10,0,10],N_res = N_res,dx = dx)

res = {}
for iter_row in range(N_rows):
    for iter_column in range(N_columns):
        
        # z_res_ind = np.squeeze(np.where(z_res[iter_column,iter_row]<z))[0]
        res_temp = resonator(start_node=node(position = [x_res[iter_column,iter_row],y_res[iter_column,iter_row],z[int(z_res_ind[iter_column,iter_row])] ], ind = [x_res_ind[iter_row],y_res_ind[iter_column],int(z_res_ind[iter_column,iter_row])],L = 0),length = L_tot[res_type[iter_column,iter_row]],r=r_tot[res_type[iter_column,iter_row]])
        res = {**res,**{f'res{iter_row*N_columnsx.shap+iter_column}':res_temp}}
liner.resonators = res

stensile = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])

[route(liner,v) for k,v in liner.resonators.items()]
paths = [v.get_path() for k,v in liner.resonators.items()]


save_dir = os.path.join(os.getcwd(),'res_out')
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

for i,n in enumerate(paths):
    if n is not None:
        np.savetxt(os.path.join(save_dir,f'res{i}_p.txt'),X = n)

r_out = []
for k,v in liner.resonators.items():
    if v.success:
        r_out.append(v.r) 

np.savetxt(os.path.join(save_dir,f'res_radii_2.txt'),X = np.array(r_out),delimiter = ',')

with open(os.path.join(save_dir,f'res_radii.txt'),'w') as f:
    np.array(r_out).tofile(f,sep=' ',format='%.18e')

# with open(os.path.join(os.getcwd(),'res.txt'),'w') as f:
#     f.write(paths[0].astype('str'))
# parent_set = liner.resonators['res1'].get_path()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect(((np.max(surfNodes[:,:,0])-np.min(surfNodes[:,:,0]))/(np.max(surfNodes[:,:,0])-np.min(surfNodes[:,:,0])),(np.max(surfNodes[:,:,1])-np.min(surfNodes[:,:,1]))/(np.max(surfNodes[:,:,0])-np.min(surfNodes[:,:,0])),(np.max(surfNodes[:,:,-1])-np.min(surfNodes[:,:,-1]))/(np.max(surfNodes[:,:,0])-np.min(surfNodes[:,:,0]))))
for k,v in liner.resonators.items():
    if v.success:
        parent_set = v.get_path()
        ax.plot(parent_set[:,0],parent_set[:,1],parent_set[:,2],linewidth = 5)
        ax.scatter(parent_set[0,0],parent_set[0,1],parent_set[0,2],c='r',marker='*')
        # ax.scatter(parent_set[-1,0],parent_set[-1,1],parent_set[-1,2],c='r',marker='^')
ax.plot_surface(surfNodes[:,:,0],surfNodes[:,:,1],surfNodes[:,:,2],alpha = .2)
# ax.set_xlim(liner.size[0],liner.size[1])
# ax.set_ylim(liner.size[2],liner.size[3])
# ax.set_zlim(liner.size[4],liner.size[5])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.grid()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(max_z_surfNodes[:,0],max_z_surfNodes[:,1],max_z_surfNodes[:,2])
ax.plot(min_z_surfNodes[:,0],min_z_surfNodes[:,1],min_z_surfNodes[:,2])
ax.scatter3D(grid_coord[0,10][grid[0,10].astype(bool)],grid_coord[1,10][grid[1,10].astype(bool)],grid_coord[2,10][grid[2,10].astype(bool)])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.grid()

fig,ax = plt.subplots(1,1,)
ax.plot(surfNodes[:int(pntsPerXsec/2)+3,0,0],surfNodes[:int(pntsPerXsec/2)+3,0,-1])
ax.plot(surfNodes[int(pntsPerXsec/2)+2:,0,0],surfNodes[int(pntsPerXsec/2)+2:,0,-1])
ax.scatter(surfNodes[:,4,0],surfNodes[:,4,-1])
# ax.scatter(suction_surfNodes[:14,0],suction_surfNodes[:14,-1])
# ax.scatter(surfNodes[:pntsPerXsec,0],surfNodes[:pntsPerXsec,-1])
ax.grid()