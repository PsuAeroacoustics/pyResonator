import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush,heappop,heappushpop
from itertools import count
import time
#%%

class sample():
    def __init__(self,size = None, N_res = None):
        self.size = size
        self.N_res = N_res
        self.occupied_nodes = deque()

class resonator():
    def __init__(self,start_node = None,length = None):
        self.start_node =start_node
        self.length = length
        self.success = False
    
    def get_path(self):
        if hasattr(self,'path'):
            path = np.flip(np.array([node.position for node in self.path]),axis = 0)
        else:
            print("This resonator hasn't been routed")
        return path

class node():
    def __init__(self,f = None,parent = None, position = None,count = None ,L = None):
        self.parent = parent
        self.position = position
        self.f = f
        self.count = count
        self.L = L

    def __lt__(self,other_node):
        return (self.f,self.count) < (other_node.f,other_node.count)

    def __gt__(self,other_node):
        return (self.f,self.count) > (other_node.f,other_node.count)

    def __eq__(self,other_node):
        return self.position == other_node.position


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
        
        print(f'Current Node: {current_node.position} - Current Length: {current_node.L}')
        if current_node.L == resonator.length:
            print('Path found!')
            resonator.path = trace_path(current_node,resonator.start_node)
            sample.occupied_nodes.extend(resonator.path)
            resonator.success = True
            break

        else:
            neighbors = current_node.position+stensile
            
            bounds = ((neighbors[:,0] <= sample.size[0]) | (neighbors[:,0] >= sample.size[1])) | ((neighbors[:,1] <= sample.size[2]) | (neighbors[:,1] >= sample.size[3])) | ((neighbors[:,2] <= sample.size[4]) | (neighbors[:,2] >= sample.size[5]))
            if np.any(bounds):
                neighbors = np.delete(neighbors,bounds,axis = 0)
            
            # if in closed set or already in open set remove from list 
            
            for i, neighbor in enumerate(neighbors):
                n = node(position = tuple(neighbor))
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
                    temp_node.L = current_node.L+1
                else:
                    temp_node.L = current_node.parent.L+2

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



res_start = [(3,3,10),(5,5,10),(4,4,10),(9,6,10)]
res_L = [100,25,50,50]

liner = sample(size = [0,10,0,10,0,10],N_res = 4)

res = {}
for i in range(liner.N_res):
    res_temp = resonator(start_node=node(position = res_start[i],L = 0),length = res_L[i])
    res = {**res,**{f'res{i}':res_temp}}
liner.resonators = res

stensile = np.array([(1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)])

[route(liner,v) for k,v in liner.resonators.items()]
# parent_set = liner.resonators['res1'].get_path()

fig = plt.figure()
ax = plt.axes(projection='3d')
for k,v in liner.resonators.items():
    if v.success:
        parent_set = v.get_path()
        ax.plot(parent_set[:,0],parent_set[:,1],parent_set[:,2],linewidth = 4)
        ax.scatter(parent_set[0,0],parent_set[0,1],parent_set[0,2])

ax.set_xlim(liner.size[0],liner.size[1])
ax.set_ylim(liner.size[2],liner.size[3])
ax.set_zlim(liner.size[4],liner.size[5])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.grid()

