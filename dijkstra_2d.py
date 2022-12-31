import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush,heappop,heappushpop
from itertools import count
#%%


class node():
    def __init__(self,f = None,parent = None, position = None,count = None):
        self.parent = parent
        self.position = position
        self.f = f
        self.count = count

    def __lt__(self,other_node):
        return (self.f,self.count) < (other_node.f,other_node.count)

    def __gt__(self,other_node):
        return (self.f,self.count) > (other_node.f,other_node.count)

    def __eq__(self,other_node):
        return self.position == other_node.position

# X,Y = np.meshgrid(np.arange(10),np.arange(10))

start_node = node(position = (10,1),f = 0,count = 0)
end_node = node(position = (1,8))

# grid = np.ones((10,10))*np.inf
# grid[start_node] = 0

open_set = []
closed_set = deque()
parent_set = deque()

cnt = count()
heappush(open_set,start_node)

stensile = np.array([(1,0),(0,-1),(-1,0),(0,1)])

while len(open_set) > 0:

    current_node = heappop(open_set)
    if current_node == end_node:
        print('Path found!')
        while current_node != start_node:
            current_node = current_node.parent
            parent_set.append(current_node.position) 
        break

    else:

        neighbors = current_node.position+stensile
        bounds = ((neighbors[:,0] >= 10) | (neighbors[:,0] <= 0)) |((neighbors[:,1] >= 10) | (neighbors[:,1] <= 0))
        if np.any(bounds):
            neighbors = np.delete(neighbors,bounds,axis = 0)
        L_init = len(neighbors)
        # if in closed set or already in open set remove from list 
        for i, neighbor in enumerate(neighbors):
            if tuple(neighbor) in closed_set:
                neighbors = np.delete(neighbors,i-(L_init-len(neighbors)),axis = 0)

        # f = np.sum(abs(neighbors-start_node.position),axis = 1)
        f = np.linalg.norm(neighbors-start_node.position,axis = 1)
        for i,pos in enumerate(neighbors):
            temp_node = node(position=tuple(pos),parent=current_node,f = f[i], count = next(cnt))
            
            ind = np.squeeze(np.where([temp_node == existing_node for existing_node in open_set]))
            if ind.size !=  0:
                if temp_node.f < open_set[ind].f:
                    open_set[ind].f = temp_node.f
                    open_set[ind].parent = temp_node.parent
            else:
                heappush(open_set,temp_node)

        print(f'{current_node.position} - {neighbors}')

    if current_node.position not in closed_set:
        closed_set.append(current_node.position) 

print(parent_set)

parent_set = np.flip(np.array(parent_set),axis =  0)
fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.scatter(parent_set[:,1],parent_set[:,0])
ax.scatter(start_node.position[1],start_node.position[0])
ax.scatter(end_node.position[1],end_node.position[0])
ax.set_xlim(0,10)
ax.set_ylim(0,10)
plt.grid()