import numpy as np
import matplotlib.pyplot as plt
from collections import deque
from heapq import heappush,heappop,heappushpop
from itertools import count
#%%


class node():
    def __init__(self,f = None,parent = None, position = None,count = None, remaining_dist = None ,L = None):
        self.parent = parent
        self.position = position
        self.f = f
        self.count = count
        self.remaining_dist = remaining_dist
        self.L = L

# add several L_remainder fo comparison instead of just a single huristic
    def __lt__(self,other_node):
        return (self.f,self.count) < (other_node.f,other_node.count)

    def __gt__(self,other_node):
        return (self.f,self.count) > (other_node.f,other_node.count)

    def __eq__(self,other_node):
        return self.position == other_node.position

def trace_path(current_node,start_node):
    parent_set = deque()
    parent_set.append(current_node.position)
    while current_node != start_node:
        current_node = current_node.parent
        parent_set.append(current_node.position) 
    return parent_set

def step(current_node, local_closed_set, local_open_set,start_node,end_node,cnt):

    # print(current_node.position)
    neighbors = current_node.position+stensile
    bounds = ((neighbors[:,0] >= 10) | (neighbors[:,0] <= 0)) |((neighbors[:,1] >= 10) | (neighbors[:,1] <= 0))
    if np.any(bounds):
        neighbors = np.delete(neighbors,bounds,axis = 0)
    L_init = len(neighbors)

    # if in closed set or already in open set remove from list 

    for i, neighbor in enumerate(neighbors):
        # [node(position =neighbor) == existing_node for existing_node in local_closed_set]
        # neighbor = node(position =neighbor)
        if tuple(neighbor) in local_closed_set:
            neighbors = np.delete(neighbors,i-(L_init-len(neighbors)),axis = 0)

    f = np.sum(abs(neighbors-start_node.position),axis = 1)

    # f = np.sum(abs(neighbors-start_node.position),axis = 1)+np.sum(abs(neighbors-end_node.position),axis = 1)
    # f = current_node.L+1+np.linalg.norm(neighbors-end_node.position,axis = -1)
    # print(f'{current_node.position} -- {neighbors}  -- {f}')
    if current_node != start_node:
        current_direction = np.diff((current_node.parent.position,current_node.position),axis = 0)
        bend_penelty_ind = np.sum((neighbors-current_node.position) ==  current_direction,axis = 1) != 2
        f[bend_penelty_ind] = f[bend_penelty_ind]+1

    for i,pos in enumerate(neighbors):
        temp_node = node(position=tuple(pos),parent=current_node,f = f[i],remaining_dist = 0, count = next(cnt),L = current_node.L+1)

        ind = np.squeeze(np.where([temp_node == existing_node for existing_node in local_open_set]))
        if ind.size !=  0:
            if temp_node.f < local_open_set[ind].f:
                local_open_set[ind].f = temp_node.f
                local_open_set[ind].parent = temp_node.parent
        else:
            heappush(local_open_set,temp_node)

    # if current_node.position not in local_closed_set:
    local_closed_set.append(current_node.position)



# def dikstra(start_node):
    
#     local_closed_set = deque()
#     open_set = []
#     cnt = count()
#     heappush(open_set,start_node)
   
#     while len(open_set) > 0:
       
#         current_node = heappop(open_set)
#         print(current_node.position)
#         if current_node in global_closed_set:
#             print('Path found!')
#             print(current_node.L)
#             parent_set = trace_path(current_node.parent,start_node)
#             break

#         else:
#             neighbors = current_node.position+stensile
#             bounds = ((neighbors[:,0] >= 10) | (neighbors[:,0] <= 0)) |((neighbors[:,1] >= 10) | (neighbors[:,1] <= 0))
#             if np.any(bounds):
#                 neighbors = np.delete(neighbors,bounds,axis = 0)
#             L_init = len(neighbors)

#             # if in closed set or already in open set remove from list 
#             for i, neighbor in enumerate(neighbors):
#                 if tuple(neighbor) in local_closed_set:
#                     neighbors = np.delete(neighbors,i-(L_init-len(neighbors)),axis = 0)

#             # f = np.sum(abs(neighbors-start_node.position),axis = 1)

#             for i,pos in enumerate(neighbors):
#                 temp_node = node(position=tuple(pos),parent=current_node,f = current_node.L+1,remaining_dist = 0, count = next(cnt),L = current_node.L+1)

#             ind = np.squeeze(np.where([temp_node == existing_node for existing_node in open_set]))
#             if ind.size !=  0:
#                 if temp_node.f < open_set[ind].f:
#                     open_set[ind].f = temp_node.f
#                     open_set[ind].parent = temp_node.parent
#             else:
#                 heappush(open_set,temp_node)

#             # if current_node.position not in local_closed_set:
#             local_closed_set.append(current_node.position)
#             print(len(local_closed_set))
#             global_closed_set.append(current_node.position)
             
#     return parent_set

def bidirect_search(start_node, end_node):
    
    local_closed_set_1, local_closed_set_2 = deque(),deque()
    local_open_set_1,local_open_set_2 = [],[]
    cnt_1,cnt_2= count(), count()

    heappush(local_open_set_1,start_node)
    heappush(local_open_set_2,end_node)

    while len(local_open_set_1) > 0: 
        current_node_1 = heappop(local_open_set_1)
        current_node_2 = heappop(local_open_set_2)
        print(f'{current_node_1.position}: {current_node_1.f}, {current_node_2.position}: {current_node_2.f}')

        step(current_node_1, local_closed_set_1, local_open_set_1,start_node,end_node,cnt_1)
        step(current_node_2, local_closed_set_2, local_open_set_2,end_node,start_node,cnt_2)

        if current_node_1 in global_closed_set:
            print('Path found!')
            parent_set_1 = trace_path(current_node_1,start_node)
            
            global_closed_set.reverse()
            for i,existing_node in enumerate(global_closed_set):
                if current_node_1==existing_node:
                    ind = i
                    break
            parent_set_2 = trace_path(global_closed_set[i],end_node)
            parent_set = np.concatenate((np.flip(np.array(parent_set_1)[1:],axis =  0),np.array(parent_set_2)))
            break
        
        global_closed_set.append(current_node_1)
        global_closed_set.append(current_node_2)

    return parent_set


# X,Y = np.meshgrid(np.arange(10),np.arange(10))

global_closed_set = deque()

# start_node = node(position = (10,1),f = 0,count = 0)
start_node = node(position = (10,5),f = 0,count = 0,L = 0)
# end_node = node(position = (7,3))
end_node = node(position = (3,9),f = 0,count = 0,L = 0)
L_min = np.sum(abs(np.diff((start_node.position,end_node.position),axis = 0)))
L = L_min+2
stensile = np.array([(1,0),(0,-1),(-1,0),(0,1)])

parent_set = bidirect_search(start_node, end_node)
# grid = np.ones((10,10))*np.inf
# grid[start_node] = 0

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.scatter(parent_set[:,1],parent_set[:,0])
ax.scatter(start_node.position[1],start_node.position[0])
ax.scatter(end_node.position[1],end_node.position[0])
ax.set_xlim(0,15)
ax.set_ylim(0,10)
plt.grid()