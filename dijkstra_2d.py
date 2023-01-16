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
        return (self.f,self.remaining_dist,self.count) < (other_node.f,other_node.remaining_dist,other_node.count)

    def __gt__(self,other_node):
        return (self.f,self.remaining_dist,self.count) > (other_node.f,other_node.remaining_dist,other_node.count)

    def __eq__(self,other_node):
        return self.position == other_node.position

# X,Y = np.meshgrid(np.arange(10),np.arange(10))

# start_node = node(position = (10,1),f = 0,count = 0)
start_node = node(position = (10,3),f = 0,count = 0,L = 0)
# end_node = node(position = (7,3))
end_node = node(position = (5,6))

# grid = np.ones((10,10))*np.inf
# grid[start_node] = 0

open_set = []
closed_set = deque()
parent_set = deque()

cnt = count()
heappush(open_set,start_node)

stensile = np.array([(1,0),(0,-1),(-1,0),(0,1)])

L_min = np.sum(abs(np.diff((start_node.position,end_node.position),axis = 0)))
L = L_min+6
round((L_min+10)/round(L_min/2))

while len(open_set) > 0:

    current_node = heappop(open_set)
    print(f'{current_node.position}: {current_node.f}: {current_node.L}')
    L_remainder = np.sum(abs(np.diff((current_node.position,end_node.position),axis = 0)))
    # current_node.remaining_dist = L_remainder
    L_current = np.sum(abs(np.diff((start_node.position,current_node.position),axis = 0)))

    # if L-L_current == L_remainder:
    #     #  print(L_remainder)

    # if current_node == end_node:
    if ((current_node == end_node) & (current_node.L == L)):
        # start_node = node(position = (10,1),f = 0,count = 0)
        # print(f'{current_node.position}: {L_remainder}')
        print('Path found!')
        print(current_node.L)
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
        
        # if L > round(1.5*L_min):
        #     f = np.sum(abs(neighbors-start_node.position),axis = 1)+(L-L_current-L_remainder)
        # L_neighbors_remain = np.sum(abs(neighbors-end_node.position),axis = 1)
        # f = np.sum(abs(neighbors-start_node.position),axis = 1)+(L-L_current-L_remainder)
        remaining_dist = np.sum(abs(neighbors-end_node.position),axis = 1)
        f = np.sum(abs(neighbors-start_node.position),axis = 1)

        # if L_remainder <= L-L_min:
        #     f = np.sum(abs(neighbors-start_node.position),axis = 1)+abs(L-current_node.L)
        # else:
        #     f = np.sum(abs(neighbors-start_node.position),axis = 1)
        f = np.zeros(len(neighbors))
        if current_node != start_node:
            current_direction = np.diff((current_node.parent.position,current_node.position),axis = 0)
            bend_penelty_ind = np.sum((neighbors-current_node.position) ==  current_direction,axis = 1) != 2
            f[bend_penelty_ind] = f[bend_penelty_ind]+1

        # if ((L-L_min > round(L_min/3)) & (round(L_min/3) == L_current)):
        #     print('start here')
        #     start_node = current_node
        #     L = L-L_current
        #     L_current = 0
        #     f = np.sum(abs(neighbors-start_node.position),axis = 1)+(L-L_current-L_remainder)

        # f = np.linalg.norm(neighbors-start_node.position,axis = 1)+(L-L_current-L_remainder)
        

        for i,pos in enumerate(neighbors):
            temp_node = node(position=tuple(pos),parent=current_node,f = current_node.L+1+f[i],remaining_dist = 0, count = next(cnt))
            if current_node ==start_node:
                temp_node.L = current_node.L+1
            else:
                temp_node.L = current_node.parent.L+2

            ind = np.squeeze(np.where([temp_node == existing_node for existing_node in open_set]))
            if ind.size !=  0:
                if temp_node.f < open_set[ind].f:
                    open_set[ind].f = temp_node.f
                    # open_set[ind].remaining_dist = abs(L-L_current-L_remainder)
                    open_set[ind].parent = temp_node.parent
            else:
                heappush(open_set,temp_node)

        # print(f'{current_node.position} - {neighbors}')

    if current_node.position not in closed_set and current_node != end_node and temp_node.L < L_min:
        closed_set.append(current_node.position) 

print(parent_set)

parent_set = np.flip(np.array(parent_set),axis =  0)

fig,ax = plt.subplots(1,1, figsize = (6.4,4.5))
ax.scatter(parent_set[:,1],parent_set[:,0])
ax.scatter(start_node.position[1],start_node.position[0])
ax.scatter(end_node.position[1],end_node.position[0])
ax.set_xlim(0,15)
ax.set_ylim(0,10)
plt.grid()