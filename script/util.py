from pymoo.indicators.hv import Hypervolume as HV
import json
import ast
import numpy as np
import copy
import numpy as np
from pymoo.problems import get_problem
from pymoo.visualization.scatter import Scatter
import os
import random
import re
import time
import numpy as np
import errno
import os
import math
import plotly.graph_objs as go
import plotly.express as px
from matplotlib import pyplot as plt
INF = float("Inf")


#    Copyright (C) 2010 Simon Wessing
#    TU Dortmund University
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import matplotlib.pyplot as plt

def is_dominated(points, candidate):
    """Check if candidate is dominated by any point in points."""
    return any(all(candidate <= point) and any(candidate < point) for point in points)

def get_pareto_front(points):
    """Extract the Pareto front, where no point is dominated by any other."""
    pareto_front = []
    for point in points:
        if not is_dominated(pareto_front, point):
            # Remove points from pareto_front if they are dominated by the new point
            pareto_front = [p for p in pareto_front if not is_dominated([point], p)]
            pareto_front.append(point)
    return np.array(pareto_front)

def plot_PF(file_path):
    # Path to the file containing data
    file_path = '/mnt/data/last_pf_1.txt'

    # Read data from the file
    with open(file_path, 'r') as file:
        data = eval(file.read())

    # Convert data values to numpy array for processing
    points_array = np.array(list(data.values()))

    # Get Pareto front
    pareto_front_points = get_pareto_front(points_array)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.scatter(points_array[:, 0], points_array[:, 1], label='All Points', alpha=0.5)
    plt.scatter(pareto_front_points[:, 0], pareto_front_points[:, 1], color='red', label='Pareto Front Points')
    plt.title('Pareto Front Points among All Points')
    plt.xlabel('Objective 1')
    plt.ylabel('Objective 2')
    plt.legend()
    plt.grid(True)
    plt.show()


from copy import deepcopy
class Object(object):
    """
        Generic object
    """
    pass

class InnerHyperVolume:
    """
    Hypervolume computation based on variant 3 of the algorithm in the paper:
    C. M. Fonseca, L. Paquete, and M. Lopez-Ibanez. An improved dimension-sweep
    algorithm for the hypervolume indicator. In IEEE Congress on Evolutionary
    Computation, pages 1157-1163, Vancouver, Canada, July 2006.

    Minimization is implicitly assumed here!

    """

    def __init__(self, referencePoint):
        """Constructor."""
        self.referencePoint = referencePoint
        self.list = []


    def compute(self, front):
        """Returns the hypervolume that is dominated by a non-dominated front.

        Before the HV computation, front and reference point are translated, so
        that the reference point is [0, ..., 0].

        """
        front = deepcopy(front)

        def weaklyDominates(point, other):
            for i in range(len(point)):
                if point[i] > other[i]:
                    return False
            return True

        relevantPoints = []
        referencePoint = self.referencePoint
        dimensions = len(referencePoint)
        for point in front:
            for i in range(len(point)):
                point[i] = -point[i]
            # only consider points that dominate the reference point
            if weaklyDominates(point, referencePoint):
                relevantPoints.append(point)
        if any(referencePoint):
            # shift points so that referencePoint == [0, ..., 0]
            # this way the reference point doesn't have to be explicitly used
            # in the HV computation
            for j in range(len(relevantPoints)):
                relevantPoints[j] = [relevantPoints[j][i] - referencePoint[i] for i in range(dimensions)]
        self.preProcess(relevantPoints)
        bounds = [-1.0e308] * dimensions
        hyperVolume = self.hvRecursive(dimensions - 1, len(relevantPoints), bounds)
        return round(hyperVolume, 4)


    def hvRecursive(self, dimIndex, length, bounds):
        """Recursive call to hypervolume calculation.

        In contrast to the paper, the code assumes that the reference point
        is [0, ..., 0]. This allows the avoidance of a few operations.

        """
        hvol = 0.0
        sentinel = self.list.sentinel
        if length == 0:
            return hvol
        elif dimIndex == 0:
            # special case: only one dimension
            # why using hypervolume at all?
            return -sentinel.next[0].cargo[0]
        elif dimIndex == 1:
            # special case: two dimensions, end recursion
            q = sentinel.next[1]
            h = q.cargo[0]
            p = q.next[1]
            while p is not sentinel:
                pCargo = p.cargo
                hvol += h * (q.cargo[1] - pCargo[1])
                if pCargo[0] < h:
                    h = pCargo[0]
                q = p
                p = q.next[1]
            hvol += h * q.cargo[1]
            return hvol
        else:
            remove = self.list.remove
            reinsert = self.list.reinsert
            hvRecursive = self.hvRecursive
            p = sentinel
            q = p.prev[dimIndex]
            while q.cargo is not None:
                if q.ignore < dimIndex:
                    q.ignore = 0
                q = q.prev[dimIndex]
            q = p.prev[dimIndex]
            while length > 1 and (q.cargo[dimIndex] > bounds[dimIndex] or q.prev[dimIndex].cargo[dimIndex] >= bounds[dimIndex]):
                p = q
                remove(p, dimIndex, bounds)
                q = p.prev[dimIndex]
                length -= 1
            qArea = q.area
            qCargo = q.cargo
            qPrevDimIndex = q.prev[dimIndex]
            if length > 1:
                hvol = qPrevDimIndex.volume[dimIndex] + qPrevDimIndex.area[dimIndex] * (qCargo[dimIndex] - qPrevDimIndex.cargo[dimIndex])
            else:
                qArea[0] = 1
                qArea[1:dimIndex+1] = [qArea[i] * -qCargo[i] for i in range(dimIndex)]
            q.volume[dimIndex] = hvol
            if q.ignore >= dimIndex:
                qArea[dimIndex] = qPrevDimIndex.area[dimIndex]
            else:
                qArea[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                if qArea[dimIndex] <= qPrevDimIndex.area[dimIndex]:
                    q.ignore = dimIndex
            while p is not sentinel:
                pCargoDimIndex = p.cargo[dimIndex]
                hvol += q.area[dimIndex] * (pCargoDimIndex - q.cargo[dimIndex])
                bounds[dimIndex] = pCargoDimIndex
                reinsert(p, dimIndex, bounds)
                length += 1
                q = p
                p = p.next[dimIndex]
                q.volume[dimIndex] = hvol
                if q.ignore >= dimIndex:
                    q.area[dimIndex] = q.prev[dimIndex].area[dimIndex]
                else:
                    q.area[dimIndex] = hvRecursive(dimIndex - 1, length, bounds)
                    if q.area[dimIndex] <= q.prev[dimIndex].area[dimIndex]:
                        q.ignore = dimIndex
            hvol -= q.area[dimIndex] * q.cargo[dimIndex]
            return hvol


    def preProcess(self, front):
        """Sets up the list Data structure needed for calculation."""
        dimensions = len(self.referencePoint)
        nodeList = MultiList(dimensions)
        nodes = [MultiList.Node(dimensions, point) for point in front]
        for i in range(dimensions):
            self.sortByDimension(nodes, i)
            nodeList.extend(nodes, i)
        self.list = nodeList


    def sortByDimension(self, nodes, i):
        """Sorts the list of nodes by the i-th value of the contained points."""
        # build a list of tuples of (point[i], node)
        decorated = [(node.cargo[i], node) for node in nodes]
        # sort by this value
        # decorated.sort()
        decorated = sorted(decorated, key=lambda x: x[0])
        # write back to original list
        nodes[:] = [node for (_, node) in decorated]
            
            
            
class MultiList: 
    """A special Data structure needed by FonsecaHyperVolume.
    
    It consists of several doubly linked lists that share common nodes. So, 
    every node has multiple predecessors and successors, one in every list.

    """

    class Node: 

        def __init__(self, numberLists, cargo=None): 
            self.cargo = cargo 
            self.next  = [None] * numberLists
            self.prev = [None] * numberLists
            self.ignore = 0
            self.area = [0.0] * numberLists
            self.volume = [0.0] * numberLists
    
        def __str__(self): 
            return str(self.cargo)
        
        
    def __init__(self, numberLists):  
        """Constructor. 
        
        Builds 'numberLists' doubly linked lists.
        Vivek: numberLists means number of dimensions

        """
        self.numberLists = numberLists
        self.sentinel = MultiList.Node(numberLists)
        self.sentinel.next = [self.sentinel] * numberLists
        self.sentinel.prev = [self.sentinel] * numberLists  
        
        
    def __str__(self):
        strings = []
        for i in range(self.numberLists):
            currentList = []
            node = self.sentinel.next[i]
            while node != self.sentinel:
                currentList.append(str(node))
                node = node.next[i]
            strings.append(str(currentList))
        stringRepr = ""
        for string in strings:
            stringRepr += string + "\n"
        return stringRepr
    
    
    def __len__(self):
        """Returns the number of lists that are included in this MultiList."""
        return self.numberLists
    
    
    def getLength(self, i):
        """Returns the length of the i-th list."""
        length = 0
        sentinel = self.sentinel
        node = sentinel.next[i]
        while node != sentinel:
            length += 1
            node = node.next[i]
        return length
            
            
    def append(self, node, index):
        """Appends a node to the end of the list at the given index."""
        lastButOne = self.sentinel.prev[index]
        node.next[index] = self.sentinel
        node.prev[index] = lastButOne
        # set the last element as the new one
        self.sentinel.prev[index] = node
        lastButOne.next[index] = node
        
        
    def extend(self, nodes, index):
        """Extends the list at the given index with the nodes."""
        sentinel = self.sentinel
        for node in nodes:
            lastButOne = sentinel.prev[index]
            node.next[index] = sentinel
            node.prev[index] = lastButOne
            # set the last element as the new one
            sentinel.prev[index] = node
            lastButOne.next[index] = node
        
        
    def remove(self, node, index, bounds): 
        """Removes and returns 'node' from all lists in [0, 'index'[."""
        for i in range(index): 
            predecessor = node.prev[i]
            successor = node.next[i]
            predecessor.next[i] = successor
            successor.prev[i] = predecessor  
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]
        return node

    def reinsert(self, node, index, bounds):
        """
        Inserts 'node' at the position it had in all lists in [0, 'index'[
        before it was removed. This method assumes that the next and previous 
        nodes of the node that is reinserted are in the list.

        """
        for i in range(index): 
            node.prev[i].next[i] = node
            node.next[i].prev[i] = node
            if bounds[i] > node.cargo[i]:
                bounds[i] = node.cargo[i]


# The pareto front of a scaled zdt1 problem
# A = get_problem("zdt1").pareto_front()
# ref_point = np.array([1.2,1.2])

def adjust_b_to_ten_times_less_than_a(a, b):
    # Calculate the scale of a and b
    scale_a = math.floor(math.log10(abs(a))) if a != 0 else 0
    scale_b = math.floor(math.log10(abs(b))) if b != 0 else 0

    # Calculate the desired scale for b
    desired_scale_b = scale_a - 1

    # Adjust b to the desired scale
    return b * 10**(desired_scale_b - scale_b)

# compute the sparsity of a given pareto front
def compute_sparsity(ep_objs_batch):
    if len(ep_objs_batch) < 2:
        return 0.0

    sparsity = 0.0
    m = len(ep_objs_batch[0])
    ep_objs_batch_np = np.array(ep_objs_batch)
    for dim in range(m):
        objs_i = np.sort(copy.deepcopy(ep_objs_batch_np.T[dim]))
        for i in range(1, len(objs_i)):
            sparsity += np.square(objs_i[i] - objs_i[i - 1])
    sparsity /= (len(ep_objs_batch) - 1)
    
    return sparsity


def calc_distance_3d(p1,p2):
    d = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2)
    return d

def calc_distance_2d(p1,p2):
    d = math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)
    return d

# compute the hypervolume of a given pareto front
def compute_hypervolume(ep_objs_batch):
    n = len(ep_objs_batch[0])
    HV = InnerHyperVolume(np.zeros(n))
    return HV.compute(ep_objs_batch)

def compute_hv(pf):
    ref_point = np.array([1.5,1.5,0])
    # ref_point = np.array([0,0,0])

    hv= HV(ref_point=ref_point)
    return hv(pf)

def crowd_dist(sorted_datas):
    """Given a list of vectors, this method computes the crowding distance of each vector, i.e. the sum of distances between neighbors for each dimension

    Arguments:
        datas {list} -- list of vectors

    Returns:
        list -- list of crowding distances
    """
    res = {}
    # sorted_datas = sorted(datas,key = lambda x:(x[0],x[1],x[2]))
    for i,p in enumerate(sorted_datas) :
        if i == 0 or i == len(sorted_datas) - 1:
            res[i] = INF
        else:
            if len(sorted_datas[i])==3:
                d1 = calc_distance_3d(sorted_datas[i],sorted_datas[i-1])
                d2 = calc_distance_3d(sorted_datas[i],sorted_datas[i+1])
            elif len(sorted_datas[i])==2:
                d1 = calc_distance_2d(sorted_datas[i],sorted_datas[i-1])
                d2 = calc_distance_2d(sorted_datas[i],sorted_datas[i+1])

            res[i] = float((d1+d2)/2)


    # points = np.array([Object() for _ in datas])
    # dimensions = len(datas[0])
    # for i, d in enumerate(datas):
    #     points[i].data = d
    #     points[i].i = i
    #     points[i].distance = 0.

    # # Compute the distance between neighbors for each dimension and add it to
    # # each point's global distance
    # for d in range(dimensions):
    #     points = sorted(points, key=lambda p: p.data[d])
    #     spread = points[-1].data[d] - points[0].data[d]
    #     for i, p in enumerate(points):
    #         if i == 0 or i == len(points) - 1:
    #             p.distance += INF
    #         else:
    #             p.distance += (
    #                 points[i + 1].data[d] - points[i - 1].data[d]) / spread

    # # Sort points back to their original order
    # points = sorted(points, key=lambda p: p.i)
    # distances = np.array([p.distance for p in points])

    return list(res.values())

def dominate_for_max_optimize(p,q):
    res = 0
    dim = len(p)
    for o_p,o_q in zip(p,q):
        if o_p>=o_q:
            res+=1
        else:
            res+=-1
    if res==dim:
        return 1
    if res==-dim:
        return -1
    else:
        return 0


def get_pareto_front_over_all_policy(S):
    res = []
    for ind,pi in enumerate(S):
        collection = list(pi.values())
        collection.extend(res)
        for i, p in enumerate(collection):
            flag = 0
            for j, q in enumerate(collection):
                if p==q:
                    continue
                else:
                    if dominate_for_max_optimize(p,q) == -1:
                        if p in res:
                            res.remove(p)
                        flag =1
                        break
            if flag == 0:
                if not p in res:
                    res.append(p)
    return list(res)
            
if __name__ == '__main__':
    env_name_list = ['Hopper-v4','HalfCheetah-v4','Walker2d-v4','Swimmer-v4']

    fig = plt.figure()

    
    # env_name = 'HalfCheetah-v4'
    # dim = 2
    # labels = ['DWCPPO2','DWCPPO1']
    # roots = [
    #         f'D:/work/PPO_MORL/PPO_max_0323/DWCPPO-discrete/results/{env_name}/dwc2',
    #         f'D:/work/PPO_MORL/PPO_max_0323/DWCPPO-discrete/results/{env_name}/dwc1']
    

    env_name = 'Hopper-v4'
    dim = 3
    labels = ['LUCB','PGDWCPPO1','DWCPPO2','DWCPPO1']
    roots = [f'D:/work/PPO_MORL/PPO_max_0323/PG_DWCPPO-discrete/results/{env_name}/LUCB',
            f'D:/work/PPO_MORL/PPO_max_0323/PG_DWCPPO-discrete/results/{env_name}/pgdwc3',
            f'D:/work/PPO_MORL/PPO_max_0323/DWCPPO-discrete/results/{env_name}/dwc2',
            f'D:/work/PPO_MORL/PPO_max_0323/DWCPPO-discrete/results/{env_name}/dwc1']



    
    # env_name = 'Swimmer-v4'
    # dim = 2      
    # labels = ['LUCB','PGDWCPPO1','DWCPPO2','DWCPPO1']
    # roots = [f'D:/work/PPO_MORL/PPO_max_0323/PG_DWCPPO-discrete/results/{env_name}/LUCB',
    #          f'D:/work/PPO_MORL/PPO_max_0323/PG_DWCPPO-discrete/results/{env_name}/pgdwc3',
    #          f'D:/work/PPO_MORL/PPO_max_0323/DWCPPO-discrete/results/{env_name}/dwc2',
    #         f'D:/work/PPO_MORL/PPO_max_0323/DWCPPO-discrete/results/{env_name}/dwc1']    

    
    markers = ['o','s','^','D','P','X']
    colors = [[255, 0, 0],[0, 120, 0],[0, 0, 255],[0, 0, 0],[255, 0, 255],
                 [12, 232, 228],[8, 134, 212],[86, 9, 219],[176, 18, 219]]
    

    
    # import pandas as pd
    # ax = fig.add_subplot()
    # ax.set_xlabel("Training steps")
    # ax.set_ylabel("Hypervolume")
    # for ind,root in enumerate(roots):
    #     l = []
    #     with open(root+f'/hv.txt', 'r') as file:
    #         hv = file.read()   
    #         hv = hv.split(',')
    #         for i in hv:
    #             if i != '':
    #                 l.append(float(i))
    #         l = l[:200]
    #         indices = np.arange(len(l))
    #         desired_indices= np.linspace(0,len(l),10000)
    #         l = np.interp(desired_indices,indices,l)
    #         std = l.std(axis= 0)

    #         pdl = pd.Series(l)
    #         sml = pdl.rolling(1000).mean()
       

    #         plt.plot([i for i in range(len(l))],sml,color = np.array(colors[ind])/255)
    #         plt.fill_between([i for i in range(len(l))],sml-std,sml+std,color=np.array(colors[ind])/255,alpha=0.05,label='Std')

    # ax.legend(loc='lower right',fontsize = 10)
    # plt.savefig(f'./results/{env_name}/hv.png',dpi=800)

    # plt.cla()
    # a =1

    if dim==3:
        ax = fig.add_subplot(111,projection='3d')
    if dim==2:
        ax = fig.add_subplot()

    for ind,root in enumerate(roots):
        if dim==3:
            ax.set_xlabel("Average velocity")
            ax.set_ylabel("Average height")
            ax.set_zlabel("Average energy efficiency")
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)
            ax.tick_params(axis='z', labelsize=8)
        if dim==2:
            ax.set_xlabel("Average velocity")
            ax.set_ylabel("Average energy efficiency")
            ax.tick_params(axis='x', labelsize=8)
            ax.tick_params(axis='y', labelsize=8)


  

        pf_list = []
        color = np.array(colors[ind])/255
    
        with open(root+'/seeds.txt', 'r') as file:
            str_seeds = file.read()
        seeds = ast.literal_eval(str_seeds)

        # seeds = [1,2,9,18,25,39,48,60]



        for s in seeds:
            with open(root+f'/pf_{s}.txt', 'r') as file:
                pf = file.read()      
            pf_ =ast.literal_eval(pf)   
            if root == f'D:/work/PPO_MORL/PPO_max_0323/PG_DWCPPO-discrete/results/Swimmer-v4/LUCB':
                for e in list(pf_.values()):
                    e[1]+=1
        
            pf_list.append(pf_ )

        res = get_pareto_front_over_all_policy(pf_list)
        with open(root+'/clean_pf.txt', 'w') as file:
            file.write(str(res))

        # res = []
        # for i in range(len(pf_list)):
        #     res.extend(list(pf_list[i].values()))


        hv1 = compute_hypervolume(res)
        s1  =compute_sparsity(res)
        hvs1 = hv1-s1
        name = root.split('/')[-1]
        if name == '':
            name = 'GPUCB'
        print(f'{name}:HV:{hv1},S:{s1},HVS:{hvs1}')
        
        # for p in res:
        if len(res[0])==3:
            ax.scatter([p[0] for p in res], [p[1] for p in res],[p[2] for p in res],c=[color],marker=markers[ind],s=4,label=labels[ind])
            
            # ax.view_init() 
            # plt.savefig(f'./results/{env_name}/{env_name}_{name}_PF_clean.png',dpi=800)
            # ax.view_init(elev = 0,azim = 0) 
            # plt.savefig(f'./results/{env_name}/{env_name}_{name}_PF_XZ_clean.png',dpi=800)
            # # YZ
            # ax.view_init(elev = 0,azim = 90) 
            # plt.savefig(f'./results/{env_name}/{env_name}_{name}_PF_YZ_clean.png',dpi=800)
            # # XY
            # ax.view_init(elev = 90,azim = -90) 
            # plt.savefig(f'./results/{env_name}/{env_name}_{name}_PF_XY_clean.png',dpi=800) 
            # plt.cla()

        if len(res[0])==2:
            ax.scatter([p[0] for p in res], [p[1] for p in res],c=[color],marker=markers[ind],s=4,label=labels[ind])

            # plt.savefig(f'./results/{env_name}/{env_name}_{name}_PF_clean.png',dpi=800)
            # plt.cla()
        
        

    if dim ==3:
        ax.legend(loc='lower left',fontsize = 4)
        ax.view_init() 
        plt.savefig(f'./results/{env_name}/{env_name}_PF_clean.png',dpi=800)
        ax.view_init(elev = 0,azim = 0) 
        plt.savefig(f'./results/{env_name}/{env_name}_PF_XZ_clean.png',dpi=800)
        # YZ
        ax.view_init(elev = 0,azim = 90) 
        plt.savefig(f'./results/{env_name}/{env_name}_PF_YZ_clean.png',dpi=800)
        # XY
        ax.view_init(elev = 90,azim = -90) 
        plt.savefig(f'./results/{env_name}/{env_name}_PF_XY_clean.png',dpi=800) 
        plt.cla()
    
    if dim ==2:
        ax.legend(loc='lower left',fontsize = 4)
        plt.savefig(f'./results/{env_name}/{env_name}_PF_clean.png',dpi=800)
        plt.cla()


    
