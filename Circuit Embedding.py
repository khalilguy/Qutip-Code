# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:21:39 2020

@author: Khalil
"""

from qutip import *
print(qutip.__version__)
from IPython.display import Image
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms import approximation


# %matplotlib inline 

from numpy import pi

import copy
from qutip.qip.operations import *
from qutip.qip.circuit import * 
import numpy as np
from qutip.qip.device import Processor
from qutip.qip.device import CircularSpinChain, LinearSpinChain
from qutip.qip.noise import RandomNoise
from qutip.operators import sigmaz, sigmay, sigmax, destroy
from qutip.states import basis
from qutip.metrics import fidelity
from qutip.qip.operations import rx, ry, rz, hadamard_transform

grid = nx.generators.lattice.grid_2d_graph(3,3 , create_using = nx.DiGraph)
grid_notdi = nx.generators.lattice.grid_2d_graph(3,3)
labels = nx.draw_networkx_labels(grid, pos=nx.spring_layout(grid))


nx.draw(grid)
edgelist = list(grid.edges())
#print(edgelist)
unique_nodelist = []

for i in range(len(edgelist)):
    for j in range(len(edgelist[i])):
        if edgelist[i][j] not in unique_nodelist:
            unique_nodelist.append(edgelist[i][j])
            

k = 0 #pointer for value of x-coordinate
h = 0 # pointer for value of y-coordinate 
i = 0 #pointer for position in index_edgelist

ordered_nodelist = []

while i != len(unique_nodelist):
    for l in range(len(unique_nodelist)):
        if k == 3:
            k = 0
            h = h + 1
        if unique_nodelist[l][0] == h:
            if unique_nodelist[l][1] == k:
                ordered_nodelist.append(unique_nodelist[l])
                i = i + 1
                k = k + 1
                break

indexed_nodelist = list(range(len(unique_nodelist)))
q1 = QubitCircuit(4)
qubit_dict = {} #key is hardware qubit, value is its position and also a mapped logical qubit if applicable
for i in range(len(unique_nodelist)):
    qubit_dict[i] = []
    qubit_dict[i].append(ordered_nodelist[i])
    

plt.figure()
steingraph = nx.algorithms.approximation.steinertree.steiner_tree(grid_notdi, [(1,0), (2,2)])
labels = nx.draw_networkx_labels(steingraph, pos=nx.spring_layout(steingraph))
nx.draw(steingraph)

connected = nx.algorithms.approximation.connectivity.all_pairs_node_connectivity(grid)
simp_paths = nx.algorithms.simple_paths.all_simple_paths(grid, (1,0), (2,2))
print(list(simp_paths))


#print(qubit_dict)





#print(ordered_nodelist)
#print(indexed_nodelist)
#print(unique_nodelist)


