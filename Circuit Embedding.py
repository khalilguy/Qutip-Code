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

def get_ordered_nodelist(grid):
    """
    Parameters:
        Networkx Graph
    """
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
    return ordered_nodelist, unique_nodelist


hardware_qubits_position_dict = {} #key is hardware qubit index, value is its position
for i in range(len(unique_nodelist)):
    hardware_qubits_position_dict[i] = ordered_nodelist[i]

    
hardware_qubits_mapped_qubit_dict = {}  #key is hardware qubit index, value is a mapped qubit

new_labels = {val:key for (key, val) in hardware_qubits_position_dict.items()}
H = nx.relabel_nodes(grid_notdi, new_labels)
'''
plt.figure()
steingraph = nx.algorithms.approximation.steinertree.steiner_tree(grid_notdi, [(1,0), (2,2)])
labels = nx.draw_networkx_labels(steingraph, pos=nx.spring_layout(steingraph))
nx.draw(steingraph)
'''

#connected = nx.algorithms.connectivity.disjoint_paths.node_disjoint_paths(grid,(1,0),(2,2)) #Worse version of all_simple_paths function 
simp_paths = nx.algorithms.simple_paths.all_simple_paths(grid, (1,0), (2,2), cutoff = 4)



def ListPossibleCombinations(arr,n,r):
    possible_combs = int(np.math.factorial(n)/(np.math.factorial(r)*(np.math.factorial(n-r))))
    combos = [0]*possible_combs
    combo = [0]*r
    return FindPossibleCombinations(arr, combos, combo, 0, n -1, 0,0, r)[0]


def FindPossibleCombinations(arr,combos, combo, start,end, combo_index,combos_index,r):
    if (combo_index == r):
        this_combo = list(combo)
        combos[combos_index] = this_combo
        combos_index = combos_index + 1
        return combos, combos_index; 
        
    i = start;  
    while(i <= end and end - i + 1 >= r - combo_index): 
        combo[combo_index] = arr[i]; 
        combos, combos_index = FindPossibleCombinations(arr,combos, combo, i + 1,end, combo_index + 1,combos_index,r);
        i += 1;
    return combos, combos_index

ordered_pairs = ListPossibleCombinations(ordered_nodelist, len(ordered_nodelist), 2);
indexed_pairs = ListPossibleCombinations(indexed_nodelist, len(ordered_nodelist), 2);

def FindPathsofLengthN(grid,start, end, N):
    paths_length_N = []
    paths = nx.algorithms.simple_paths.all_simple_paths(grid, start, end, cutoff = N)
    for i in range(len(paths)):
         if len(paths[i]) == N:
             paths_length_N.append(paths[i])
             
print(H.edges())

               
noise_dict = {list(H.edges())[i]:np.random.rand() for i in range(len(H.edges()))}

print()
q1 = QubitCircuit(9)
q1.add_gate("SNOT",0)
q1.add_gate("SWAP",[0,1])
q1.add_gate("SWAP",[1,2])
q1.add_gate("SWAP",[2,5])

Ulist = q1.propagators()

print(Ulist[1].full == swap(9).full)
    
#print(qubit_dict)





#print(ordered_nodelist)
#print(indexed_nodelist)
#print(unique_nodelist)


