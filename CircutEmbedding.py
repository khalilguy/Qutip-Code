# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:36:44 2020

@author: Khalil
"""

from qutip import *
from IPython.display import Image
import networkx as nx
import matplotlib.pyplot as plt
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

import CircuitUtils as cu



grid = nx.generators.lattice.grid_2d_graph(3,3 , create_using = nx.DiGraph)
grid_notdi = nx.generators.lattice.grid_2d_graph(3,3)
#nx.draw_networkx_labels(grid, pos=nx.spring_layout(grid))
#nx.draw(grid)


ordered_nodelist, unique_nodelist, indexed_nodelist = cu.get_ordered_nodelist(grid)
hardware_qubits_position_dict = {} #key is hardware qubit index, value is its position


for i in range(len(unique_nodelist)):
    hardware_qubits_position_dict[i] = ordered_nodelist[i]


hardware_qubits_mapped_qubit_dict = {}  #key is hardware qubit index, value is a mapped qubit

new_labels = {val:key for (key, val) in hardware_qubits_position_dict.items()}
H = nx.relabel_nodes(grid_notdi, new_labels)


ordered_pairs = ListPossibleCombinations(ordered_nodelist, len(ordered_nodelist), 2);
indexed_pairs = ListPossibleCombinations(indexed_nodelist, len(ordered_nodelist), 2);

print(H.edges())

               
noise_dict = {list(H.edges())[i]:np.random.rand() for i in range(len(H.edges()))}

def user_gate1(a):
    b= 1-a 
    mat = np.zeros((4,4), dtype = np.complex)
    mat[0,0] = mat[1,1] = 1
    mat[2,2] = b/np.sqrt(a**2 + b**2)
    mat[3,3] = -mat[2,2]
    mat[3,2] = mat[2,3] =  a/np.sqrt(a**2 +b**2)
    return Qobj(mat, dims=[[2, 2], [2, 2]])

q1 = QubitCircuit(9)
q1.add_gate("SNOT",0)
q1.add_gate("SWAP",[0,1])
q1.add_gate("SWAP",[1,2])
q1.add_gate("SWAP",[2,5])

Ulist = q1.propagators()

print(Ulist[1].full == swap(9).full)
    
#print(qubit_dict)
