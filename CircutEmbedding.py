# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 22:36:44 2020

@author: Khalil
"""

from qutip import *
import numpy as np
from IPython.display import Image
import networkx as nx
import matplotlib.pyplot as plt
# %matplotlib inline 



import copy
from qutip.qip.operations import *
from qutip.qip.circuit import * 
from qutip.qip.device import Processor
from qutip.qip.device import CircularSpinChain, LinearSpinChain
from qutip.qip.noise import RandomNoise
from qutip.operators import sigmaz, sigmay, sigmax, destroy
from qutip.states import basis
from qutip.metrics import fidelity
from qutip.qip.operations import rx, ry, rz, hadamard_transform

import CircuitUtils as cu
import MyGates as mg



grid = nx.generators.lattice.grid_2d_graph(3,3 , create_using = nx.DiGraph)
grid_notdi = nx.generators.lattice.grid_2d_graph(3,3)
#nx.draw_networkx_labels(grid, pos=nx.spring_layout(grid))
#nx.draw(grid)


ordered_nodelist, unique_nodelist, indexed_nodelist = cu.get_ordered_nodelist(grid)
hardware_qubits_position_dict = {} #key is hardware qubit index, value is its position


for i in range(len(unique_nodelist)):
    hardware_qubits_position_dict[i] = ordered_nodelist[i]


hardware_qubits_mapped_qubit_dict = {i:i for i in range(len(ordered_nodelist))}  #key is hardware qubit index, value is a mapped qubit


new_labels = {val:key for (key, val) in hardware_qubits_position_dict.items()}

H = nx.relabel_nodes(grid, new_labels)
G = nx.relabel_nodes(grid_notdi, new_labels)
nx.draw_networkx_labels(H, pos=nx.spring_layout(H))
nx.draw(H)
print(G.edges())

ordered_pairs = cu.ListPossibleCombinations(ordered_nodelist, len(ordered_nodelist), 2);
indexed_pairs = cu.ListPossibleCombinations(indexed_nodelist, len(ordered_nodelist), 2);

#print(H.edges())

               
noise_dict_position_keys = {list(grid.edges())[i]:np.random.rand() for i in range(len(grid.edges()))}
for i in range(len(list(grid_notdi.edges))):
    noise_dict_position_keys[(list(grid_notdi.edges())[i][1], list(grid_notdi.edges())[i][0])] = noise_dict_position_keys[(list(grid_notdi.edges())[i][0], list(grid_notdi.edges())[i][1])]
        
noise_dict_index_keys = {list(H.edges())[i]:np.random.rand() for i in range(len(H.edges()))}
for i in range(len(list(G.edges))):
    noise_dict_index_keys[(list(G.edges())[i][1], list(G.edges())[i][0])] = noise_dict_index_keys[(list(G.edges())[i][0], list(G.edges())[i][1])]
          

#print(noise_dict_index_keys)
print(indexed_pairs)

q1 = QubitCircuit(4)
q1.user_gates = {"NU": mg.user_cnot, "CSWAP":mg.cnot_swap, "USWAP": mg.user_swap}
q1.add_gate("SNOT",0)
q1.add_gate("USWAP", targets = [0,1], arg_value = 1)
q1.add_gate("USWAP", targets = [1,2], arg_value = 1)
q1.add_gate("USWAP", targets = [2,3], arg_value = 1)

Ulist = q1.propagators()
print(q1.gates[1].name)
#print(ordered_pairs)

       
#paths = list(nx.algorithms.simple_paths.all_simple_paths(grid, (0,0), (1,2), cutoff = 4))
#print(paths)

length_n_list = cu.FindPathsofLengthN(grid,ordered_pairs[2][0],ordered_pairs[2][1],4)
length_n_list = cu.FindPathsofLengthN(H,indexed_pairs[6][0],indexed_pairs[6][1],4)

print(length_n_list)
#print(list(grid.edges()))
previous_fidelity  = 0
print(indexed_pairs)

og_gates_list = q1.gates
transpiled_gates_list_with_noise = []
set_of_hardware_qubits = set()
for gate in og_gates_list:
    if gate.name == "SNOT":
        transpiled_gates_list.append(gate)
    else:

        for pair in indexed_pairs:
            paths = list(nx.algorithms.simple_paths.all_simple_paths(H,pair[0], pair[1]))
            #length_n_list = cu.FindPathsofLengthN(H,indexed_pairs[u][0],indexed_pairs[u][0],4)
            hardware_qubits_mapped_qubit_dict = {i:i for i in range(len(ordered_nodelist))}  #key is hardware qubit index, value is a mapped qubit
            j = 0
            for path in map(nx.utils.pairwise, paths):
                pairwise_edges = list(path)
                num_qubits = len(paths[j])
                i = 0 
                for edge in pairwise_edges:
                    if len(pairwise_edges) == 1:
                        q1.add_gate("NU", targets = [i,i+1], arg_value = 1)
                        q2.add_gate("NU", targets = [i,i+1], arg_value = noise_dict_index_keys[edge])
                        i = i + 1
                        
                    elif i == (num_qubits - 1):
                        q1.add_gate("NU", targets = [i-1,i], arg_value = 1)
                        q2.add_gate("NU", targets = [i-1,i], arg_value = noise_dict_index_keys[edge])
                        i = i + 1
                    elif i < (num_qubits - 2):
                        q1.add_gate("USWAP", targets = [i,i+1], arg_value = 1)
                        q2.add_gate("USWAP", targets = [i,i+1], arg_value = noise_dict_index_keys[edge])
                        hardware_qubits_mapped_qubit_dict[edge[0]], hardware_qubits_mapped_qubit_dict[edge[1]] = hardware_qubits_mapped_qubit_dict[edge[1]], hardware_qubits_mapped_qubit_dict[edge[0]]
                        i = i + 1
                y = gate_sequence_product(q1.propagators())*tensor([basis(2,0)]*num_qubits)
                y2 = gate_sequence_product(q2.propagators())*tensor([basis(2,0)]*num_qubits)
                fidel = fidelity(y,y2)
                if fidel > previous_fidelity:
                    qc = q2
                    best_path = pairwise_edges
                    previous_fidelity = fidel #highest fidelity
                    print(qc.gates)
                    print(best_path)
                    print(previous_fidelity)
                j = j + 1

#y = gate_sequence_product(q1.propagators())*tensor(basis(2,0),basis(2,0),basis(2,0),basis(2,0))
#y2 = gate_sequence_product(q2.propagators())*tensor(basis(2,0),basis(2,0),basis(2,0),basis(2,0))

#print(noise_dict)
#print(hardware_qubits_mapped_qubit_dict)

