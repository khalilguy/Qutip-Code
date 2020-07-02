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
import itertools

import CircuitUtils as cu
import MyGates as mg



grid = nx.generators.lattice.grid_2d_graph(3,3 , create_using = nx.DiGraph)
grid_notdi = nx.generators.lattice.grid_2d_graph(3,3)
#nx.draw_networkx_labels(grid, pos=nx.spring_layout(grid))
#nx.draw(grid)


ordered_nodelist, indexed_nodelist = cu.get_ordered_nodelist(grid)
hardware_qubits_position_dict = {} #key is hardware qubit index, value is its position


for i in range(len(ordered_nodelist)):
    hardware_qubits_position_dict[i] = ordered_nodelist[i]


hardware_qubits_mapped_qubit_dict = {i:i for i in range(len(ordered_nodelist))}  #key is hardware qubit index, value is a mapped qubit


new_labels = {val:key for (key, val) in hardware_qubits_position_dict.items()}

H = nx.relabel_nodes(grid, new_labels)
G = nx.relabel_nodes(grid_notdi, new_labels)
nx.draw_networkx_labels(H, pos=nx.spring_layout(H))
nx.draw(H)
#print(G.edges())

ordered_pairs = cu.ListPossibleCombinations(ordered_nodelist, len(ordered_nodelist), 2);
indexed_pairs = cu.ListPossibleCombinations(indexed_nodelist, len(ordered_nodelist), 2);

#print(H.edges())

               
noise_dict_position_keys = {list(grid.edges())[i]:np.random.rand() for i in range(len(grid.edges()))}
for i in range(len(list(grid_notdi.edges))):
    noise_dict_position_keys[(list(grid_notdi.edges())[i][1], list(grid_notdi.edges())[i][0])] = noise_dict_position_keys[(list(grid_notdi.edges())[i][0], list(grid_notdi.edges())[i][1])]
        
noise_dict_index_keys = {list(H.edges())[i]:np.random.rand() for i in range(len(H.edges()))}
for i in range(len(list(G.edges))):
    noise_dict_index_keys[(list(G.edges())[i][1], list(G.edges())[i][0])] = noise_dict_index_keys[(list(G.edges())[i][0], list(G.edges())[i][1])]
          

print(noise_dict_index_keys)
q1 = QubitCircuit(4)
q1.user_gates = {"NU": mg.user_cnot, "CSWAP":mg.cnot_swap, "USWAP": mg.user_swap}
q1.add_gate("SNOT",0)
q1.add_gate("USWAP", targets = [0,1], arg_value = 1)
q1.add_gate("USWAP", targets = [1,2], arg_value = 1)
q1.add_gate("USWAP", targets = [2,3], arg_value = 1)

Ulist = q1.propagators()

#print(ordered_pairs)

        
#paths = list(nx.algorithms.simple_paths.all_simple_paths(grid, (0,0), (1,2), cutoff = 4))
#print(paths)


#print(list(grid.edges()))
y = gate_sequence_product(q1.propagators())*tensor(basis(2,0),basis(2,0),basis(2,0),basis(2,0))
previous_fidelity  = 0
#print(indexed_pairs)



#y = gate_sequence_product(q1.propagators())*tensor(basis(2,0),basis(2,0),basis(2,0),basis(2,0))
#y2 = gate_sequence_product(q2.propagators())*tensor(basis(2,0),basis(2,0),basis(2,0),basis(2,0))


triples = cu.ListPossibleCombinations(indexed_nodelist, len(ordered_nodelist), 2);
#print(triples)
pers = list(itertools.permutations(triples[0]))
#print(permutes)
#ps = [list(p) for p in map(nx.utils.pairwise, permutes)]
#print(ps[0][0])

def get_best_circuit(grid,permute, qc,best_path, highest_fidelity):
    #permutes = list(itertools.permutations(working_qubits))[0]
    
    pairs_of_qubits = nx.utils.pairwise(permute)
    paths = []
    pairs_of_qubits = list(pairs_of_qubits)
    #print(pairs_of_qubits)
    for i, p in enumerate(pairs_of_qubits):
        paths.append(list(nx.algorithms.simple_paths.all_simple_paths(H,pairs_of_qubits[i][0], pairs_of_qubits[i][1])))
    #paths.append(list(nx.algorithms.simple_paths.all_simple_paths(grid,pairs_of_qubits[0], pairs_of_qubits[1])))
    branch_size_list = [0]*len(paths)
    num_nodes = H.number_of_nodes()
    return create_circuit(num_nodes,paths, 0,[], branch_size_list,qc,best_path,highest_fidelity)

def get_bell_state(position_grid,circuit_width):
    coordinate_hardware_qubits, indexed_hardware_qubits = cu.get_ordered_nodelist(position_grid)
    hardware_qubits_position_dict = {i:ordered_nodelist[i] for i in range(len(coordinate_hardware_qubits))} #key is hardware qubit index, value is its position


    for i in range(len(ordered_nodelist)):
        hardware_qubits_position_dict[i] = ordered_nodelist[i]
    new_labels = {val:key for (key, val) in hardware_qubits_position_dict.items()}

    H = nx.relabel_nodes(grid, new_labels)
    combination_of_hardware_qubits = cu.ListPossibleCombinations(indexed_hardware_qubits, len(indexed_hardware_qubits), circuit_width);
    qc = 0
    best_path = 0
    highest_fidelity = 0 
    for i, working_qubits in enumerate(combination_of_hardware_qubits):
        permutes = list(itertools.permutations(working_qubits))
        for permute in permutes:
            qc, best_path, highest_fidelity = get_best_circuit(H,permute, qc, best_path, highest_fidelity)
    return qc, best_path, highest_fidelity

    
def create_circuit(num_nodes,paths, current_path_index,paths_list,bran_size_list,qc,best_path,highest_fidelity):
    for i, p in enumerate(map(nx.utils.pairwise, paths[current_path_index])):
        #print(list(p))
        cp = list(p)
        #print(current_paths_list)
        bran_size_list[current_path_index] = len(cp)
        current_paths_list = paths_list + cp
        #print(current_paths_list)
        #print(current_path_index)
        if (current_path_index + 1) == len(paths):
            qubits = set()
            for k in range(len(current_paths_list)):
                qubits.add(current_paths_list[k][0])
                qubits.add(current_paths_list[k][1])
                if len(qubits) == num_nodes:
                    break
            num_qubits = len(qubits)
            mapping = {list(qubits)[m]:m for m in range(num_qubits)}
            hardware_qubits_mapped_qubit_dict = {k:k for k in range(num_qubits)}
            q1 = QubitCircuit(num_qubits)
            q2 = QubitCircuit(num_qubits)
            q1.user_gates = {"NU": mg.user_cnot, "CSWAP":mg.cnot_swap, "USWAP": mg.user_swap}
            q2.user_gates = {"NU": mg.user_cnot, "CSWAP":mg.cnot_swap, "USWAP": mg.user_swap}
            q1.add_gate("SNOT",mapping[current_paths_list[0][0]])
            q2.add_gate("SNOT",mapping[current_paths_list[0][0]])
            starting_place = 0
            ending = 0
            #print(bran_size_list)
            for size in bran_size_list:
                ending = ending + size
 
                for l, edge in enumerate(current_paths_list[starting_place:ending]):
                    # if len(current_paths_list) == 1:
                    #     q1.add_gate("NU", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = 1)
                    #     q2.add_gate("NU", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = noise_dict_index_keys[edge])
                
                    if l == (num_qubits - 2):
                        q1.add_gate("NU", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = 1)
                        q2.add_gate("NU", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = noise_dict_index_keys[edge])
                    elif l < (num_qubits - 2):
                        q1.add_gate("USWAP", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = 1)
                        q2.add_gate("USWAP", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = noise_dict_index_keys[edge])
                        hardware_qubits_mapped_qubit_dict[mapping[edge[0]]], hardware_qubits_mapped_qubit_dict[mapping[edge[1]]] = hardware_qubits_mapped_qubit_dict[mapping[edge[1]]], hardware_qubits_mapped_qubit_dict[mapping[edge[0]]]
                #print(current_paths_list[starting_place:ending])    
                starting_place = size
            y = gate_sequence_product(q1.propagators())*tensor([basis(2,0)]*num_qubits)
            y2 = gate_sequence_product(q2.propagators())*tensor([basis(2,0)]*num_qubits)
            fidel = fidelity(y,y2)
            #print(fidel)
            if fidel > highest_fidelity:
                qc = q2
                best_path = current_paths_list
                highest_fidelity = fidel #highest fidelity
                print(qc.gates)
                print(mapping)
                print(best_path)
        
        
        if (i+1) == len(paths[current_path_index]) and (current_path_index + 1) == len(paths):
            return qc, best_path, highest_fidelity
        j = current_path_index
        
        if (current_path_index + 1) < len(paths):
            j += 1
            qc,best_path, highest_fidelity = create_circuit(num_nodes,paths,j, current_paths_list,bran_size_list,qc,best_path,highest_fidelity)
    return qc, best_path, highest_fidelity
qc, best_path, highest_fidelity = get_bell_state(grid,2)            

ls = []
pd = [1,2,3]
ls = ls + pd
print(highest_fidelity)


#print(ls[0:1])
#print(ls)
# first_paths = list(nx.algorithms.simple_paths.all_simple_paths(H,ps[0][0][0], ps[0][0][1]))
# second_paths = list(nx.algorithms.simple_paths.all_simple_paths(H,ps[0][1][0], ps[0][1][1]))
# print(first_paths)
# print(second_paths)
# fp = [list(p) for p in map(nx.utils.pairwise, first_paths)]
# print(fp)

#print(qc.gates)  


#print(qubit_dict)
