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


def relabel_grid(grid):
    new_labels = {node:i for i, node in enumerate(cu.get_ordered_nodelist(grid)[0])} #key is hardware qubit index, value is its position
    return nx.relabel_nodes(grid, new_labels)  #indexed directed graph
    
 
            
def get_noise_dict(indexed_directed_grid, indexed_undirected_grid):
    noise_dict_index_keys = {list(indexed_directed_grid.edges())[i]:np.random.rand() for i in range(len(indexed_directed_grid.edges()))}
    
    for node in list(indexed_undirected_grid.edges):
        noise_dict_index_keys[(node[1], node[0])] = noise_dict_index_keys[(node[0], node[1])]
    return noise_dict_index_keys

def get_bell_state(position_grid,circuit_width,noise_dict):
    
    coordinates_hardware_qubits, indexed_hardware_qubits = cu.get_ordered_nodelist(position_grid)
    H = relabel_grid(position_grid)
    combinations_of_hardware_qubits = cu.ListPossibleCombinations(indexed_hardware_qubits, len(indexed_hardware_qubits), circuit_width);
    qc = 0
    best_path = 0
    highest_fidelity = 0 
    for i, working_qubits in enumerate(combinations_of_hardware_qubits):
        permutes = list(itertools.permutations(working_qubits))
        for permute in permutes:
            qc, best_path, highest_fidelity = get_best_circuit(H,permute, qc, best_path, highest_fidelity, noise_dict)
    return qc, best_path, highest_fidelity         

def get_best_circuit(grid,permute, qc,best_path, highest_fidelity,noise_dict):
    #permutes = list(itertools.permutations(working_qubits))[0]
    
    pairs_of_qubits = nx.utils.pairwise(permute)
    paths = []
    pairs_of_qubits = list(pairs_of_qubits)
    #print(pairs_of_qubits)
    for i, p in enumerate(pairs_of_qubits):
        paths.append(list(nx.algorithms.simple_paths.all_simple_paths(grid,pairs_of_qubits[i][0], pairs_of_qubits[i][1])))
    #paths.append(list(nx.algorithms.simple_paths.all_simple_paths(grid,pairs_of_qubits[0], pairs_of_qubits[1])))
    branch_size_list = [0]*len(paths)
    num_nodes = grid.number_of_nodes()
    return create_circuit(num_nodes,paths, 0,[], branch_size_list,qc,best_path,highest_fidelity,noise_dict,set())

    
def create_circuit(num_nodes,paths, current_path_index,paths_list,bran_size_list,qc,best_path,highest_fidelity, noise_dict, qubit_set):
    #print(paths[current_path_index])
    for i, p in enumerate(map(nx.utils.pairwise, paths[current_path_index])):
        if current_path_index == 0:
            qubit_set = set()
            qubit_set.update(paths[current_path_index][i])
            other_set = set()
        if current_path_index > 0:
            other_set = set(paths[current_path_index][i])
            print(other_set)
            print(len(qubit_set.intersection(other_set)))
        if len(qubit_set.intersection(other_set)) == 1 or  current_path_index == 0:
            qubit_set.update(paths[current_path_index][i])
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
                            q2.add_gate("NU", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = noise_dict[edge])
                        elif l < (num_qubits - 2):
                            q1.add_gate("USWAP", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = 1)
                            q2.add_gate("USWAP", targets = [mapping[edge[0]],mapping[edge[1]]], arg_value = noise_dict[edge])
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
                    print(highest_fidelity)
                    print(qc.gates)
                    print(mapping)
                    print(best_path)
        
        
        if (i+1) == len(paths[current_path_index]) and (current_path_index + 1) == len(paths):
            return qc, best_path, highest_fidelity
        j = current_path_index
        
        if (current_path_index + 1) < len(paths):
            j += 1
            qc,best_path, highest_fidelity = create_circuit(num_nodes,paths,j, current_paths_list,bran_size_list,qc,best_path,highest_fidelity,noise_dict)
    return qc, best_path, highest_fidelity




# first_paths = list(nx.algorithms.simple_paths.all_simple_paths(H,ps[0][0][0], ps[0][0][1]))
# second_paths = list(nx.algorithms.simple_paths.all_simple_paths(H,ps[0][1][0], ps[0][1][1]))
