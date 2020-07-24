# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:38:40 2020

@author: Khalil
"""

import CircutEmbedding as ce
import networkx as nx
import matplotlib.pyplot as plt

grid = nx.generators.lattice.grid_2d_graph(3,3 , create_using = nx.DiGraph)
grid_notdi = nx.generators.lattice.grid_2d_graph(3,3)

H = ce.relabel_grid(grid)
G = ce.relabel_grid(grid_notdi)

noise_dict = ce.get_noise_dict(grid,grid_notdi)
for key,value in noise_dict.items():
    noise_dict[key] = round(value,2)
nx.draw_networkx_edge_labels(grid,pos = nx.spring_layout(grid),edge_labels=noise_dict)
#nx.draw_networkx_labels(grid, pos=nx.spring_layout(grid))
nx.draw(grid)
noise_dict = ce.get_noise_dict(H,G)
#print(noise_dict)

iter_list = [0]*5
circuit_widths = list(range(2,7))

'''
for i,j in enumerate(circuit_widths):
    qc, best_path, highest_fidelity,iterations = ce.get_bell_state(grid,j,noise_dict)
    print(iterations)
    print(qc.gates)
    print(highest_fidelity)
    iter_list[i] = iterations

print(iter_list)
fig, ax = plt.subplots()

ax.plot(circuit_widths, iter_list)
ax.set_xlabel('circuit_width')  # Add an x-label to the axes.
ax.set_ylabel('iterations')  # Add a y-label to the axes.
'''