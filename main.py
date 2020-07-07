# -*- coding: utf-8 -*-
"""
Created on Mon Jul  6 11:38:40 2020

@author: Khalil
"""

import CircutEmbedding as ce
import networkx as nx

grid = nx.generators.lattice.grid_2d_graph(3,3 , create_using = nx.DiGraph)
grid_notdi = nx.generators.lattice.grid_2d_graph(3,3)

H = ce.relabel_grid(grid)
G = ce.relabel_grid(grid_notdi)

noise_dict = ce.get_noise_dict(H,G)
print(noise_dict)


qc, best_path, highest_fidelity = ce.get_bell_state(grid,2,noise_dict)          