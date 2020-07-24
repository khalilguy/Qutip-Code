# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:21:39 2020

@author: Khalil
"""

from IPython.display import Image
import networkx as nx
import matplotlib.pyplot as plt


import numpy as np




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
    return ordered_nodelist, indexed_nodelist

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



def FindPathsofLengthN(grid,start, end, N):
    paths_length_N = []
    paths = list(nx.algorithms.simple_paths.all_simple_paths(grid, start, end, cutoff = N))
    for i in range(len(paths)):
         if len(paths[i]) == N:
             paths_length_N.append(paths[i])
    return paths_length_N
             




