#!/usr/bin/env python
# coding: utf-8

# In[24]:


import gym
from gym import spaces
import gym_qch
import numpy as np
from numpy.random import randint
import networkx as nx


from qutip import *
from IPython.display import Image
from qutip.qip.operations import *
from qutip.qip.circuit import * 
from qutip.qip.device import Processor
from qutip.qip.device import CircularSpinChain, LinearSpinChain
from qutip.qip.noise import RandomNoise
from qutip.operators import sigmaz, sigmay, sigmax, destroy
from qutip.states import basis
from qutip.metrics import fidelity
from qutip.qip.operations import rx, ry, rz, hadamard_transform


import copy
import MyGates as mg
import CircuitUtils as cu
import CircutEmbedding as ce


# In[120]:


class QubitHardwareEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    movements = ['up', 'down', 'left', 'right']
    
    def __init__(self, height_grid, width_grid, width_circuit,current_position=None):  
        
        """
        # Define action and observation space

        We will have two sub-actions:
        
        1. Apply a hadamard, apply a cnot, or apply a swap
        2. Move a pointer left, right, up, or down. This action is ignored on the
        first step because we apply a hadamard 
        #can apply gates to from current possition and choose one
        
        #Instead of allowing the pointer to teleport anywhere it wants restrict it to moving one space in a direction from its current position
        
        """
        super(QubitHardwareEnv, self).__init__()
        self.trapped = False
        self.used_points = []
        self.num_cnots = 0
        self.time_step = 0
        self.movements = ['up', 'down', 'left', 'right']
        self.width_circuit = width_circuit
        self.height_grid = height_grid
        self.width_grid = width_grid
        self.gates_list = []
        
        if current_position != None:
            self.current_position = current_position
        else:
            self.current_position = None
        #self.previous_position = 0
        #create a noise dictionary to represent coherent noise in the hardware
        self.hardware = nx.generators.lattice.grid_2d_graph(self.height_grid, self.width_grid, create_using = nx.DiGraph)
        self.undirected_hardware = nx.generators.lattice.grid_2d_graph(self.height_grid, self.width_grid)
        self.noise_dict = ce.get_noise_dict(self.hardware,self.undirected_hardware)
        '''
        self.circuit_graph = 0 
        self.circuit = 0
        '''
        #hardware = ce.relabel(hardware)
        self.num_qubits  = self.hardware.number_of_nodes()
        #self.gates_list = []
        self.adj_dict = {node:list(edges) for node,edges in dict(self.hardware.adj).items()}
        self.mapping = {i:j for i,j in enumerate(list(self.hardware.nodes))}
    
    

       
        self.gates = 3 #Hadamard, CNOT, SWAP

        self.action_space = spaces.Tuple([spaces.Discrete(len(self.movements)), spaces.Discrete(self.gates),spaces.Discrete(len(self.mapping))])
        self.observation_space = spaces.Box(low=0, high=1, shape = (self.num_qubits,self.num_qubits) ,dtype=np.float16)
        
    def step(self, action):
        #Gate actions for which gate to apply and which direction to apply that gate
        #print(action)
        direction, gate, initial_position = action
        self.previous_position = copy.copy(self.current_position)
        x,y = self._move(direction)
        done = False
        #print(initial_position)
        #Check if we are out of bounds, if we are end the episode
   

        #This makes is so that we only apply a Hadamard at the first time step. If we apply a different state, 
        #end the episode and if we apply one after the first time step end the episode
        if self.time_step == 0:
            if gate == 0:
                self.current_position = self.mapping[initial_position.tolist()]
                self.used_points.append(self.current_position)
                self.apply_gate(gate)
                reward = self.avg_fidelity()
                self.time_step += 1
            elif gate > 0:
                done = True
                reward = 0
        elif self.time_step > 0:
            if gate == 0:
                done = True
                reward = 0
            elif gate > 0:
                #Use this to check if the circuit is trapped by itself. If so, end the episode
                if gate == 1:
                    self.num_cnots += 1
                
                
                if x >= self.width_grid or x < 0 or y >= self.height_grid or y < 0:
                    done = True
                    reward = 0
                #check if the action put us at a point which has already been passed over.
                #If so, end episode
                elif (x,y) in set(self.used_points):
                    done = True
                    reward = 0
                
                if not done:
                #Use this to check if the circuit is trapped by itself. If so, end the episode
                    p = 0
                    for adj_point in self.adj_dict[(x,y)]:
                        if adj_point in set(self.used_points):
                            p += 1
                        
                    if len(self.adj_dict[(x,y)]) == p and self.num_cnots < (self.width_circuit - 1):
                            done = True
                            reward = 0
                    
                    #Check if we are out of bounds, if we are end the episode
               
                if self.num_cnots < (self.width_circuit - 1) and not done:
                    self.current_position = x,y
                    self.used_points.append(self.current_position)
                    self.apply_gate(gate)
                    self.circuit = nx.convert_matrix.to_numpy_array(self.circuit_graph)
                    reward = self.avg_fidelity()
                    self.time_step += 1

        if self.num_cnots == (self.width_circuit - 1) and not done:
            self.current_position = x,y
            self.used_points.append(self.current_position)
            self.apply_gate(gate)
            self.circuit = nx.convert_matrix.to_numpy_array(self.circuit_graph)
            done = True
            reward = self._fidelity(self.gates_list, self.used_points)
        state =  nx.convert_matrix.to_numpy_array(self.circuit_graph)
        return state, reward, done, {}
    
    def avg_fidelity(self):
        target = 0
        avg_fidelity = 0 
        
        while target < 10:
            current_cnots = copy.copy(self.num_cnots)
            current_gates_list = copy.copy(self.gates_list)
            current_used_points = copy.copy(self.used_points)
            current_point = copy.copy(self.current_position)
            done = False
            trapped = False
            #Need to consider case where it gets stuck in a corner or gets trapped by itself in general
            #print(current_gates_list)
            
            while not done:
                rand_gate = randint(1,3) # CNOT, SWAP
                rand_adj_point = self.adj_dict[current_point][randint(0,len(self.adj_dict[current_point]))]
                
                while rand_adj_point in set(current_used_points):
                    rand_adj_point = self.adj_dict[current_point][randint(0,len(self.adj_dict[current_point]))]
                
                
                current_point = rand_adj_point
                current_used_points.append(current_point)
                
                if rand_gate == 1:
                    current_cnots += 1
                    current_gates_list.append("CNOT")
                elif rand_gate == 2:
                    current_gates_list.append("SWAP")
                    
                p = 0
                for adj_point in self.adj_dict[current_point]:
                    if adj_point in set(current_used_points):
                        p += 1
                if len(self.adj_dict[current_point]) == p and current_cnots < (self.width_circuit - 1) :
                    trapped = True
                    break
                    
                if current_cnots == (self.width_circuit - 1):
                    done = True
                    
            if not trapped:
                avg_fidelity += self._fidelity(current_gates_list, current_used_points)
                target += 1
                #print("target")
        return avg_fidelity/10
            
    def _fidelity(self, gates_list, used_points):
        num_qubits = len(used_points)
        q1 = QubitCircuit(num_qubits)
        q2 = QubitCircuit(num_qubits)
        #print(gates_list)
        q1.user_gates = {"UCNOT": mg.user_cnot, "RCNOT":mg.cnot_swap, "USWAP": mg.user_swap}
        q2.user_gates = {"UCNOT": mg.user_cnot, "RCNOT":mg.cnot_swap, "USWAP": mg.user_swap}
        edges = list(nx.utils.pairwise(used_points))
        #print(edges)
        
        i = 0
        for gate in gates_list:
            if gate == "H":
                q1.add_gate("SNOT",i)
                q2.add_gate("SNOT",i)
                
            elif gate == "CNOT":
                q1.add_gate("UCNOT",targets = [i,i+1 ], arg_value = 1)
                q2.add_gate("UCNOT",targets = [i, i+1 ], arg_value = self.noise_dict[edges[i]])
                i += 1
                
            elif gate == "SWAP":
                q1.add_gate("USWAP",targets = [i,i+1 ], arg_value = 1)
                q2.add_gate("USWAP",targets = [i,i+1], arg_value = self.noise_dict[edges[i]])
                i += 1
        #print(q1.gates)       
        y = gate_sequence_product(q1.propagators())*tensor([basis(2,0)]*num_qubits)
        y2 = gate_sequence_product(q2.propagators())*tensor([basis(2,0)]*num_qubits)
        return fidelity(y,y2)
            
        
    def _move(self, direction):
        x, y = self.current_position
        if self.movements[direction] == 'up':
            y += 1
        elif self.movements[direction] == 'down':
            y -= 1
        elif self.movements[direction] == 'left':
            x -= 1
        else:
            x += 1
        return x,y
    
    def apply_gate(self, gate):
        if gate == 0:
            self.gates_list.append("H")
            
        if gate == 1:
            self.circuit_graph.add_edge(self.previous_position, self.current_position)
            self.gates_list.append("CNOT")

        if gate == 2:
            self.circuit_graph.add_edge(self.previous_position, self.current_position)
            self.gates_list.append("SWAP")
        
        
    def reset(self):
        self.trapped = False
        self.used_points = []
        self.num_cnots = 0
        self.time_step = 0
        if self.current_position == None:
            self.current_position = (randint(0, self.width_grid),randint(0, self.height_grid))
        self.circuit_graph = nx.DiGraph()
        self.circuit_graph.add_nodes_from(self.hardware)
        circuit = nx.convert_matrix.to_numpy_array(self.circuit_graph)
        self.circuit = circuit
        self.gates_list = []
        return circuit

    # Render the environment to the screen
    def render(self, mode='console'):
        print(self.mapping)
        print(self.current_position)
        print(self.used_points)
        print(self.circuit)
    


# In[131]:

'''
env = gym.make('qch-v0')
env = QubitHardwareEnv(3,3,3,(0,0))
print(env.hardware.nodes)
env.reset()
env.render()
#print(env.adj_dict)
#env.circuit_graph.add_edge((0,0),(1,0))
#print(nx.convert_matrix.to_numpy_array(env.circuit_graph))
movements = ['up', 'down', 'left', 'right']
print(env.current_position)
'''





