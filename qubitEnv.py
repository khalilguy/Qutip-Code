# qubit placement environment
import gym
from gym import spaces
import numpy as np
from numpy.random import randint
import networkx as nx

class QubitPlacementEnv(gym.Env):
    ''' An environment for generating the inital qubit placement of a circuit with an RL agent

    Observation: 
        hardware graph, circuit/algorithm to compile, hardware noises, current circuit generated, the 
    stage in the qubit placement

    Actions: qubit placement on 3*3 grid
        Num        Action
        0          (0,0)
        1          (0,1)
        2          (0,2)
        3          (1,0)
        4          (1,1)
        5          (1,2)
        6          (2,0)
        7          (2,1)
        8          (2,2)

    Reward:
        Fidelity of generated circuit

    Starting state:
        No qubit placed
    '''

    reward_range = (0, 1)

    def __init__(self, hardware_graph, circuit_graph, circuit_size, noises):
        self.hardware_graph = hardware_graph
        self.circuit_graph = circuit_graph
        self.circuit_size = circuit_size
        self.noises = noises
        
        self.current_circuit = np.zeros((3,3))
        self.num_qubits_places = 0

        self.action_space = gym.spaces.Tuple([gym.spaces.Discrete(2), gym.spaces.Discrete(2)])
        self.observation_space = gym.spaces.Dict({
            'hardware_graph':
            gym.spaces.Box(low=0, high=1, shape=(9,9), dtype=np.int32),
            'circuit':
            gym.spaces.Box(low=0, high=np.Inf, shape=(9,9), dtype=np.int32),
            'noises':
            gym.spaces.Box(low=0, high=1, shape=(9,9), dtype=np.float32),
            'current circuit':
            gym.spaces.Box(low=0, high=9, shape=(3,3), dtype=np.int32),
            'placement stage':
            gym.spaces.Discrete(9),
        }) 

    def step(self, action):
        """Run one timestep of the environment's dynamics. When end of
        episode is reached, you are responsible for calling `reset()`
        to reset this environment's state.

        Accepts an action and returns a tuple (observation, reward, done, info).

        Args:
            action (object): an action provided by the agent

        Returns:
            observation (object): agent's observation of the current environment
            reward (float) : amount of reward returned after previous action
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        """
        self.num_qubits_places = self.num_qubits_places + 1

        if self.num_qubits_places >= self.circuit_size:
            done = True
        else:
            done = False

        self.current_circuit[action[0]][action[1]] = self.num_qubits_places

        state = {
             'hardware_graph':
            self.hardware_graph,
            'circuit':
            self.circuit_graph,
            'noises':
            self.noises,
            'current circuit':
            self.current_circuit,
            'placement stage':
            self.num_qubits_places
        }

        if done == False:
            reward = 0
        else:
            reward = self.compute_reward()

        return state, reward, done, {}

    def compute_reward(self):
        return 1

    def reset(self):
        """Resets the state of the environment and returns an initial observation.
        
        Returns:
            observation (object): the initial observation.
        """

        self.current_circuit = np.zeros((3,3))
        self.num_qubits_places = 0

        state = {
             'hardware_graph':
            self.hardware_graph,
            'circuit':
            self.circuit_graph,
            'noises':
            self.noises,
            'current circuit':
            self.current_circuit,
            'placement stage':
            self.num_qubits_places
        }

        return state

    # def render(self):

hardware_graph = np.ones((9,9))
circuit_graph = np.zeros((9,9))
circuit_graph[0][1] = 1
circuit_graph[1][0] = 1
noises = np.ones((9,9))
env = QubitPlacementEnv(hardware_graph, circuit_graph, 2, noises)
print(env.current_circuit)

env.step((0,0))
print(env.current_circuit)

env.step((0,1))
print(env.current_circuit)

        



        








