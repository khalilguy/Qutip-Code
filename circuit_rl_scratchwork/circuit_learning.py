# ADDITIONAL REQUIREMENTS
# python3 -m pip install --user gym
# sudo apt-get install python-gl
"""
NOTES

1/24 algo is stable but noticing the following, TODO:
    - need to get back to tuning rewards, since they seem stochastic w/r
        to the actions...

"""
import json
import collections
import random

import cirq
import numpy as np
import sympy
import tensorflow as tf
TFQ = False
if TFQ:
    try:
        import tensorflow_quantum as tfq
    except ModuleNotFoundError:
        TFQ = False

import gym
from tensorflow.keras import layers
import networkx as nx

from qkm.utils import (grid_maps, circuit_generators, backends, _matplotlib)

#
# def default_loss(vals, locs, kappa):
#     """Mean squared error
#
#     Args:
#         vals: list of values for sampled kernel locations
#         locs: list corresponding locations for values in `k`
#         kappa: full ideal kernel matrix
#     """
#     out = 0
#     for val, loc in zip(vals, locs):
#         out += (val - kappa[loc])**2 / len(vals)
#     return out

def fidelity_wf(psi_a, psi_b):
    """Compute the fidelity between two wavefunctions along the second axis.

    `psi_a` and `psi_b` should have shape (m, 2**n) and each contain m-many
    wavefunctions over n qubits.

    Args:
        psi_a, psi_b: `(m, 2**n)` array of wavefunctions with amplitudes along
            the second axis.
    Returns:
        fidelity: `(m,)` array of computed fidelities.
    """
    return np.diag(np.abs(psi_a.conj().dot(psi_b.T))**2)


def pop_rz_iswap(op):
    return isinstance(op.gate, (cirq.ISwapPowGate, cirq.ZPowGate))


def load_circuit(path):
    """Load in a previously optimized and serialized circuit from `path`"""
    circuit_text = cirq.read_json(open(path))
    return cirq.read_json(json_text=circuit_text)


GATE_ACTION_SET = [
    cirq.ISWAP**0.5,  # 5
    cirq.rz,  # 6
    cirq.X,  # 7
    "REMOVE",  # 8
]

GATE_SET = [
    None,
    cirq.ISwapPowGate,
    cirq.ZPowGate,
    cirq.YPowGate,
    cirq.XPowGate,
]


def apply_circuit_noise(circuit):
    """PLACEHOLDER: This is where an iteration's circuit would become noisy.

    probably this would be done with something like cirq.Circuit().with_noise.
    """
    return circuit

def gate_set_index(gate):
    for i, g in enumerate(GATE_SET):
        if g and isinstance(gate, g):
            return i
    raise ValueError("Gate {} not in list".format(gate))


class CircuitGeneratorEnv(gym.Env):
    """Subclass a gym environment to interface nicely with DQL.

    Description:
        A d-qubit circuit is initialized

    Observation:
        Full description of circuit at each moment, qubit

    Actions:
        Type: Discrete([len(GATE_ACTION_SET) + 5])
        Num     Action
        0       No move
        1       Move down
        2       Move up
        3       Move right
        4       Move left
        ...     Insert gate from GATE_SET, including gate removal

    Reward:
        Error of current circuit kernel with respect to 'ideal kernel'

    Starting State:
        Initial circuit generated by `base_generator` and input params

    """
    def __init__(self,
                 X,
                 y,
                 base_generator,
                 dimension,
                 seed=1,
                 batch_size=50,
                 render=True,
                 fig=None,
                 ax=None):
        """
        Args:
            X: Training dataset
            y: Training labels (+1/-1 form)
            base_generator: Generator to stage initial circuit.
            seed: RNG
        """

        self.X = X
        self.dimension = dimension


        # The loss depends on how the noiseless vs. noisy invocation of
        # a given circuit performs
        self.noiseless_simulator = cirq.Simulator()
        self.noisy_simulator = cirq.Simulator() #TODO

        # Compose an ordered sequence of gates to try to pop
        qubit_maps, self.gate_map = grid_maps.get_hardware_maps(
            num_qubits=dimension, num_circuits=1)

        # Circuit and generator initialization
        # These will be used to query kernel values at each learning step.
        self.qubit_map = qubit_maps[0]
        self.qubits = [cirq.GridQubit(*xy) for xy in self.qubit_map]
        self.generator_init = base_generator(self.qubit_map,
                                             self.gate_map,
                                             n_layers=2,
                                             cid=0,
                                             log=None)
        self.phases = self.generator_init.linear_encoding(
            X=self.X, c=0.2, n_qubits=self.dimension, gate_map=self.gate_map)
        self.phases_X = np.hstack((self.phases, self.X))

        self.base_circuit = self.generator_init.simulation_circuit()
        self.symbols = ["psi_{}".format(i) for i in range(len(self.qubits))]
        # Maintain a connectivity graph for this qubit architecture
        self.G = nx.Graph()
        self.G.add_edges_from(self.gate_map)

        # Limits on cursor position: half-open intervals
        self.row_min, self.row_max = (0, dimension)
        self.col_min, self.col_max = (0, len(self.base_circuit))
        self.state = self.reset()
        self.current_circuit = self.base_circuit.copy()

        # "Cursor" model: restrict changes to be local, to reduce action space
        self.action_space = gym.spaces.Discrete(len(GATE_ACTION_SET) + 4)

        # Current environment is characterized by gate id for each (row, col)
        # and the current cursor position
        self.observation_space = gym.spaces.Dict({
            "circuit_state":
            gym.spaces.Box(low=0,
                           high=len(GATE_SET),
                           shape=(self.row_max, self.col_max),
                           dtype=np.uint8),
            "cursor_state":
            gym.spaces.MultiDiscrete([self.row_max, self.col_max])
        })

        self.kappa = self._make_ideal_kernel(y)
        # how large of a minibatch of kernel element values to compute.
        self.batch_size = batch_size
        self.memory_length = 32
        self.previous_loss = None

        self.do_render = render
        if render:
            self.circuit_artist = _matplotlib.CircuitRender(self.current_circuit,
                                                            self.qubits,
                                                            fig=fig,
                                                            ax=ax)

        self.seed(seed=seed)

    @staticmethod
    def _make_ideal_kernel(y):
        """Construct K: K(x, x') = 1 if y=y', else 0"""
        out = np.zeros((y.shape[0], y.shape[0]))
        for i, y1 in enumerate(y):
            for j, y2 in enumerate(y):
                out[i, j] = max(0, y1 * y2)
        return out

    def _generate_circuit_state(self, circuit):
        """Construct a (row x col) state of a circuit.

        Constraints:
            - Circuit must be composed only from gates present in GATE_SET

        """
        state = np.zeros((self.row_max, self.col_max))
        for col, moment in enumerate(circuit.moments):
            for op in moment.operations:
                gate_id = gate_set_index(op.gate)
                for qid in op.qubits:
                    # Hash: (qubit_row, qubit_col) -> row
                    qid_row = self.qubit_map.index((qid.row, qid.col))
                    try:
                        state[qid_row, col] = gate_id
                    except:
                        import pdb
                        pdb.set_trace()
        return state

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    @staticmethod
    def _action_to_grid(action):
        """Map actions to grid position changes."""
        if action == 0:
            return np.asarray((-1, 0))
        if action == 1:
            return np.asarray((1, 0))
        if action == 2:
            return np.asarray((0, 1))
        if action == 3:
            return np.asarray((0, -1))

    def _calculate_reward(self, circuit):
        """Calculate the reward to give based on the current state of the circuit.

         Reward is defined as the average fidelity of the noiseless state
        compared to the noisy state over some minibatch of input data.
        """

        batch_inds = np.random.choice(all_inds,
                                    size=self.batch_size,
                                    replace=False)
        data_batch = self.phases[batch_inds, :]
        resolvers = [dict(zip(self.symbols, data)) for data in data_batch]

        # Simulate the current iteration of the circuit with/without noise.
        noisy_states = self.noisy_simulator.simulate_sweep(apply_circuit_noise(circuit), params=resolvers)
        true_states = self.noiseless_simulator.simulate_sweep(circuit, params=resolvers)

        loss = np.average(fidelity_wf(noisy_states, true_states))
        # regularize the very first reward computation
        if self.previous_loss is None:
            self.previous_loss = loss
        reward = loss - self.previous_loss
        self.previous_loss = loss
        return 100 * reward

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
        assert self.action_space.contains(action), "{} ({}) invalid".format(
            action, type(action))

        # print("CURSOR:", self.state["cursor_state"])
        # print("ACTION:", action)

        # Move the cursor
        if action <= 3:
            reward = 0
            self.state["cursor_state"] += self._action_to_grid(action)
            # Terminate for going off of the edge of the circuit
            done = (self.state["cursor_state"][0] < self.row_min
                    or self.state["cursor_state"][1] < self.col_min
                    or self.state["cursor_state"][0] >= self.row_max
                    or self.state["cursor_state"][1] >= self.col_max)
            return self.state, reward, done, {}

        done = False
        # Add / Remove / No Change a gate at cursor
        gate_to_add = GATE_ACTION_SET[action - 4]

        # Restrict ourselves to touching only the data element corresponding to
        # this qubit.
        qubit_i = self.state["cursor_state"][0]
        qubit = self.qubits[qubit_i]
        # datum = self.X[qubit_i]

        moment = self.state["cursor_state"][1]
        op_to_remove = self.current_circuit.operation_at(qubit, moment)
        if op_to_remove:
            self.current_circuit.batch_remove([(moment, op_to_remove)])
        if gate_to_add == "REMOVE" and not op_to_remove:
            # In this case, we did nothing; don't recalculate.
            return self.state, 0, done, {}
        elif gate_to_add is cirq.H:
            self.current_circuit.insert(moment,
                                        gate_to_add.on(qubit),
                                        strategy=cirq.InsertStrategy.EARLIEST)
        elif gate_to_add is cirq.rz:
            # OPTION: How are angles for inserted parametrized gates chosen?

            # symbol = self.generator_init.symbols()[qubit_i]
            # Here I add a new symbol, that refers _directly_ to X[i]
            symbol = sympy.Symbol("psi_{}".format(qubit_i))
            self.current_circuit.insert(moment,
                                        gate_to_add(symbol)(qubit),
                                        strategy=cirq.InsertStrategy.EARLIEST)
        elif isinstance(gate_to_add, cirq.ISwapPowGate):
            # OPTION: how are target/control for two-qubit gate insertions chosen?
            # Randomly assign iSWAP connectivity according to gate map
            neighbors = list(self.G.neighbors(qubit_i))
            neighbor_qubits = [
                cirq.GridQubit(*self.qubit_map[n]) for n in neighbors
            ]
            # Avoid neighboring qubits if they're occupied
            available_neighbor_qubits = [
                q for q in neighbor_qubits
                if not self.current_circuit.operation_at(q, moment)
            ]
            # Clear out a neighboring space to insert a swap
            # import pdb; pdb.set_trace()
            if not any(available_neighbor_qubits):
                open_qubit = np.random.choice(neighbor_qubits)
                op_to_remove = self.current_circuit.operation_at(
                    open_qubit, moment)
                self.current_circuit.batch_remove([(moment, op_to_remove)])
                available_neighbor_qubits.append(open_qubit)

            target_qubit = np.random.choice(available_neighbor_qubits)
            self.current_circuit.insert(moment,
                                        gate_to_add.on(qubit,
                                                       target_qubit)**0.5,
                                        strategy=cirq.InsertStrategy.EARLIEST)

        reward = self._calculate_reward(self.current_circuit)
        self.state = {
            "circuit_state":
            self._generate_circuit_state(self.current_circuit),
            "cursor_state": self.state["cursor_state"],
        }
        return self.state, reward, done, {}

    def reset(self):
        """Resets the state of the circuit.

        Reverts to the original generated circuit and places the cursor in
        the approximate center of the circuit spacetime diagram.
        """
        circuit_state = self._generate_circuit_state(self.base_circuit)
        self.current_circuit = self.base_circuit.copy()
        cursor_position = np.asarray([self.row_max // 2, self.col_max // 2])
        self.state = {
            "circuit_state": circuit_state,
            "cursor_state": cursor_position,
        }
        return self.state

    def render(self, mode='ascii'):
        if not self.do_render:
            return
        if mode == 'ascii':
            print(self.current_circuit)
        elif mode == 'mpl':
            y, x = self.state["cursor_state"]
            self.circuit_artist.render(self.current_circuit, (x, y))


class DQN:
    """Deep Q-learning reinforcement learning model."""
    def __init__(self, env, gamma=0.9, epsilon=0.95):
        self.env = env
        self.memory = collections.deque(maxlen=2500)

        # Q-learning hyperparameters
        self.memory_length = 4
        self.learning_rate = 0.01
        # future reward deprecation
        self.gamma = gamma

        # Exploration hyperparameters
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999

        # Dueling network architectures: arXiv:1511.06581
        # The model will train w/r to value function (expected reward for this
        # state), while the target_model will train w/r to the advantage
        # function (Q-function minus current value)
        self.model = self.create_model()
        self.target_model = self.create_model()

    def _flatten_state(self, state):
        return np.hstack((state["circuit_state"].flatten(),
                          state["cursor_state"])).reshape(1, -1)

    def create_model(self):
        """Generate a DNN for learning w/r to policy."""
        model = tf.keras.Sequential()

        # flatten our state space of cursor_state + circuit_state
        input_dim = np.prod(
            self.env.observation_space["circuit_state"].shape
        ) + self.env.observation_space["cursor_state"].shape[0]

        # Autoencoder model
        model.add(layers.Dense(24, input_dim=input_dim, activation="relu"))
        model.add(layers.Dense(48, activation="relu"))
        model.add(layers.Dense(24, activation="relu"))
        model.add(layers.Dense(self.env.action_space.n))
        model.compile(
            loss="mean_squared_error",
            optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def remember(self, state, action, reward, new_state, done):
        """Retain state and reward memory.

        Learning will take place with respect to a random subset of previous
        state/reward outcomes stored in memory.
        """
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        """Implement Q-learning on a random subset of previous state/rewards."""

        if len(self.memory) < self.memory_length:
            return

        samples = random.sample(self.memory, self.memory_length)
        for sample in samples:
            state, action, reward, new_state, done = sample
            state = self._flatten_state(state)
            new_state = self._flatten_state(new_state)
            target = self.target_model.predict(state)
            if done:
                target[0][action] = reward
            else:
                # Calculate advantage function
                q = max(self.target_model.predict(new_state)[0])
                target[0][action] = reward + q * self.gamma

            self.model.fit(state, target, epochs=1, verbose=0)

    def target_train(self):
        """Update the target model with the current model's behavior."""
        self.target_model.set_weights(self.model.get_weights())

    def act(self, state):
        """Perform an action on the environment."""

        # Retain some exploration at later stages.
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        # EXPLORATION: sample an action from the environment's action space
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        # EXPLOITATION: implement an action based on previously observed state/action values
        return np.argmax(self.model.predict(self._flatten_state(state))[0])
