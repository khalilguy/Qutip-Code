"""circuit_generators.py

Defines the base circuit structure to start optimization from.
"""
import abc
from typing import Callable, Iterable, List, Optional, Tuple, Dict

import numpy as np
import sympy
import cirq


def _H_wall_op_list(qubits, dagger=False):
    """Warning: `dagger` is for preserving decomposed H under inversion. Only
    call with `dagger=True` if the circuit is going to be inverted in a
    symbol-friendly way in the future, i.e. circuit -> circuit[::-1]"""
    out = [cirq.decompose(cirq.H(q)) for q in qubits]
    if dagger:
        out = (cirq.Circuit(*out)[::-1]).all_operations()
    return out


class BaseCircuit(metaclass=abc.ABCMeta):
    """Template class on which circuit architectures are designed.

    """
    def __init__(self, qubits, n_layers, optimizers=None, cid=0, log=print):
        """
        Args:
            optimizers: A list of optimizers to apply to the circuit,
                in the order they occur in the list.
        """
        self._qubits = qubits
        self._n_layers = n_layers
        self.cid = cid
        self.optimizers = optimizers
        self.log = log

    @abc.abstractmethod
    def _U_circuit(self, params_row: np.ndarray, dagger=False) -> cirq.Circuit:
        """Defines the entangling unitary to be interspliced with Hadamards.

        IMPORTANT:

        This method must provide a reversed version of itself if `dagger` is
        specified, but should NOT impose negation on input parameters as
        this will break sympy serialization. For example, if a call to this
        function with SYMBOLS `a` `b` `c` produces

         _U_circuit([a, b, c], dagger=False) = A(a) B(b) C(c)

         Then the following should be the result of `dagger=True`:

         _U_circuit([a, b, c], dagger=True) = C(d(c)) B(d(b)) A(d(a))

         where `d` denotes a string modifier that denotes symbols to be
         negated at parameter resolution time. Typically this is the string
         'dag'.

        Args:
            params_row: a _single row_ from the params array
        """
        raise NotImplementedError(
            "Circuit must define its `U_circuit` method.")

    @abc.abstractmethod
    def _validate_params_to_resolve(self, all_params):
        raise NotImplementedError(
            "Assert params shape requriements for this class.")

    @abc.abstractmethod
    def strings(self, dagger=False):
        """Generate a string-form of symbols for _U_circuit."""
        raise NotImplementedError("Circuit must define its symbols.")

    @abc.abstractproperty
    def resolver_from_params(self,
                             params: np.ndarray,
                             dagger=False) -> Dict[str, float]:
        """Construct a resolver for the input params with on-the-fly symbols.

        This is necessary for any decompositions that add additional symbols.
        This function defines how those symbols are evaluated at circuit
        runtime.

        IMPORTANT:

        This method must negate its input params when `dagger` is specified.
        """
        raise NotImplementedError("Circuit must define its resolver method.")

    def symbols(self, dagger=False):
        """Generate a symbolic form of symbols for _U_circuit.

        WARNING: If this circuit contains decompositions, the decompositions
        are performed at execution time and the resolver used does _not_
        reflect the contents of `symbols`

        @peterse
        TODO: enforcing a shape for params should take care of `symbols`,
        `strings`, `_validate_params_shape`, and `resolver_from_params`.

        Make all of these private, and enforce a `data_shape` property
        """
        return [sympy.Symbol(s) for s in self.strings(dagger=dagger)]

    @property
    def n_layers(self):
        """Return the number of layers for this architecture."""
        return self._n_layers

    @property
    def qubits(self):
        """Return the number of layers for this architecture."""
        return self._qubits

    def _raw_layers_no_outer_hadamards(self,
                                       layer_params,
                                       dagger=False) -> cirq.Circuit:
        """Compose a subcircuit for this architecture, leaving out first and last layers.

        i.e. if the full architecture is H*U*H*U*H, this gives U*H*U. Then a
        hardware circuit can be composed as H*(U*H*U)*(W*H*W)*H where U, W are
        each calls to `_U_circuit`
        Args:
            all_params: A _shaped_ array of parameters to parametrize this circuit.
        """
        circuit = cirq.Circuit()
        # Compose a dagger circuit in reverse, WITHOUT NEGATION.
        # Because of symbol flattening requirements, negation must take
        # place at circuit resolver construction time.

        for i in range(self.n_layers):
            circuit += self._U_circuit(layer_params, dagger=dagger)
            if i != self.n_layers - 1:
                circuit.append(_H_wall_op_list(self.qubits))
        return circuit

    def optimize_circuit(self, circuit):
        """Apply optimizers in sequence."""
        if self.optimizers:
            for optimizer in self.optimizers:
                circuit = optimizer(circuit)
        return circuit

    def simulation_circuit(self) -> cirq.Circuit:
        """Compose a circuit for simulating wavefunctions and density matrices.

        This circuit is fully symbolically parametrized according to the
        symbols/strings implemented by the class.

        Args:
            all_params: A _shaped_ array of parameters to parametrize this circuit.
        """
        circuit = cirq.Circuit(*_H_wall_op_list(self.qubits))
        circuit += self._raw_layers_no_outer_hadamards(self.symbols())
        circuit.append(_H_wall_op_list(self.qubits))
        self.log("SIMULATION CIRCUIT:\n{}".format(circuit))
        self.log("INITIAL PARAMS:\n{}".format(self.symbols()))
        return circuit

    def hardware_circuit(self,
                         measure=True,
                         simplify_one_layer=False) -> cirq.Circuit:
        """Compose a circuit for simulating U(x) U*(y).

        NOTE: This method will automatically generatoe the adjoint U*, which
        means that input parameters should _not_ be negated.

        This circuit is fully symbolically parametrized according to the
        symbols/strings implemented by the class.

        Args:
            layer_params: shaped like (generator.symbols.shape).
            adjoint_layer_params: Same shape, _not_ negated.
            measure: If True, append measurements with default keys (Qid name).
                This is required for any *.run methods to work.
        """
        if simplify_one_layer:

            circuit = cirq.Circuit(*_H_wall_op_list(self.qubits))
            circuit += self._raw_layers_no_outer_hadamards(
                np.asarray(self.symbols()))
            circuit.append(_H_wall_op_list(self.qubits))
        else:
            circuit = cirq.Circuit(*_H_wall_op_list(self.qubits))
            circuit += self._raw_layers_no_outer_hadamards(self.symbols())
            # Dagger kwarg just modifies named symbols in decomposition, and does
            # NOT affect structure
            circuit += self._raw_layers_no_outer_hadamards(
                self.symbols(dagger=True), dagger=True)
            circuit.append(_H_wall_op_list(self.qubits))
        if measure:
            circuit.append(
                cirq.measure(*self.qubits, key="m{}".format(self.cid)))
        self.log("SIMULATION CIRCUIT:\n{}".format(circuit))
        self.log("INITIAL PARAMS:\n{}".format(self.symbols()))
        self.log("INITIAL ADJOINT PARAMS:\n{}".format(
            self.symbols(dagger=True)))
        return self.optimize_circuit(circuit)


class BasePairwiseGenerator(BaseCircuit, metaclass=abc.ABCMeta):
    """Template for pairwise grid circuit generators.

    To implement this class, subclass it and define the `entangling_gate`
    to be the desired entangling gate (does not support decomposition).
    """
    def __init__(self,
                 qubit_map: Iterable[Tuple[int, int]],
                 gate_map: List[Tuple[int, int]],
                 n_layers: int,
                 cid: int = 0,
                 optimizers: Optional[Iterable] = None,
                 log: Optional[Callable] = None):
        """
        Args:
        qubit_map: List of tuples with the qubit coordinates on the grid.
          This list defines the qubit ordering.
        gate_map: List of tuples that define the qubit pairs for ZZ gates.
          Eg. the tuple (2, 4) means that this ZZ gate will act on the 2nd
          and 4th qubits as they are defined by `qubit_map`.
        n_layers: how many layers of pairwise entangling gate sets to generate.
        gate
        """
        # Uses the `qubit_map` and `gate_map` parameters as in `occidentalis`
        qubits = [cirq.GridQubit(i, j) for i, j in qubit_map]
        self._gate_map = gate_map
        if log is None:
            log = print
        super().__init__(qubits,
                         n_layers,
                         cid=cid,
                         optimizers=optimizers,
                         log=log)

    @abc.abstractproperty
    def entangling_gate(self, angle, q1, q2, index):
        """Provides a specific gate for _U_circuit.
        `index` allows on-the-fly symbol generation with a unique index
        corresponding to the original parameter passed.
        """
        raise NotImplementedError(
            "Circuit must define a decomposed entangling gate.")

    def _validate_params_to_resolve(self, layer_params):
        """Defines the proper shape for the parameters array passed to a single circuit."""
        n_gates = len(self._gate_map)
        if len(layer_params) != n_gates:
            raise ValueError("Input params for this circuit needs to "
                             "have length {}. Instead, got shape {}."
                             "".format(n_gates, len(layer_params)))

    def _U_circuit(self, layer_params, dagger=False):
        circuit = cirq.Circuit()

        def reverse_if_dagger(iterator):
            if dagger:
                return list(reversed(list(iterator)))
            else:
                return iterator

        for i, (gm, param) in reverse_if_dagger(
                enumerate(zip(self._gate_map, layer_params))):
            q1, q2 = self.qubits[gm[0]], self.qubits[gm[1]]
            gate = self.entangling_gate(param, q1, q2, i, dagger=dagger)
            circuit.append(gate, strategy=cirq.InsertStrategy.EARLIEST)
        return circuit

    def strings(self, dagger=False):
        dagstr = "dag" if dagger else ""
        n_gates = len(self._gate_map)
        return [
            "phi{}_{}{}".format(self.cid, gate_num, dagstr)
            for gate_num in range(n_gates)
        ]

    @classmethod
    def linear_encoding(cls, X: np.ndarray, c: float, n_qubits: int,
                        gate_map: Iterable[Tuple[int, int]]):
        """Encode X like c*(x_i + x_j) for use by this circuit.

        Args:
        X: Input float data with shape (n_batch, data_dim).
        c: Free weight parameter of the feature map.
        n_qubits: Number of qubits in the circuit.
          Should be the same with `data_dim`.
        gate_map: See `__init__`.
        n_layers: Number of layers in the encoding circuit.

        Returns:
        Weights of the pairs kernel as an array with shape (n_layers, n-choose-2).
        """
        if X.shape[1] != n_qubits:
            raise ValueError(
                "For pairwise linear encoding, data length must be "
                "equal to n_qubits. Instead, got {} for "
                "n_qubits={}".format(X.shape[1], n_qubits))
        n_gates = len(gate_map)
        W = np.zeros((n_qubits, n_gates))
        for i, p in enumerate(gate_map):
            W[p[0], i] = 1
            W[p[1], i] = 1
        return X.dot(W) * c

    @classmethod
    def random_encoding(cls, c, n_qubits, n_layers, X):
        raise NotImplementedError

    @classmethod
    def single_linear_encoding(cls, X: np.ndarray, d: float, n_qubits: int):
        """Encode X like `d * x_i` for use by this circuit.

        Args:
        n_qubits: Number of qubits in the circuit. For the pair kernel
            `n_qubits` == dimension of data.
        n_layers: Number of layers for the circuit.
        data: Dataset to encode

        Returns:
        The weights of the singles kernel as an array with shape X.shape
        """

        if not all(len(x) == n_qubits for x in X):
            raise ValueError("For single linear encoding, data length must be "
                             "equal to n_qubits. Instead, got {} for "
                             "n_qubits={}".format(len(X[0]), n_qubits))
        # "n Choose 2" number of gates
        return d * X


class PairwiseGridqubits(BasePairwiseGenerator):
    """Circuit constructed by a `gate_map` that respects hardware connectivity."""

    # TODO: Perhaps consider merging this with `PairwiseLinequbits`.
    # I don't do this right now to avoid breaking tests.
    def resolver_from_params(self,
                             params: np.ndarray,
                             dagger=False) -> Dict[str, float]:
        self._validate_params_to_resolve(params)
        sgn = -1 if dagger else 1
        return dict(zip(self.strings(dagger=dagger), sgn * np.asarray(params)))

    def entangling_gate(self, angle, q1, q2, index, dagger=False):
        """Provides a specific gate for _U_circuit."""
        return cirq.ZZPowGate(exponent=angle)(q1, q2)



class PairwiseHardwareInspired(PairwiseGridqubits):
    """aka HW-1"""
    def entangling_gate(self, angle, q1, q2, index, dagger=False):
        dagstr = "dag" if dagger else ""
        exp = sympy.Symbol("phi{}_{}".format(self.cid, index) + dagstr)
        dagsign = -1. if dagger else 1

        def gate_generator():
            yield cirq.ISWAP(q1, q2)**(-0.5 * dagsign)
            yield cirq.ZPowGate(exponent=exp).on(q1)
            yield cirq.ZPowGate(exponent=exp).on(q2)

        if dagger:
            out = reversed(list(gate_generator()))
        else:
            out = gate_generator()

        return out