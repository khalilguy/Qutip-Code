# ---
# jupyter:
#   jupytext:
#     formats: ipynb,md,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

from qutip import *
print(qutip.__version__)
from IPython.display import Image


# %matplotlib inline 

from numpy import pi

import copy
from qutip.qip.operations import *
from qutip.qip.circuit import * 
import numpy as np
from qutip.qip.device import Processor
from qutip.qip.device import CircularSpinChain, LinearSpinChain
from qutip.qip.noise import RandomNoise
from qutip.operators import sigmaz, sigmay, sigmax, destroy
from qutip.states import basis
from qutip.metrics import fidelity
from qutip.qip.operations import rx, ry, rz, hadamard_transform


# +
q1 = QubitCircuit(4)
q1.add_gate("SNOT", [0])
q1.add_gate("SWAP", [2, 3])
q1.add_gate("SWAP", [1, 2])
q1.add_gate("SWAP", [0, 1])
q2 = QubitCircuit(9)
q2.add_gate("SNOT",[0])
q2.add_gate("SWAP",[0,3])


# -

U_gate1 =  gate_sequence_product(q1.propagators())
U_gate2 =  gate_sequence_product(q2.propagators())
print(U_gate2.full().shape)
new_b = basis(16,0)
#np.matmul(U_gate, new_b)

'''
tlist = np.linspace(0,2,20)
circuit_result = mesolve(U_gate, new_b,tlist)
circuit_result.states[-1] # the finial state
'''

# +
processor = Processor(N=4)
processor.add_control(swap(), targets=[0,1])
processor.add_control(swap(), targets=[2,3])
processor.add_control(swap(), targets=[2,1])
processor.set_all_tlist(np.array([0., pi/2., 2*pi/2, 3*pi/2]))
processor.pulses[0].coeff = np.array([1,1,1])

processor.pulses[1].coeff = np.array([0,1,1])

processor.pulses[2].coeff = np.array([0,0,1])

processor


# +
p1 = CircularSpinChain(9)
p2 = CircularSpinChain(9)

U_list1 = p1.run(q1)
U_physical = gate_sequence_product(U_list1)

U_list2 = p2.run(q2)
U_physical2 = gate_sequence_product(U_list2)

# -

(U_gate1 - U_physical).norm()
(U_gate2 - U_physical2).norm()

basis0 = basis(2**9)
result = p1.run_state(basis0)
print(result.states[-1].tidyup(1.0e-4))
result2 = p2.run_state(basis0)
print(result2.states[-1].tidyup(1.0e-4))

# +
gassian_white = RandomNoise(rand_gen=np.random.normal, dt=0.1, loc=-0.05, scale=0.02)
processor_white = copy.deepcopy(p1)
processor_white.add_noise(gassian_white)  # gausian white noise

processor_white2 = copy.deepcopy(p2)
processor_white2.add_noise(RandomNoise(rand_gen=np.random.normal, dt=0.1, loc=-0.05, scale=0.02))  # gausian white noise

# +
result_white = processor_white.run_state(basis0)
result_white.states[-1].tidyup(1.0e-4)

result_white2 = processor_white2.run_state(basis0)
result_white2.states[-1].tidyup(1.0e-4)
# -

fidelity(result.states[-1], result_white.states[-1])

fidelity(result2.states[-1], result_white2.states[-1])

# +
N = 9
qc_Grover = QubitCircuit(9)


'''
qc_Grover.add_gate("SNOT",[0])
qc_Grover.add_gate("SNOT",[1])
qc_Grover.add_gate("SNOT",[2])
qc_Grover.add_gate("SNOT",[3])

qc_Grover.add_gate("CRZ",controls = [0], targets = [3],arg_value=pi/4)
'''
qc_Grover.add_gate("CNOT",controls = [0], targets = [2])
'''
qc_Grover.add_gate("CRZ",controls = [1], targets = [3],arg_value=-pi/4)
qc_Grover.add_gate("CNOT",controls = [0], targets = [1])
qc_Grover.add_gate("CRZ",controls = [1], targets = [3],arg_value=pi/4)
qc_Grover.add_gate("CNOT",controls = [1], targets = [2])
qc_Grover.add_gate("CRZ",controls = [2], targets = [3],arg_value=-pi/4)
qc_Grover.add_gate("CNOT",controls = [1], targets = [2])
qc_Grover.add_gate("CNOT",controls = [0], targets = [1])
qc_Grover.add_gate("CNOT",controls = [1], targets = [2])
qc_Grover.add_gate("CNOT",controls = [0], targets = [1])
qc_Grover.add_gate("CRZ",controls = [2], targets = [3],arg_value=pi/4)
qc_Grover.add_gate("CNOT",controls = [1], targets = [2])
qc_Grover.add_gate("CRZ",controls = [2], targets = [3],arg_value=-pi/4)
qc_Grover.add_gate("CNOT",controls = [0], targets = [2])
qc_Grover.add_gate("CRZ",controls = [2], targets = [3],arg_value=pi/4)
qc_Grover.add_gate("CRZ",controls = [2], targets = [3],arg_value=pi/4)
'''
qc_Grover2 = qc_Grover.adjacent_gates()





# -

qc_Grover2.png

gassian_white
for pulses in processor_white.get_noisy_pulses():
    pulses.print_info()

print(len(p1.pulses))
print(len(processor_white.get_noisy_pulses()))

for pulse in processor_white2.pulses:
    pulse.print_info()

a = .5*cnot() + .5*csign()

a

a.inv()

a.isunitary


