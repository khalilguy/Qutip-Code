# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 13:21:39 2020

@author: Khalil
"""

from qutip import *
print(qutip.__version__)
from IPython.display import Image
import networkx as nx


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
