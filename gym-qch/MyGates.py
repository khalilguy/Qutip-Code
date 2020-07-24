# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 12:34:56 2020

@author: Khalil
"""
import numpy as np

from qutip import *
from qutip.qip.operations import *
from qutip.qip.circuit import * 
from qutip.qip.device import Processor
from qutip.qip.device import CircularSpinChain, LinearSpinChain
from qutip.qip.noise import RandomNoise
from qutip.operators import sigmaz, sigmay, sigmax, destroy

def user_cnot(a):
    b= 1-a 
    mat = np.zeros((4,4), dtype = np.complex)
    mat[0,0] = mat[1,1] = 1
    mat[2,2] = b/np.sqrt(a**2 + b**2)
    mat[3,3] = -mat[2,2]
    mat[3,2] = mat[2,3] =  a/np.sqrt(a**2 +b**2)
    return Qobj(mat, dims=[[2, 2], [2, 2]])
def cnot_swap(a):
    return tensor([snot(),snot()])*user_cnot(a)*tensor([snot(),snot()])

def user_swap(a):
    return user_cnot(a)*cnot_swap(a)*user_cnot(a)