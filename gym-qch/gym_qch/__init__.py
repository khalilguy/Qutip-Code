# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 11:05:44 2020

@author: Khalil
"""

import gym


def register(id, entry_point, kwargs, force=True):
    env_specs = gym.envs.registry.env_specs
    if id in env_specs.keys():
        if not force:
            return
        del env_specs[id]
    gym.register(
        id=id,
        entry_point=entry_point,
        kwargs = kwargs
    )
    
register(
    id='qch-v0',
    entry_point='gym_qch.envs:QubitHardwareEnv',
    kwargs = {'height_grid' : 3, 'width_grid' : 3, 'width_circuit' : 3}
)