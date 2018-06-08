#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 23:43:27 2018

@author: btimar

Intended as an interface to existing quspin code.

As a test: looking at ground states on either side of QPT.
Then: what about the middle of the spectrum?

"""
import sys
import numpy as np
import tensorflow as tf
sys.path.append("/Users/btimar/Documents/ryd-theory-code/python_code")
from ryd_base import make_1d_TFI_spin
from tools import get_sent
from quspin.operators import hamiltonian

def z(i, basis):
    static = [['z', [[1.0, i]]]]
    return hamiltonian(static, [], basis=basis)

def zz(i, j, basis):
    static = [['zz', [[1.0, i, j]]]]
    return hamiltonian(static, [], basis=basis)

def parse_psi(psi, op_list):
    """ Extract observables from psi. Quspin conventions for everything."""
    return [o.expt_value(psi) for o in op_list]

def make_features_dict(psi, basis):
    ops = []
    feature_labels = []
    for i in range(basis.L):
        for j in range(i+1, basis.L):
            feature_labels.append('zz{0}{1}'.format(i,j))
            ops.append(zz(i,j,basis))
    
    feature_vals = parse_psi(psi, ops)
    return dict(zip(feature_labels, feature_vals))
    

def make_labels


from quspin.basis import spin_basis_1d
L=6
basis = spin_basis_1d(L)

        
features_dict = make_features_dict(s, basis)





#
#
#
#
#a = np.random.rand(100,100)
#d = tf.data.Dataset.from_tensor_slices(a)
#itr = d.make_one_shot_iterator()
#sess = tf.Session()
#r = sess.run(itr.get_next())
