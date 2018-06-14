#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 00:13:18 2018

@author: btimar

Visualizing ground states?


"""

import sys
import numpy as np
import tensorflow as tf
sys.path.append("/Users/btimar/Documents/ryd-theory-code/python_code")
from ryd_base import make_1d_TFI_spin
from tools import get_sent
from quspin.operators import hamiltonian
from tsne import tsne

def z(i, basis):
    static = [['z', [[1.0, i]]]]
    return hamiltonian(static, [], basis=basis)

def zz(i, j, basis):
    static = [['zz', [[1.0, i, j]]]]
    return hamiltonian(static, [], basis=basis, dtype=np.float64)

def make_labels(gvals):
    return (gvals <1).astype(int)

def get_features(gvals, basis):
    N=len(gvals)
    s = np.empty((N, basis.Ns)) 
    for i in range(N):
        h = make_1d_TFI_spin(1, gvals[i], basis, dtype=np.float64)
        _, psi0 = h.eigsh(k=1, which='SA')
        s[i, :] = psi0.reshape(basis.Ns)
    return s

from quspin.basis import spin_basis_1d
L=8
basis = spin_basis_1d(L, kblock=0, pblock=1)
basis_full = spin_basis_1d(L)
proj=basis.get_proj(np.float64)


N=500
gvals = np.linspace(0, 2.0, N)

states = get_features(gvals, basis)
labels = make_labels(gvals)


psi_full = np.asarray(proj.todense().dot( features.transpose()))

zzops = [zz(0, i, basis_full) for i in range(1,L//2)]
features = np.empty((N, len(zzops)))
for i in range(len(zzops)):
    features[:, i] = zzops[i].expt_value(psi_full)
    

from tools import overlap
overlaps = [np.abs(overlap(features[i, :], features[-1, :]))**2 for i in range(N-1)]
zz1 = zz1op.expt_value(psi_full)




#tSNE params
no_dims = 2
#dimensionality of raw data
d = features.shape[1]
initial_dims = d
perplexity = 30.0



print("Passing to tsne")
y = tsne(features, no_dims=no_dims, initial_dims=initial_dims, perplexity=perplexity)

import matplotlib.pyplot as plt
from EDIO import save

fig, ax=plt.subplots()
plt.scatter(y[:, 0], y[:, 1], c=labels)
#save(fig, "20180613/tsne-tfi-symm-wfs-L={0}".format(L),which='mac')




blockA = (y[:, 1]>0)*(y[:, 0]<15)
blockB = np.logical_not(blockA)

plt.plot(gvals[blockA], zz1[blockA], label='A')
plt.plot(gvals[blockB], zz1[blockB], 'rx',label='B')




