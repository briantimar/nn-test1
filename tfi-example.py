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
    return hamiltonian(static, [], basis=basis, dtype=np.float64)

def parse_psi(psi, op_list):
    """ Extract observables from psi. Quspin conventions for everything."""
    return [o.expt_value(psi) for o in op_list]

def make_features_dict(psi, basis):
    ops = []
    feature_labels = []
    for i in range(basis.L):
        feature_labels.append('zz{0}{1}'.format(0,i))
        ops.append(zz(0,i,basis))
    
    feature_vals = parse_psi(psi, ops)
    return dict(zip(feature_labels, feature_vals))
    

def make_labels(gvals):
    return (gvals <1).astype(int)

def get_features(gvals, basis):
    N=len(gvals)
    s = np.empty((basis.Ns,N)) 
    for i in range(N):
        h = make_1d_TFI_spin(1, gvals[i], basis)
        _, psi0 = h.eigsh(k=1, which='SA')
        s[:, i] = psi0.reshape(basis.Ns)
    return make_features_dict(s, basis)

from quspin.basis import spin_basis_1d
L=6
basis = spin_basis_1d(L)
N=500
gvals_low = np.linspace(0, 0.5, N//2)
gvals_high =np.linspace(1.5, 2, N//2)
gvals = np.concatenate((gvals_low, gvals_high), axis=0)

features_train = get_features(gvals, basis)
labels_train = make_labels(gvals)

Ntst = 100
gtest = np.linspace(0, 2, Ntst)
features_test = get_features(gtest, basis)
labels_test = make_labels(gtest)

feature_cols = [tf.feature_column.numeric_column(key=f) for f in features_train.keys()]

def train_input_fn(features, labels, batch_size):
    """ Puts features and labels into one dataset, shuffles it, batches it up, and returns the whole thing."""
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    return dataset.shuffle(2*N).repeat().batch(batch_size)

classifier = tf.estimator.DNNClassifier(
            feature_columns=feature_cols, 
            hidden_units=[5,5], 
            n_classes=2)

batch_size = 50
nsteps=2000

def eval_input_fn(features, labels=None, batch_size=50):
    if labels is not None:
        d=(features, labels)
    else:
        d=features
    dataset = tf.data.Dataset.from_tensor_slices(d)
    return dataset.batch(batch_size)

classifier.train( input_fn = lambda: train_input_fn(features_train, labels_train,batch_size ), steps=nsteps)

ev_result = classifier.evaluate(input_fn = lambda: eval_input_fn(features_test, labels_test, batch_size))

print("test accuracy:", ev_result['accuracy'])

predictions = [p for p in classifier.predict(input_fn=lambda: eval_input_fn(features_test,batch_size=batch_size)) ]

logits = [p['logits'] for p in predictions]
pvals = [p['probabilities'].reshape((1,2)) for p in predictions]
pvals = np.concatenate(pvals, axis=0)



import matplotlib.pyplot as plt
plt.plot(gtest, pvals)
plt.plot(gvals, labels_train, 'x')



