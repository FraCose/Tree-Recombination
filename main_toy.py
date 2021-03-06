# THIS REPOSITORY CONTAINS THE ALGORITHMS EXPLAINED IN THE WORK
# Cosentino, Abate, Oberhauser
# "Markov Chain Approximations to Stochastic Differential Equations by Recombination on Lattice Trees"

####################################################################################
# 
#  - This is the main file to be run to obtain the necessary files for the plots relatively to the toy model.
#
#  - Consider changing the list n = [] if you want to obtain the same plot of the paper (more time is needed).
#
#  - The toy model simulation requires a cuda gpu. However, if a cuda gpu is not available you can use the function 
#    simulate_HM() (uncomment/comment the relative lines).
# 
#  - See the reference and the README file for more details. 
# 
####################################################################################

import numpy as np
from numba import njit, prange, vectorize
import timeit as timeit
import recombination as rb
import cvxpy as cvxpy
import cupy as cp
import copy as copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import string
from scipy.optimize import nnls
import scipy.sparse as sp
import create_grid_toy as grid
import pickle

x_0 = np.array([0.,0.])
T = 1
eps = 1.

FIG = False
DEBUG = False

n = [10, 20]
# n = [10, 20, 30, 40, 50, 60]

for i in n:
    tmp = grid.build_tree(x_0, grid.mu_func, grid.SIGMA_func, i, T, eps, FIG)
    name = 'Results_toy/tree_dict_n%i'%i+'.txt'
    with open(name, 'wb') as fp: pickle.dump(tmp, fp)
    
    name = 'Results_toy/sparse_matrix_n%i'%i
    tree_matrix = grid.from_tree_to_matrix(tmp[0])
    sp.save_npz(name, tree_matrix[0], compressed=True)
    
    name = 'Results_toy/matrix_keys_n%i'%i+'.txt'
    with open(name, 'wb') as fp: pickle.dump([tree_matrix[1], tree_matrix[2]], fp)
    
    name = 'Results_toy/probs_n%i'%i+'.npy'
    npMatrix = tree_matrix[0].toarray()
    prob = grid.final_prob(npMatrix, i)
    with open(name, 'wb') as f: np.save(f, prob)

time_steps_MC = 1000
x_0 = cp.array(x_0)
X = grid.simulate_toy_cupy(x_0, time_steps_MC, T)
# X = grid.simulate_toy(x_0, time_steps_MC, T)
name = 'Results_toy/MC_%i'%time_steps_MC+'.npy'
with open(name, 'wb') as f: np.save(f, X)