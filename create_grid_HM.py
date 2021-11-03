# THIS REPOSITORY CONTAINS THE ALGORITHMS EXPLAINED IN THE WORK
# Cosentino, Abate, Oberhauser
# "Markov Chain Approximations to Stochastic Differential Equations by Recombination on Lattice Trees"

####################################################################################
# 
#  - This is the library with all the algorithms/functions. 
# 
#  - build_tree is the most important function, where the main algorithm is coded.
#  
#  - mu*, sigma*, SIGMA* are the functions relative to the Heston Model considered in the reference.
# 
#  - *_cupy are functions that require a cuda gpu to be run.
#
#  - The other functions have a brief description of what they do if it is not already clear from their name.
# 
#  - See the reference for more details. 
# 
####################################################################################

import numpy as np
import cupy as cp
# from numba import njit, prange, vectorize
import timeit as timeit

import recombination as rb
import cvxpy as cvxpy
import copy as copy
import matplotlib.pyplot as plt
from tqdm import tqdm
import string

from scipy.optimize import nnls
import scipy.sparse as sp

# tree building
def build_tree(x_0, mu, SIGMA, n, 
        mu_p, C, K, lambd, xi, rho, 
        T=1, eps=0.1, FIG=False, DEBUG = False):
    
    if FIG: 
        plt.ion()
        plt.xlabel('log(P)')
        plt.ylabel('Variance')
        plt.title('Grid building, n = %i, log(P) | Variance,' % n)

    gamma = np.sqrt(eps/n)

    tree = {}       # the tree is built as a dictionary, and then will be transalted into a matrix
    queue_done = np.array([])
    queue_todo = np.array([x_0])
    queue_approx = np.array([])
    nec_states = np.zeros((5,n))
    
    skipped = 0 # must remain 0
    approx = 0 # count the states that have been approximated, e.g. grid too coarse

    # the time interval is divided in Tn steps
    for i in range(int(T*n+1/n)):
        
        queue_tmp = np.copy(queue_todo)
        print("####################### step = ", i)
        print("####################### approx = ", approx)
        print("####################### todo = ", len(queue_todo))
        print("####################### done = ", len(queue_done))
        print("####################### len tree = ", len(tree))
        

        nec_states[:,i] = np.array([approx, skipped, len(queue_done), len(queue_todo), len(queue_done)+len(queue_todo)])

        mom1 = mu_func(queue_todo, mu_p, C, K, lambd)/n
        mu2 = np.concatenate((mom1[:,0]**2, mom1[:,1]**2, mom1[:,0]*mom1[:,1]), axis=0).reshape(len(mom1),-1)
        var_reshaped, evalues = SIGMA_func(queue_todo, xi, rho)
        var_reshaped /= n
        evalues /= n
        mom2 = mu2 + var_reshaped
        recomb_vector = np.concatenate((mom1[:],mom2[:]), axis=1)
        M = np.max(np.abs(mom1),1) + np.sqrt(2*evalues[:,0]) + np.sqrt(2*evalues[:,1])
        M_grid = np.around(np.max(M) / gamma, 0) * gamma + gamma

        if i>=1: 
            grid = create_grid(-M_grid, M_grid+gamma, gamma)
        else: 
            x_0_new = np.zeros(5)
            x_0_new[0] = np.around(x_0[0] / gamma, 0) * gamma
            x_0_new[1] = np.around(x_0[1] / gamma, 0) * gamma
            x_0_new = [x_0_new[0], x_0_new[1], x_0_new[0]**2, 
                        x_0_new[1]**2, x_0_new[0]*x_0_new[1]]
            grid = create_grid(-M_grid-gamma, M_grid+2*gamma, gamma) + x_0_new
            recomb_vector += x_0_new


        for ii in tqdm(range(len(queue_tmp)), mininterval=1):
        # for ii in range(len(queue_tmp)):
            
            x = queue_tmp[ii,:]
            tmp = np.min(grid[:,1]+x[1])
            if i==0:
                x = np.zeros(2)                 # to be caraful with this
                tmp = np.min(grid[:,1])
                if tmp<0: grid_tmp = np.copy(grid[grid[:,1]>0,:])
                else: grid_tmp = np.copy(grid)
            elif tmp<0:
                # positivity of the variance
                grid_tmp = np.copy(grid[grid[:,1]+x[1]>0,:])
            else:
                idx_tmp = np.sum(np.abs(grid[:,:2])<=M[i]+gamma,1)==2
                grid_tmp = np.copy(grid[idx_tmp,:])

            # ERROR CHECK, it should never happen
            if len(grid_tmp)<=5:
                print("PROBLEM GRID")
                skipped += 1
                continue

            idx = search_idx(x, grid_tmp, queue_done, queue_todo) 
            this_is_approx = 0

            try: w_star, idx_star, _, _, ERR = compute_w_recomb(grid_tmp.copy(), recomb_vector[ii,:], x, queue_done, queue_todo, idx)
            except: ERR = 1
            if ERR != 0: w_star, idx_star, _, _, ERR = optimization_compute_w(grid_tmp.copy(), recomb_vector[ii,:], x, queue_done, queue_todo, DEBUG)[:-1]
            boolean = np.sum(w_star<0)==0 and np.sum(w_star)==1 and ERR!=0
            assert not boolean, "Problem recombination, try to decrease epsilon"
            check = np.sum(grid_tmp[idx_star,:]*w_star[:,np.newaxis],axis=0)
            if not np.allclose(check,recomb_vector[ii,:], rtol=1e-03, atol=1e-05): #r tol=1e-05, atol=1e-08
                approx += 1
                this_is_approx += 1
                queue_approx = np.append(queue_approx, x).reshape(-1,2)
            
            x = queue_tmp[ii,:]
            idx_tmp = np.arange(len(queue_todo))[np.sum(np.isclose(x,queue_todo),1)==2]
            queue_todo = np.delete(queue_todo,idx_tmp,axis=0).reshape(-1,2)
            queue_done = np.append(queue_done, x).reshape(-1,2)

            points = grid_tmp[idx_star,:2] + x
            if i>=1: points = grid_tmp[idx_star,:2] + x
            else: points = grid_tmp[idx_star,:2]

            idx_isin_qt = isin(points.reshape(-1,2), queue_todo.reshape(-1,2))
            idx_isin_qd = isin(points.reshape(-1,2), queue_done.reshape(-1,2))
            idx_isnotin = np.logical_not((idx_isin_qt+idx_isin_qd)>=1)
            queue_todo = np.append(queue_todo, points[idx_isnotin,:]).reshape(-1,2)
            
            string_from = np.array2string(x, precision=16, separator=', ')
            string_to = np.array2string(points[0,:], precision=16, separator=', ')
            tree[string_from] = [[string_to, w_star[0]]]
            
            for j in range(1,len(idx_star)):
                point = points[j,:]
                string_to = np.array2string(point, precision=16, separator=', ')
                tree[string_from].append([string_to, w_star[j]])
            
            if FIG:
                point_0 = plt.plot(x_0[0],x_0[1], 'X', color='blue', label='x_0')[0]
                # points = plt.plot(grid_tmp[:,0]+x[0],grid_tmp[:,1]+x[1], '.', color='grey', alpha=0.2)[0]
                if i>0: points = plt.plot(grid_tmp[:,0]+x[0],grid_tmp[:,1]+x[1], '.', color='grey', alpha=0.2, label='considered grid')[0]
                else: points = plt.plot(grid_tmp[:,0],grid_tmp[:,1], '.', color='grey', alpha=0.2, label='considered grid')[0]
                points1 = plt.plot(queue_done[:,0],queue_done[:,1], 'o', markersize=1.2, color='green', label='points done')[0]
                points2 = plt.plot(queue_todo[:,0],queue_todo[:,1], 'x', markersize=1.2, color='orange', label='points to do')[0]
                if len(queue_approx)>0:
                    points3 = plt.plot(queue_approx[:,0],queue_approx[:,1], 'o', markersize=1.2, color='red', label='points approximated')[0]
                plt.legend(loc='upper right')
                plt.pause(1e-7) 

                point_0.remove()
                points.remove()
                points1.remove()
                points2.remove()

                if len(queue_approx)>0:
                    points3.remove()

    for i in queue_todo:
        string_from = np.array2string(i, precision=16, separator=', ')
        tree[string_from] = [[string_from, 1.]]
    
    print("skipped states = ", skipped)
    print("approximated states = ", approx)
    return tree, nec_states    #, queue_done, queue_todo, queue_approx

# @njit(parallel=True, cache=True)
def isin(a, b, precision=1e-6):
    # a.shape[1] == b.shape[1]
    assert a.shape[1] == b.shape[1], "Error in isin, vectors of different sizes"
    d = b.shape[1]
    a_log = np.zeros(a.shape[0])
    # for i in prange(a.shape[0]):
    for i in range(a.shape[0]):
        tmp = np.sum(np.abs(a[i,:] - b) < precision, 1) == d
        a_log[i] = np.sum(tmp)>0
    return a_log

# @njit(parallel=True, cache=True)
def create_grid(min,max,gamma):
    x = np.arange(min,max,gamma)
    grid = np.zeros((len(x)**2,2+3))
    l = len(x)
    x = np.concatenate((x.reshape(-1,1),np.square(x).reshape(-1,1)),axis=1)
    # for i in prange(l):
    for i in range(l):
        tmp = [x[i,0],x[i,0],x[i,1],x[i,1],x[i,1]]
        grid[l*i+i] = tmp
        for j in range(i):
            tmp = [x[i,0],x[j,0],x[i,1],x[j,1],x[i,0]*x[j,0]]
            grid[l*i+j] = tmp
            tmp = [x[j,0],x[i,0],x[j,1],x[i,1],tmp[-1]]
            grid[l*j+i] = tmp
    return grid

def compute_w_recomb(grid, recomb_vector, x, queue_done, queue_todo, idx=[]):
    # we build the random variable \phi(x) with support on grid[idx_star,:] and probabilities w_star
    # such that the first two moments are matched from recomb_vector

    # recomb_log CANNOT be used, because it needs the weights
    MAX_ITER = 30
    w_star, idx_star, x_star, toc, ERR = rb.recomb_Mor_reset(grid-recomb_vector, MAX_ITER, idx)[:5]
    if ERR==0:
        if np.max(np.abs(grid[idx_star,:].T @ w_star-recomb_vector))>=1e-10:
            print("Recombination not precise")
    if ERR == 0 and len(idx_star)==grid.shape[1]+1:
        # this step is not necessary
        w_star, idx_star, x_star, toc, ERR = optimize_choice_points(grid, recomb_vector, x,
                        idx_star, w_star, x_star, toc, ERR, 
                        queue_done, queue_todo)
    
    return w_star, idx_star, x_star, toc, ERR

def optimize_choice_points(grid, recomb_vector, x,
                            idx_star, w_star, x_star, toc, ERR, 
                            queue_done, queue_todo):
    # using Cosentino et al. "A randomized algorithm to reduce the support of discrete measures "
    # we try to reduce the support og the recombined tree.
    # this step is not necessary
    X = grid-recomb_vector
    l = len(idx_star)
    idx_star_tmp = np.copy(idx_star)
    
    queu_tot = np.append(queue_done.reshape(-1,2), queue_todo.reshape(-1,2),axis=0)

    if l<=X.shape[1]: return w_star, idx_star, x_star, toc, ERR
    
    points = grid[idx_star,:2]+x
    idx_not_in = isin(points.reshape(-1,2), queu_tot.reshape(-1,2))
    idx_not_in = np.arange(len(idx_not_in))[np.logical_not(idx_not_in)]

    for i in idx_not_in:

        idx = np.arange(l) == i
        idx = np.logical_not(idx)
        idx = idx_star[idx]
        cone_basis = X[idx,:]
        try: A = np.linalg.inv(np.transpose(cone_basis))
        except: continue
        AX = A @ np.transpose(X)
        # tmp_1 = indices of the points inside (if any) the inverse cone defined via cone_basis
        tmp_1 = np.transpose(AX<=0)
        tmp_1 = np.arange(len(grid))[np.sum(tmp_1,1)==grid.shape[1]]
        
        if len(tmp_1)>1:
            
            points = grid[tmp_1,:2]+x
            tmp = isin(points.reshape(-1,2),queu_tot.reshape(-1,2))
            tmp = np.arange(points.shape[0])[tmp==1]

            if len(tmp)>0:
                idx_star[i] = tmp_1[tmp[0]]
            else:
                ar_min = np.sum(X[tmp_1,:]**2,1)
                ar_min = np.argmin(ar_min)
                idx_star[i] = tmp_1[ar_min]

    tmp = idx_star != idx_star_tmp
    # if not np.allclose(idx_star,idx_star_tmp):
    if np.sum(tmp)>0:
        x_star = X[idx_star,:]
        d = X.shape[1]
        tmp = np.append(np.transpose(x_star), np.ones(l).reshape(1,l), axis=0)
        tmp1 = np.append(np.zeros(d).reshape(1,d),1)

        try: 
            w_star = nnls(tmp, tmp1)[0]
        except: 
            w_star = cvxpy.Variable(len(idx_star))
            constraints = [0 <= w_star, w_star <= 1, np.ones((1,len(grid))) @ w_star == 1]
            objective = cvxpy.Minimize(cvxpy.sum_squares(x_star.T @ w_star - recomb_vector))
            prob = cvxpy.Problem(objective, constraints)
            # already eps_abs=1e-02 gives a final err of order e-07 TO CHECK BETTER
            result = prob.solve() # verbose=True, eps_abs = 1e-04
            w_star = w_star.value
            w_star[w_star<0] = 0.
    
    return w_star, idx_star, x_star, toc, ERR

def optimization_compute_w(grid, recomb_vector, x, queue_done, queue_todo, DEBUG = False):
    # if the recombination algorithm cannot find a solution we solve here an optimization problem
    # and then use the recombination algorithm with the weights obtained by the optimization problem
    # to reduce the support of the measure. 
    try:
        w = nnls(np.append(grid.T,np.ones(grid.shape[0]).reshape(1,-1),axis=0), 
                np.append(recomb_vector,1))[0]
    except:
        w = cvxpy.Variable(len(grid))
        constraints = [0 <= w, w <= 1, np.ones((1,len(grid))) @ w == 1]
        objective = cvxpy.Minimize(cvxpy.sum_squares(grid.T @ w - recomb_vector))
        prob = cvxpy.Problem(objective, constraints)
        # already eps_abs=1e-02 gives a final err of order e-07
        result = prob.solve() # verbose=True, eps_abs = 1e-04
        w = w.value
    
    w[w<0] = 0.
    # CHECK
    rel_err = np.abs((np.dot(grid.T,w)-recomb_vector)/recomb_vector)
    
    if np.max(rel_err*100)>1e-02 and DEBUG:
        print("opt not precise: rel err ", rel_err*100, " %")
        print("point = ", x)

    if w[w>0].shape[0]<=(grid.shape[1]+1):
        w_star = w[w>0]
        idx_star = np.arange(grid.shape[0])[w>0]
        x_star = grid[idx_star,:]
        toc, ERR, w = np.nan, 0., w
        return w_star, idx_star, x_star, toc, ERR, w
    
    MAX_ITER = 100
    w_star, idx_star, x_star, toc, ERR = rb.recomb_combined(np.copy(grid), MAX_ITER, np.copy(w))[:5]
    # try: w_star, idx_star, x_star, toc, ERR = rb.recomb_Mor_reset(np.copy(grid-np.dot(grid.T,w)), MAX_ITER)[:5]
    # except: w_star, idx_star, x_star, toc, ERR = rb.Tchernychova_Lyons(np.copy(grid), np.copy(w))[:5]
    # except: w_star, idx_star, x_star, toc, ERR = rb.recomb_combined(np.copy(grid), 100, np.copy(w))[:5]
    # if ERR != 0:
    #     # w_star, idx_star, x_star, toc, ERR = rb.Tchernychova_Lyons(np.copy(grid), np.copy(w))[:5]
    #     w_star, idx_star, x_star, toc, ERR = rb.recomb_combined(np.copy(grid), MAX_ITER, np.copy(w))[:5]
    # if np.max(np.abs(grid[idx_star,:].T @ w_star - recomb_vector))>5e-3:
    #     print("problem in recomb_log")
    
    if ERR == 0 and len(idx_star)==grid.shape[1]+1:
        w_star, idx_star, x_star, toc, ERR = optimize_choice_points(grid, np.dot(grid.T,w), x,
                        idx_star, w_star, x_star, toc, ERR, 
                        queue_done, queue_todo)
    assert np.all(w_star>=0), "ERROR WEIGHTS" 
    return w_star, idx_star, x_star, toc, ERR, w

#################### Heston Model definition (after log transformation of the price)
# @njit(cache=True, parallel=True)
def mu_func(x, mu_p, C, K, lambd):
    # dimenion of x must be 2
    mu_1 = mu_p-0.5*x[:,1]
    theta = C /(1+np.exp(x[:,0])) + K
    mu_2 = lambd * (theta - x[:,1])
    return np.append(mu_1.reshape(-1,1),mu_2.reshape(-1,1),1)

# @njit(cache=True, parallel=True)
# einsum not supported by numba
def sigma_fun(x, xi, rho):
    return np.einsum('i,jk->ijk',x[:,1]**0.5,np.array([[1,0],[xi*rho,xi*((1-rho**2)**0.5)]]))

# @njit(parallel=True, cache=True)
def SIGMA_func(x, xi, rho):
    aa = 1
    bb = xi**2
    ab = xi*rho
    evalues = np.zeros((len(x),2))
    var_reshaped = np.zeros((len(x),3))

    # for i in prange(len(x)):
    for i in range(len(x)):
        var = x[i,1]*np.array([[aa,ab],[ab,bb]])
        var_reshaped[i,:] = x[i,1] * np.array([aa, bb, ab])
        # the eigenvalues could be explicited in the modified Heston model considered
        evalues[i,:] = np.linalg.eigvals(var)

    return var_reshaped, evalues

################################################################################

# @njit(cache=True, parallel=True) 
def search_idx(x, grid, queue_done, queue_todo):
    idx = []
    gridx = grid[:,:2]+x
    idx_0 = np.sum(np.abs(grid[:,:2])<1e-10,1)==2
    idx.extend(np.arange(len(grid))[idx_0])
    queu_tot = np.append(queue_done.reshape(-1,2), queue_todo.reshape(-1,2),axis=0)

    # if len(queu_tot)>grid.shape[0]*grid.shape[1]:
    #     idx_tmp = np.random.choice(np.arange(len(queu_tot)), 
    #                         size= grid.shape[0]*grid.shape[1], 
    #                         replace=False)
    #     queu_tot = queu_tot[idx_tmp,:]
    
    idx_in = isin(gridx.reshape(-1,2), queu_tot.reshape(-1,2))
    idx_in = np.arange(len(gridx))[idx_in==1]
    
    if len(idx_in)>=np.shape(grid)[1]:
        idx_in = idx_in[:np.shape(grid)[1]]
        idx.extend(list(idx_in))
    else: #len(idx)<np.shape(grid)[1]:
        idx.extend(list(idx_in))
        idx_tmp = np.arange(len(grid))
        idx_tmp = np.delete(idx_tmp,idx)
        idx_tmp = np.random.choice(idx_tmp, 
                        size=np.shape(grid)[1]-len(idx), 
                        replace=False)
        idx.extend(list(idx_tmp))

    idx[0], idx[-1] = idx[-1], idx[0]
    return idx

def from_tree_to_matrix(tree_dict):
    keys = tree_dict.keys()
    map_dict = dict(zip(list(keys), range(len(keys))))
    rows, cols, vals = [], [], []
    
    key_arr = [i.replace("[", "").replace("]","") for i in keys]
    key_arr = np.array([np.fromstring(i, dtype=np.float64, sep=", ") for i in key_arr]).reshape(-1,2)

    # values_arr = [tree_dict[k] for k in tree_dict.keys()]
    # values_arr = [item[0] for sublist in values_arr for item in sublist]
    # values_arr = list(set(values_arr))
    # values_arr = [values.replace("[", "").replace("]", "") for values in values_arr]
    # values_arr = np.array([np.fromstring(v, dtype=np.float64, sep=", ") for v in values_arr]).reshape(-1,2)
    # values_arr = np.unique(values_arr,axis=0)

    for key, values in tree_dict.items():
        for value in values:
            rows.append(map_dict[key])
            vals.append(value[1])
            try: cols.append(map_dict[value[0]])
            except: # numerical approximation
                value0_not_string = value[0].replace("[", "").replace("]", "")
                value0_not_string = np.array(np.fromstring(value0_not_string, dtype=np.float64, sep=", ")).reshape(-1,2)
                idx_tmp = np.arange(len(key_arr))[np.sum(np.isclose(value0_not_string, key_arr),1)==2][0]
                cols.append(map_dict[list(keys)[idx_tmp]])
    
    return sp.csr_matrix((vals, (rows, cols))), map_dict, key_arr

# @njit(cache=True)
def final_prob(A,n,idx=0):
    # it computes the probability vector of where the system will be in the lattice after n steps
    prob = A[idx,:]
    for i in range(n-1):
        prob = prob @ A
    return prob

###################################### FUNCTIONS for the simulation

# uses the GPU
def simulate_HM_cupy(x_0, n, T,
                mu_p, C, K, lambd, xi, rho,
                N=True):
    if N: N = n*n

    X = cp.zeros((N,2)) + x_0
    for i in range(n):
        mu = mu_func_cupy(X, mu_p, C, K, lambd)
        sigma = sigma_fun_cupy(X, xi, rho)
        DeltaW = cp.random.normal(0.,1.,(N,2))
        X += mu / n + cp.einsum('ijk,ik->ij',sigma,DeltaW[:,:]) / n**0.5
    return X

# uses the GPU
def mu_func_cupy(x, mu_p, C, K, lambd):
    # dimenion of x must be 2
    mu = cp.zeros((len(x),2))
    mu[:,0] = mu_p-0.5*x[:,1]
    theta = C /(1+cp.exp(x[:,0])) + K
    mu[:,1] = lambd * (theta - x[:,1])
    return mu
  
# uses the GPU
def sigma_fun_cupy(x, xi, rho):
    return cp.einsum('i,jk->ijk',x[:,1]**0.5,cp.array([[1,0],[xi*rho,xi*((1-rho**2)**0.5)]]))

# uses the CPU
def simulate_HM(x_0, n, T,
                mu_p, C, K, lambd, xi, rho,
                N=True):
    if N: N = n*n

    X = np.zeros((N,2)) + x_0
    DeltaW = np.random.normal(0.,1.,(N,n,2))
    for i in range(n):
        mu = mu_func(X, mu_p, C, K, lambd)
        sigma = sigma_fun(X, xi, rho)
        X += mu / n + np.einsum('ijk,ik->ij',sigma,DeltaW[:,i,:]) / n**0.5
    return X

##################################################################

if __name__ == 'HM_simulation':
    n = 300
    mu_p = 0. 
    C = 10 
    K = 3
    lambd = 1
    xi = 1
    rho = 1/10
    T = 1

    eps = 0.5

    x_0 = np.array([20.,4.])

    tic = timeit.default_timer()
    simulate_HM(x_0, n, T,
                mu_p, C, K, lambd, xi, rho)
    print('time cpu = ', timeit.default_timer()-tic)

    x_0 = cp.array([20.,4.])

    tic = timeit.default_timer()
    simulate_HM_cupy(x_0, n, T,
                mu_p, C, K, lambd, xi, rho)
    print('time gpu = ', timeit.default_timer()-tic)

if __name__ == 'test__main__':
    n = 10
    
    mu_p = 0.
    lambd = 2.
    rho = -0.2
    xi = 1.
    
    C = 2.
    K = 5.
    
    T = 1
    eps = 1.
    FIG = False
    DEBUG = False

    x_0 = np.array([np.log(100),5.])
    tic = timeit.default_timer()
    tree_dict, nec_states = build_tree(x_0, mu_func, SIGMA_func, n, 
        mu_p, C, K, lambd, xi, rho, 
        T, eps, FIG, DEBUG)

    print("time = ", timeit.default_timer()-tic)
    print(len(tree_dict))

    # lines = plt.plot(np.arange(n), nec_states.T)
    # plt.legend(lines[:5], ['approx', 'skipped', 'queue_done', 'queue_todo', 'queue_tot'])
    # plt.show()

    # X does 900 steps
    X = simulate_HM(x_0, n*n, T,
                mu_p, C, K, lambd, xi, rho)
    
    tree_matrix, map_keys, key_arr = from_tree_to_matrix(tree_dict)
    # OPTION max(P-15, 0) = max(exp(P)-exp(15), 0)
    npMatrix = tree_matrix.toarray()
    prob = np.linalg.matrix_power(npMatrix, n)[0,:]

    option_price_MC = np.mean(np.cos(X[:,0]))

    option_price_tree = np.cos(key_arr[:,0])
    option_price_tree = option_price_tree @ prob

    print(tree_matrix.shape)
