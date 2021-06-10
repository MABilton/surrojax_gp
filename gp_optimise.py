import jax
import jax.numpy as jnp

import numpy as np

from math import pi, inf

from scipy.linalg import cho_solve
from scipy.optimize import minimize, dual_annealing, shgo

from gp_utilities import chol_decomp

# Calls Scipy Optimise function to tune hyperparameters:
def opt_hyperparams(x_train, y_train, K_fun, K_grad_fun, constraints, num_repeats = 5):
    bounds, bounds_array, idx_2_key = create_bounds(constraints)
    best_loss = inf
    loss_and_grad = lambda params : loss_and_grad_func_template(params, x_train, y_train, K_fun, K_grad_fun, idx_2_key)
    #loss_and_grad = lambda params : loss_and_grad_func_template(params, x_train, y_train, K_fun, K_grad_fun, idx_2_key)[0]
    for i in range(num_repeats):
        rand_vec = np.random.rand(bounds_array.shape[1])
        x_0 = bounds_array[0,:] + (bounds_array[1,:] - bounds_array[0,:])*rand_vec
        opt_result =  minimize(loss_and_grad, x_0, method='L-BFGS-B', jac=True, bounds=bounds) 
        #opt_result = dual_annealing(loss_and_grad, bounds=bounds)
        print(opt_result)
        if opt_result["fun"] < best_loss:
            best_loss = opt_result["fun"]
            best_params = opt_result["x"]
    opt_params = create_param_dict(best_params, idx_2_key)
    return opt_params

def loss_and_grad_func_template(params, x_train, y_train, K_fun, K_grad_fun, idx_2_key):
    # Convert array of param values into dictionary:
    param_dict = create_param_dict(params, idx_2_key)
    K = K_fun(x_train, x_train, param_dict)
    L = chol_decomp(K)
    alpha = jnp.atleast_2d(cho_solve((L, True), y_train)).reshape(K.shape[0], 1)
    alpha_outer = jnp.outer(alpha, alpha)
    K_grad_vals = K_grad_fun(x_train, x_train, param_dict)
    loss_grad = []
    for key in param_dict.keys():
        #K_grad = K_grad_fun[key](x_train, x_train, *params)
        #loss_grad.append(-0.5*jnp.trace(alpha_outer @ K_grad - cho_solve((L, True), K_grad)))
        loss_grad.append(-0.5*jnp.trace(alpha_outer @ K_grad_vals[key] - cho_solve((L, True), K_grad_vals[key])))
    loss  = 0.5*y_train.T @ alpha + jnp.log(jnp.diag(L)).sum() + (K.shape[0]/2)*jnp.log(2*pi)
    loss, loss_grad = np.array(loss, dtype=np.float64).ravel(), np.array(loss_grad, dtype=np.float64).squeeze()
    print(loss)
    return (loss, loss_grad)

def create_param_dict(params, idx_2_key):
    param_dict = {}
    for i, value in enumerate(params):
        param_dict[idx_2_key[i]] = value
    return param_dict

def create_bounds(constraints):
    idx_2_key, lb, ub = [], [], []
    max_val = 1.*10.**5
    min_val =  -1*max_val
    for i, (key, value) in enumerate(constraints.items()):
        idx_2_key.append(key)
        if value is None:
            lb.append(min_val)
            ub.append(max_val)
        else:
            if ">" in value: lb.append(value[">"])  
            else: lb.append(min_val)
            if "<" in value: ub.append(value["<"])  
            else: ub.append(max_val)
    bounds, bounds_array = list(zip(lb, ub)), np.array([lb, ub], dtype=np.float32)
    return (bounds, bounds_array, idx_2_key)