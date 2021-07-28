import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
import numpy as np
from numpy import random
from math import pi, inf
from scipy.optimize import minimize, approx_fprime
from gp_utilities import chol_decomp

# For reproducability, set random seed:
random.seed(2)

# Calls Scipy Optimise function to tune hyperparameters:
def fit_hyperparameters(x_train, y_train, noisy_K, constraints, num_repeats=5, jit_flag=True):
    # Create functions to compute gradient of covariance matrix wrt hyperparameters:
    K_grad_fun = create_K_grad_fun(noisy_K)
    # Jit functions if requested:
    if jit_flag:
        noisy_K = jax.jit(noisy_K) 
        K_grad_fun = jax.jit(K_grad_fun)
    # Create bounds to pass to minimize and to compute initial x guess:
    bounds, bounds_array = create_bounds(constraints)
    # Create loss and gradient of loss function to minimise:
    loss_and_grad = create_loss_and_grad(x_train, y_train, noisy_K, K_grad_fun, constraints)
    # Repeat (non-convex) optimisation problem multiple times; pick best solution:
    best_loss = inf
    for i in range(num_repeats):
        # Initialise random guess:
        rand_vec = random.rand(bounds_array.shape[1])
        x_0 = bounds_array[0,:] + (bounds_array[1,:] - bounds_array[0,:])*rand_vec
        # Perform minimisation:
        opt_result = minimize(loss_and_grad, x_0, method='L-BFGS-B', jac=True, bounds=bounds) 
        print(opt_result)
        # Store solution if best seen:
        if opt_result["fun"] < best_loss:
            best_loss, best_params = opt_result["fun"], opt_result["x"]
    # Convert optimal parameter array into optimal parameter dictionary:
    opt_params = create_param_dict(best_params, constraints.keys())
    return opt_params

def create_loss_and_grad(x_train, y_train, noisy_K, K_grad_fun, constraints):
    # Compute constant term in log-probability loss:
    loss_constant = (x_train.shape[0]/2)*jnp.log(2*pi)
    param_list = list(constraints.keys())
    # Create vectorised version of cho_solve:
    cho_solve_vmap = jax.vmap(cho_solve, in_axes=(None, 0), out_axes=0)
    def loss_and_grad(params):
        # Convert array of param values into dictionary:
        param_dict = create_param_dict(params, param_list)
        K_val = noisy_K(x_train, param_dict)
        L_val = chol_decomp(K_val)
        alpha = jnp.atleast_1d(cho_solve((L_val, True), y_train))
        K_grad_vals = K_grad_fun(x_train, param_dict)
        loss = 0.5*jnp.einsum("i,i->", y_train, alpha) \
             + (jnp.log(jnp.diag(L_val))).sum() + loss_constant
        print(loss)
        alpha_outer = jnp.einsum("i,j->ij", alpha, alpha)
        loss_grad = -0.5*jnp.einsum("ik,jki->j", alpha_outer, K_grad_vals) + \
                     0.5*jnp.einsum("kii->k", cho_solve_vmap((L_val, True), K_grad_vals))
        return (loss, loss_grad)

    # Returned loss and loss_grad must be convert to float64 Numpy arrays: 
    def loss_and_grad_wrapper(params):
        loss, loss_grad = loss_and_grad(params)
        loss, loss_grad = np.array(loss, dtype=np.float64), np.array(loss_grad, dtype=np.float64)
        return (loss, loss_grad)
    return loss_and_grad_wrapper

def create_K_grad_fun(noisy_K):
    K_grad = jax.jacfwd(noisy_K, argnums=1)
    def create_grad_matrix(K_grad_dict):
        K_grad_array = []
        for key in sorted(K_grad_dict.keys()):
            dict_val = jnp.atleast_2d(K_grad_dict[key])
            K_grad_array.append(dict_val)
        K_grad_array = jnp.array(K_grad_array)
        return K_grad_array
    def K_grad_fun(x_train, param_dict):
        K_grad_dict = K_grad(x_train, param_dict)
        K_grad_array = create_grad_matrix(K_grad_dict)
        # Calling psutil.cpu_precent() for 4 seconds
        return K_grad_array
    return K_grad_fun

def create_bounds(constraints, min_val=-10**3, max_val=10**3):
    lb, ub = [], []
    for key in sorted(constraints.keys()):
        value = constraints[key]
        if value is None:
            lb.append(min_val)
            ub.append(max_val)
        else:
            if ">" in value: 
                lb.append(value[">"])  
            else: 
                lb.append(min_val)
            if "<" in value: 
                ub.append(value["<"])  
            else: 
                ub.append(max_val)
    bounds, bounds_array = list(zip(lb, ub)), jnp.array([lb, ub])
    return (bounds, bounds_array)

def create_param_dict(params, param_list):
    param_dict = {}
    for i, key in enumerate(sorted(param_list)):
        param_dict[key] = params[i]
    return param_dict