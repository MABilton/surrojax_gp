import jax
import numpy as jnp
import jax.scipy.linalg as jlinalg

from nearest_pd import nearestPD

from math import pi, inf

import numpy as np

from gp_func_factory import create_cov_diag_func, create_matrix_func, create_noisy_cov_func, create_kernel_grad_func

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve, solve_triangular
from scipy.optimize import minimize

# y_train = (num_samples) - learning a single function!
# x_train.size = (num_samples, num_features)

# Constraints of the form: {"a":None, "b":{"<":1, ">":-1}, "c":None, "d":{"<":10}}

class GP_Surrogate:
    def __init__(self, kernel_func, x_train, y_train, constraints):
        
        # TODO: Check dimensions of x_train and y_train
        self.x_train = x_train
        self.y_train = y_train
    
        # Create functions to compute covariance matrix:
        self.cov_func = create_matrix_func(kernel_func, (0,None,None), (None,0,None))

        # Create function which adds noise or jitter to covariance matrix:
        noise_flag = True if "noise" in constraints else False
        noisy_cov_func = create_noisy_cov_func(self.cov_func, noise_flag)

        # Create functions to compute gradient of covariance matrix wrt hyperparameters:
        param_names = list(constraints.keys())
        in_axes_inner = [0 if x == 0 else None for x in range(len(param_names)+2)]
        in_axes_outer = [0 if x == 1 else None for x in range(len(param_names)+2)]
        grad_funcs = {}
        for i, name in enumerate(param_names):
            if name != "noise":
                kernel_grad_func = create_kernel_grad_func(kernel_func, param_names, i)
                grad_funcs[name] = create_matrix_func(kernel_grad_func, in_axes_inner, in_axes_outer)
            else:
                grad_funcs[name] = lambda x, y, *params : jnp.identity(x.shape[0])
        
        # Optimise hyperparameters of covariance function:    
        #self.params = opt_hyperparams(x_train, y_train, noisy_cov_func, grad_funcs, constraints)
        self.params = {"const":1., "length":1., "noise":1.}

        # With optimal hyperparameters, compute covariance matrix and inverse covariance matrix:
        train_K = noisy_cov_func(x_train, x_train, self.params)
        self.L = chol_decomp(train_K)
        self.alpha = cho_solve((self.L, True), y_train)
        self.cov_diag_func = create_cov_diag_func(kernel_func, noise_flag)

    def predict(self, x_new):
        x_new = jnp.atleast_2d(x_new)
        k = self.cov_func(self.x_train, x_new, self.params)
        mean = k.T @ self.alpha 
        v = solve_triangular(self.L, k, lower=True)
        var = self.cov_diag_func(x_new, x_new, self.params) -  jnp.sum(v*v, axis=0, keepdims=True)
        return (mean.squeeze(), var.squeeze())

    # Able to be grad'd - TODO:
    def predict_mean():
        pass
    # Able to be grad'd - TODO:
    def predict_var():
        pass

# Calls Scipy Optimise function to tune hyperparameters:
def opt_hyperparams(x_train, y_train, K_fun, K_grad_fun, constraints, num_repeats = 5):
    bounds, bounds_array, idx_2_key = create_bounds(constraints)
    best_loss = inf
    loss_func = lambda params : loss_func_template(params, x_train, y_train, K_fun, idx_2_key)
    loss_grad_func = lambda params : loss_grad_func_template(params, x_train, y_train, K_fun, K_grad_fun, idx_2_key)
    for i in range(num_repeats):
        rand_vec = np.random.rand(bounds_array.shape[1])
        x_0 = bounds_array[0,:] + (bounds_array[1,:] - bounds_array[0,:])*rand_vec
        opt_result =  minimize(loss_func, x_0, method='L-BFGS-B', jac=loss_grad_func, bounds=bounds) 
        print(opt_result)
        if opt_result["fun"] < best_loss:
            best_loss = opt_result["fun"]
            best_params = opt_result["x"]
    opt_params = create_param_dict(best_params, idx_2_key)
    return opt_params

def loss_grad_func_template(params, x_train, y_train, K_fun, K_grad_fun, idx_2_key):
    # Convert array of param values into dictionary:
    param_dict = create_param_dict(params, idx_2_key)
    # Iterate over hyperparameters:
    K = K_fun(x_train, x_train, param_dict)
    L = chol_decomp(K)
    alpha = cho_solve((L, True), y_train)
    alpha_dot = alpha.T @ alpha
    loss_grad = []
    for key in param_dict.keys():
        K_grad = K_grad_fun[key](x_train, x_train, *params)
        loss_grad.append(-0.5*(alpha_dot - jnp.trace(cho_solve((L, True), K_grad))))
    return loss_grad

def loss_func_template(params, x_train, y_train, K_fun, idx_2_key):
    # Convert array of param values into dictionary:
    params = create_param_dict(params, idx_2_key)
    # Compute kernel matrix:
    K = K_fun(x_train, x_train, params)
    # Cholesky decomposition:
    L = chol_decomp(K)
    alpha = cho_solve((L, True), y_train)
    #n = y_train.shape[0]
    loss  = 0.5*y_train.T @ alpha + (jnp.log(jnp.diag(L))).sum() 
    print(loss)
    return loss.ravel()

def create_param_dict(params, idx_2_key):
    param_dict = {}
    for i, value in enumerate(params):
        param_dict[idx_2_key[i]] = value
    return param_dict

def create_bounds(constraints):
    idx_2_key, lb, ub = [], [], []
    max_val = 10^2
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
    bounds, bounds_array = list(zip(lb, ub)), jnp.array([lb, ub])
    return (bounds, bounds_array, idx_2_key)

def chol_decomp(A):
    try:
        L = cholesky(A, lower=True)
    except:
        L = cholesky(nearestPD(A), lower=True)
    return L
