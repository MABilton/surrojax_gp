import jax
import jax.numpy as jnp

import numpy as np

from gp_utilities import create_cov_diag_func, create_matrix_func, create_noisy_cov_func, create_kernel_grad_func
from gp_utilities import chol_decomp
from gp_optimise import opt_hyperparams

from scipy.linalg import cho_solve, solve_triangular

from math import pi

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
        self.params = opt_hyperparams(x_train, y_train, noisy_cov_func, grad_funcs, constraints)

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