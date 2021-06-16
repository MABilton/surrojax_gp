import jax
import jax.numpy as jnp

import numpy as np

from gp_utilities import chol_decomp, create_noisy_K, create_kernel_matrix_func, compute_L_and_alpha, create_cov_diag_func
from gp_optimise import opt_hyperparams

from scipy.linalg import solve_triangular

from math import pi

# y_train = (num_samples) - learning a single function!
# x_train.size = (num_samples, num_features)
# Constraints of the form: {"a":None, "b":{"<":1, ">":-1}, "c":None, "d":{"<":10}}

class GP_Surrogate:
    def __init__(self, kernel_func, x_train, y_train, constraints):

        # TODO: Check dimensions of x_train and y_train
        self.kernel = kernel_func
        self.x_train = x_train
        self.y_train = y_train
    
        # Create functions to compute covariance matrix:
        self.K = create_kernel_matrix_func(kernel_func)

        # Create function which adds noise or jitter to covariance matrix:
        noise_flag = True if "noise" in constraints else False
        noisy_K = jax.jit(create_noisy_K(self.K, noise_flag))
        
        # Create functions to compute gradient of covariance matrix wrt hyperparameters:
        grad_funcs = jax.jit(jax.jacrev(noisy_K, argnums=2))

        # Optimise hyperparameters of covariance function:    
        self.params = opt_hyperparams(x_train, y_train, noisy_K, grad_funcs, constraints)

        # With optimal hyperparameters, compute covariance matrix and inverse covariance matrix:
        self.L, self.alpha = compute_L_and_alpha(noisy_K, self.x_train, self.y_train, self.params)
        self.cov_diag_func = create_cov_diag_func(kernel_func)

    def predict(self, x_new, min_var=10**(-9)):
        x_new = jnp.atleast_2d(x_new)
        k = self.K(self.x_train, x_new, self.params)
        mean = k.T @ self.alpha 
        v = solve_triangular(self.L, k, lower=True)
        var = self.cov_diag_func(x_new, x_new, self.params) -  jnp.sum(v*v, axis=0, keepdims=True)
        var = jax.ops.index_update(var, var<min_var, min_var)
        return (mean.squeeze(), var.squeeze())