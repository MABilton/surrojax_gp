import jax
import jax.numpy as jnp
from jax.scipy.linalg import cholesky, cho_solve
from nearest_pd import nearestPD

def compute_L_and_alpha(noisy_K, x_train, y_train, params):
    train_K = noisy_K(x_train, params)
    L = chol_decomp(train_K)
    alpha = jnp.atleast_1d(cho_solve((L, True), y_train).squeeze())
    return (L, alpha)

def create_noisy_K(cov_func, noise_flag):
    jitter = 10**(-8)
    def noisy_K(x, params):  
        K = cov_func(x, x, params)
        K = K + jnp.identity(K.shape[0])*jitter
        K = K + jnp.identity(K.shape[0])*params["noise"] if noise_flag else K
        return K
    return noisy_K

def create_K(kernel_func): # , out_axes=0 , out_axes=1
    matrix_func = jax.vmap(jax.vmap(kernel_func, in_axes=(0,None,None), out_axes=0), in_axes=(None,0,None), out_axes=1)
    def kernel_matrix_func(x_1, x_2, params):
        matrix_val = matrix_func(x_1, x_2, params)
        matrix_val = jnp.atleast_2d(matrix_val.squeeze())
        #atrix_val = matrix_val.reshape(x_1.shape[0], x_2.shape[0])
        return matrix_val
    return kernel_matrix_func

def create_cov_diag(kernel_func):
    vectorised_kernel = jax.vmap(kernel_func, in_axes=(0,0,None), out_axes=0)
    def cov_diag_func(x, params):
        return jnp.atleast_1d(vectorised_kernel(x, x, params).squeeze())
    return cov_diag_func

def chol_decomp(A):
    A_pd = nearestPD(A)
    return cholesky(A_pd, lower=True)