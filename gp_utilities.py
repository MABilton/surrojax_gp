import jax
import jax.numpy as jnp
from nearest_pd import nearestPD
from scipy.linalg import cholesky

def create_noisy_K(cov_func, noise_flag):
    jitter = 10**(-9)
    def K_noisy(x_1, x_2, params):  
        K = cov_func(x_1, x_2, params)
        K = K + jnp.identity(K.shape[0])*jitter
        K = K + jnp.identity(K.shape[0])*params["noise"] if noise_flag else K
        return K
    return K_noisy

def create_kernel_matrix_func(kernel_func):
    matrix_func = jax.vmap(jax.vmap(kernel_func, in_axes=(0,None,None), out_axes=0), in_axes=(None,0,None), out_axes=1)
    def matrix_scalar_func(x_1, x_2, params):
        return jnp.atleast_2d(matrix_func(x_1, x_2, params).squeeze())
    return matrix_scalar_func

def create_cov_diag_func(kernel_func):
    vectorised_kernel = jax.vmap(kernel_func, in_axes=(0,0,None), out_axes=0)
    def vectorised_kernel_func(x_1, x_2, params):
        return jnp.atleast_2d(vectorised_kernel(x_1, x_2, params).squeeze())
    return vectorised_kernel_func

def chol_decomp(A):
    try:
        L = cholesky(A, lower=True)
    except:
        L = cholesky(nearestPD(A), lower=True)
    return L