import jax
import jax.numpy as jnp

def create_kernel_grad_func(kernel_func, param_names, d_idx):
    def kernel_func_to_grad(x, y, *params):
        param_dict = {}
        for i, value in enumerate(params):
            param_dict[param_names[i]] = value
        kernel_val = kernel_func(x, y, param_dict)
        return kernel_val.reshape(())
    return jax.grad(kernel_func_to_grad, argnums=(d_idx+2))

def create_noisy_cov_func(cov_func, noise_flag):
    jitter = 10**(-10)
    def K_noisy(x_1, x_2, params):  
        K = cov_func(x_1, x_2, params)
        K = K + jnp.identity(K.shape[0])*jitter
        if noise_flag: K = K + jnp.identity(K.shape[0])*params["noise"]
        return K
    return K_noisy

def create_matrix_func(scalar_func, in_axes_inner, in_axes_outer):
    matrix_func = jax.vmap(jax.vmap(scalar_func, in_axes=in_axes_inner, out_axes=0), in_axes=in_axes_outer, out_axes=1)
    def matrix_scalar_func(x_1, x_2, *params):
        return jnp.atleast_2d(matrix_func(x_1, x_2, *params).squeeze())
    return matrix_scalar_func

def create_cov_diag_func(kernel_func, noise_flag):
    def kernel_plus_noise(x_1, x_2, params):
        noise = params["noise"] if noise_flag else 0.
        return kernel_func(x_1, x_2, params) + noise
    vectorised_kernel = jax.vmap(kernel_plus_noise, in_axes=(0,0,None), out_axes=0)
    def vectorised_kernel_func(x_1, x_2, params):
        return jnp.atleast_2d(vectorised_kernel(x_1, x_2, params).squeeze())
    return vectorised_kernel_func