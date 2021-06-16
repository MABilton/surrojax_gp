
import jax
import jax.numpy as jnp

from gp_utilities import compute_L_and_alpha

def create_grad_gp(GP, order):
    # Compute gradient of kernel we want to predict:
    kernel_grad = GP.kernel
    for dim, diff_order in enumerate(order):
        for i in range(diff_order):
            kernel_grad = create_kernel_grad_fun(kernel_grad, dim)
    
    # Replace kernel in GP model:
    GP.kernel = grad_kernel
    GP.K = create_kernel_matrix_func(GP.kernel)
    return GP_grad
    
def create_kernel_grad_fun(kernel, dim):
    kernel_grad = jax.grad(kernel_grad, argnum=(0,1))
    def kernel_grad_fun(x_1, x_2, params):
        return jnp.atleast2d(kernel_grad(x_1, x_2, params))[dim,dim].squeeze()
    return kernel_grad_fun