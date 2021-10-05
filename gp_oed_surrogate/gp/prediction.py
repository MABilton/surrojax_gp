import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from functools import reduce
from operator import mul

MIN_VAR = 1e-9

def create_predict_method(gp_obj, grad=None):
    
    # First create required kernel functions:
    K_1, K_01_diag, K_01 = create_kernel_funs(gp_obj, grad)
    
    K_1, K_01_diag, K_01 = jax.jit(K_1), jax.jit(K_01_diag), jax.jit(K_01)
    
    # Use created kernel functions to create prediction method:
    def predict_method(self, x_new, return_var, return_cov):
        
        if x_new.ndim < 2:
            x_new = x_new.reshape(1, -1)

        # Initialise dictionary to store outputs:
        output_dict = {}
        
        # Compute mean:
        k_1 = K_1(self.x_train, x_new, self.params)
        
        output_dict['mean'] = jnp.einsum("ij...,i->j...", k_1, self.alpha)
        
        if return_var:
            k_01_diag = K_01_diag(x_new, x_new, self.params)
            solve_shape = (k_1.shape[0], reduce(mul, k_1.shape[1:], 1))
            v = cho_solve((self.L, True), k_1.reshape(solve_shape)).reshape(k_1.shape)
            output_dict['var'] = k_01_diag - jnp.einsum("ki...,ki...->i...", k_1, v)
        
        if return_cov:
            k_01 = K(x_new, x_new)
            v = cho_solve((self.L, True), k_1)
            cov = k_01 - jnp.einsum("ki...,kj...->ij...", k_1, v)
            output_dict['cov'] = jax.ops.index_update(cov, (cov<MIN_VAR) & jnp.identity(x_new.shape[0]), MIN_VAR)

        return output_dict

    # Associate this newly-defined method with our GP object:
    predict_method = predict_method.__get__(gp_obj)

    return predict_method

def create_kernel_funs(gp_obj, grad):
    
    # Compute gradients if required:
    if grad is not None:
        kernel_1, kernel_01, shape_1, shape_01 = create_K_grad(gp_obj, grad)
    else:
        kernel_1 = kernel_01 = gp_obj.kernel
        shape_1 = shape_01 = []
        
    # Ensure correct output structure:
    kernel_1_out = lambda x_1, x_2, params : kernel_1(x_1, x_2, params).reshape(shape_1)
    kernel_01_diag_out = lambda x_1, x_2, params : kernel_01(x_1, x_2, params).reshape(shape_1)
    kernel_01_out = lambda x_1, x_2, params : kernel_01(x_1, x_2, params).reshape(shape_01)
    
    # Vectorise:
    K_1 = vectorise_kernel(kernel_1_out)
    K_01_diag = vectorise_kernel(kernel_01_diag_out, diag_only=True)
    K_01 = vectorise_kernel(kernel_01_out)
    
    return (K_1, K_01_diag, K_01)

def vectorise_kernel(kernel, diag_only=False):
    if diag_only:
        kernel_vmap = jax.vmap(kernel, in_axes=(0,0,None), out_axes=0)
    else:
        kernel_vmap = jax.vmap(jax.vmap(kernel, in_axes=(0,None,None), out_axes=0), in_axes=(None,0,None), out_axes=1)
    return kernel_vmap

def create_K_grad(gp_obj, grad):

    for grad_i in grad:

        # Unpack grad dictionary:
        grad_idx, grad_order = grad_i['idx'], grad_i['order']

        # Sort indices which we'd like to differentiate:
        grad_idx.sort()

        # Work out indices NOT to differentiate:
        nongrad_idx = [i for i in range(gp_obj.x_dim) if i not in grad_idx]

        # Convert these to Jax arrays:
        grad_idx = jnp.array(grad_idx).astype(jnp.int32) 
        nongrad_idx = jnp.array(nongrad_idx).astype(jnp.int32)

        # Restructure call signature:
        kernel_1 = unpack_args(gp_obj.kernel, grad_idx, nongrad_idx)
        kernel_01 = unpack_args(gp_obj.kernel, grad_idx, nongrad_idx)
        
        # Keep track of ouput shape:
        grad_1_shape, grad_01_shape = [], []

        # Perform differentiation:
        for _ in range(grad_order):
            # Note that the 1st arg = components we're diffing wrt
            kernel_1 = jax.jacrev(kernel_1, argnums=1)
            kernel_01 = jax.jacfwd(jax.jacrev(kernel_01, argnums=1), argnums=0) 
            # Update shape:
            grad_1_shape += [len(grad_idx)]
            grad_01_shape += [len(grad_idx), len(grad_idx)]
        
        # Repack call signature 'back to normal':
        kernel_1 = repack_args(kernel_1, grad_idx, nongrad_idx)
        kernel_01 = repack_args(kernel_01, grad_idx, nongrad_idx)

    return (kernel_1, kernel_01, grad_1_shape, grad_01_shape)
    
def unpack_args(packed_fun, grad_idx, nongrad_idx):
    
    idx_order = jnp.concatenate((grad_idx, nongrad_idx))
    
    def repack_x(xg, xng):
        x = jnp.concatenate((xg, xng))[idx_order] #if xng else xg
        return x
    
    def unpacked_fun(xg_1, xg_2, xng_1, xng_2, params):
        x_1, x_2 = repack_x(xg_1, xng_1), repack_x(xg_2, xng_2)
        return packed_fun(x_1, x_2, params)
    
    return unpacked_fun

def repack_args(unpacked_fun, grad_idx, nongrad_idx):
    def unpack_x(x):
        xg, xng = x[grad_idx], x[nongrad_idx]
        return (xg, xng)
    def repacked_fun(x_1, x_2, params):
        xg_1, xng_1 = unpack_x(x_1)
        xg_2, xng_2 = unpack_x(x_2)
        return unpacked_fun(xg_1, xg_2, xng_1, xng_2, params)
    return repacked_fun

def create_grad_key(grad):
    if grad is not None:
        grad_strs = []
        for dict_i in grad:
            
            idx_i, order_i = dict_i['idx'], dict_i['order']

            # Remove repeated indices and sort:
            idx_i = list(set(idx_i))
            idx_i.sort()

            # Create string corresponding to current differentiation operation:
            idx_str = ",".join(map(str, idx_i))
            grad_strs.append(f"{idx_str}:{order_i}")

        grad_key = "-".join(grad_strs)
    else:
        grad_key = ''
    return grad_key