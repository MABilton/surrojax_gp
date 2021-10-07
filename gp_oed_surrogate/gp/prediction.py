import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from functools import reduce, update_wrapper
from operator import mul
from types import FunctionType
from inspect import signature

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
            output_dict['cov'] = k_01 - jnp.einsum("ki...,kj...->ij...", k_1, v)

        output_dict = ensure_positivevar(output_dict)
        
        return output_dict

    # Associate this newly-defined method with our GP object:
    predict_method = predict_method.__get__(gp_obj)

    return predict_method

def ensure_positivevar(output_dict):

    if 'var' in output_dict:
        var = output_dict['var']
        output_dict['var'] = jax.ops.index_update(var, var<MIN_VAR, MIN_VAR) 
    
    if 'cov' in output_dict:
        cov = output_dict['cov']
        cov_shape = cov.shape
        cov_axes = list(len(cov_shape))
        diag_mask = jnp.broadcast_to(jnp.identity(cov_shape[0:2]), reversed(cov_shape))
        diag_mask = jnp.moveaxis(diag_mask, cov_axes, reversed(cov_axes))
        output_dict['cov'] = jax.ops.index_update(var, (cov<MIN_VAR) & diag_mask, MIN_VAR) 

    return output_dict

def create_kernel_funs(gp_obj, grad):
    
    # Compute gradients if required:
    if grad is not None:
        kernel_1, kernel_01_diag, kernel_01 = create_K_grad(gp_obj, grad)
    else:
        kernel_1 = kernel_01_diag = kernel_01 = gp_obj.kernel
    
    # Vectorise:
    K_1 = vectorise_kernel(kernel_1)
    K_01_diag = vectorise_kernel(kernel_01_diag, diag_only=True)
    K_01 = vectorise_kernel(kernel_01)
    
    return (K_1, K_01_diag, K_01)

def vectorise_kernel(kernel, diag_only=False):
    if diag_only:
        kernel_vmap = jax.vmap(kernel, in_axes=(0,0,None), out_axes=0)
    else:
        kernel_vmap = jax.vmap(jax.vmap(kernel, in_axes=(0,None,None), out_axes=0), in_axes=(None,0,None), out_axes=1)
    return kernel_vmap

def create_K_grad(gp_obj, grad):

    # Initialise the kernel functions we've creating:
    kernel_1 = kernel_01_diag = kernel_01 = gp_obj.kernel
    # kernel_1, kernel_01_diag, kernel_01 = deepcopy_fun(gp_obj.kernel, num_copies=3)

    # Keep track of ouput shape:
    grad_1_shape, grad_01_shape = [], []

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
        
        # Unpack call structure of kernel functions:
        kernel_1 = unpack_args(kernel_1, grad_idx, nongrad_idx)
        kernel_01_diag = unpack_args(kernel_01_diag, grad_idx, nongrad_idx)
        kernel_01 = unpack_args(kernel_01, grad_idx, nongrad_idx)

        # Perform differentiation:
        for i in range(grad_order):
            # Note that the 1st arg = components we're diffing wrt
            kernel_1 = jax.jacrev(kernel_1, argnums=1)
            kernel_01_diag = jax.jacrev(jax.jacrev(kernel_01_diag, argnums=1), argnums=0) 
            kernel_01 = jax.jacrev(jax.jacrev(kernel_01, argnums=1), argnums=0) 

            # Extract diagonal elements for diagonal kernel:
            kernel_01_diag = create_diagonal_kernel(kernel_01_diag)
            
            # Need to extract diagonal element if i>0, since we'll be computing cross-derivatives at this point:
            if i>0:
                kernel_1 = create_diagonal_kernel(kernel_1)

            # Update shape:
            grad_1_shape += [len(grad_idx)]
            grad_01_shape += [len(grad_idx), len(grad_idx)]
        
        # Repack call signature 'back to normal':
        kernel_1 = repack_args(kernel_1, grad_idx, nongrad_idx)
        kernel_01_diag = repack_args(kernel_01_diag, grad_idx, nongrad_idx)
        kernel_01 = repack_args(kernel_01, grad_idx, nongrad_idx)

    # Now ensure correct output shape:
    kernel_1 = reshape_fun(kernel_1, grad_1_shape)
    kernel_01_diag = reshape_fun(kernel_01_diag, grad_1_shape)
    kernel_01 = reshape_fun(kernel_01, grad_01_shape)

    return (kernel_1, kernel_01_diag, kernel_01)

def reshape_fun(fun, output_shape):
    fun_reshape = lambda *vargs : fun(*vargs).reshape(output_shape)
    return fun_reshape


# def deepcopy_fun(f, num_copies=1):
#     fun_copies = []
#     for _ in range(num_copies):
#         g = FunctionType(f.__code__, f.__globals__, name=f.__name__,
#                          argdefs=f.__defaults__,
#                          closure=f.__closure__)
#         g = update_wrapper(g, f)
#         g.__kwdefaults__ = f.__kwdefaults__
#         fun_copies.append(g)
#     return fun_copies if num_copies>1 else fun_copies[0]

def unpack_args(packed_fun, grad_idx, nongrad_idx):
    
    idx_order = jnp.concatenate((grad_idx, nongrad_idx))
    
    def repack_x(xg, xng):
        x = jnp.concatenate((xg, xng))[idx_order] #if xng else xg
        return x
    
    def unpacked_fun(xg_1, xg_2, xng_1, xng_2, params):
        x_1, x_2 = repack_x(xg_1, xng_1), repack_x(xg_2, xng_2)
        return packed_fun(x_1, x_2, params)
    
    return unpacked_fun

def create_diagonal_kernel(kernel):

    def kernel_diag(*vargs):
        k = kernel(*vargs)
        k_diag = jnp.diagonal(k, axis1=-2, axis2=-1)
        return k_diag

    return kernel_diag

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