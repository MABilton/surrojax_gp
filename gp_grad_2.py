import jax
import jax.numpy as jnp
from jax.scipy.linalg import cho_solve
from gp_class import GP_Surrogate
from gp_utilities import create_K, create_cov_diag

# order = list of same size as feature vector; i'th entry of order specifies order of differentiation
def create_derivative_gp(GP, idx_2_diff):
    # TODO: Input checks:
    return GP_Grad(GP, idx_2_diff)

# grad_order = LIST of tuples -> each tuple is (list_of_x_indices, order_of_diff)
# Create two functions: One where kernel diff wrt first arg, other where diff wrt both args
class GP_Grad(GP_Surrogate):
    def __init__(self, GP, idx_2_diff): 
        # Transfer relevant attributes from GP input to new GP Grad object:
        self.kernel_file, self.kernel_name, self.kernel = GP.kernel_file, GP.kernel_name, GP.kernel
        self.x_train, self.y_train = GP.x_train, GP.y_train
        self.constraints, self.params = GP.constraints, GP.params
        self.L, self.alpha, self.K = GP.L, GP.alpha, GP.K

        # Create kernel gradient functions:
        x_dim = self.x_train.shape[1]
        grad_kernel_1, grad_kernel_10, grad_kernel_shapes = create_grad_kernels(self.kernel, idx_2_diff, x_dim)

        # Vectorise these two gradient kernel functions:
        # Note that K_grad returns (..., num_train, num_pred)
        #           grad_cov_diag returns (..., num_train, num_pred)
        self.K_grad = jax.vmap(jax.vmap(grad_kernel_1, in_axes=(0,None,None), out_axes=-1), in_axes=(None,0,None), out_axes=-1)
        self.grad_cov_diag = create_grad_cov_diag(grad_kernel_10, grad_kernel_shapes)
        
        # Vectorised cho_solve:
        self.cho_solve_vec = jax.vmap(cho_solve, in_axes=(None,0))

    def predict_grad(self, x_new):
        k_grad = self.compute_k_grad(x_new)
        grad_mean = self.predict_grad_mean(x_new, k_grad=k_grad)
        grad_var = self.predict_grad_var(x_new, k_grad=k_grad)
        return (grad_mean, grad_var)

    def predict_grad_mean(self, x_new, k_grad=None):
        x_new = jnp.atleast_2d(x_new)
        k_grad = self.compute_k_grad(x_new) if k_grad is None else k_grad
        grad_mean = jnp.einsum("...ij,i->j...", k_grad, self.alpha)
        return grad_mean

    def predict_grad_var(self, x_new, k_grad=None, min_var=10**-9):
        x_new = jnp.atleast_2d(x_new)
        k_grad = self.compute_k_grad(x_new) if k_grad is None else k_grad
        grad_reshape = (jnp.prod(jnp.array(k_grad.shape[0:-2])).item(), *k_grad.shape[-2:])
        v = self.cho_solve_vec((self.L, True), k_grad.reshape(grad_reshape))
        v = v.reshape(k_grad.shape)
        grad_var = self.grad_cov_diag(x_new, self.params) - jnp.einsum("...ij,...ij->j...", k_grad, v)
        grad_var = jax.ops.index_update(grad_var, grad_var<min_var, min_var)
        return grad_var

    def compute_k_grad(self, x_new):
        return self.K_grad(self.x_train, x_new, self.params)

# if out_shape > x_diff_dim, then use jacfwd
# If out_shape <= x_diff_dim, use jacrev
def create_grad_kernels(kernel, idx_2_diff, x_dim):
    out_size_1, out_size_10 = 1, 1
    out_shape_1, out_shape_10 = [], []
    grad_kernel_1, grad_kernel_10 = kernel, kernel
    for idx, order in idx_2_diff:
        for i in range(order):
            idx = jnp.array(idx, dtype=jnp.int32)
            out_size_1 *= len(idx)
            fwd_flag = out_size_1<=len(idx)
            grad_kernel_1 = differentiate_kernel_1(grad_kernel_1, idx, x_dim, fwd_flag)
            out_size_10 *= len(idx)
            fwd_flag_1 = out_size_10<=len(idx)
            out_size_10 *= len(idx)
            fwd_flag_2 = out_size_10<=len(idx)
            grad_kernel_10 = differentiate_kernel_10(grad_kernel_10, idx, x_dim, fwd_flag_1, fwd_flag_2)
            # Compute shape of outputs:
            out_shape_1 += [len(idx)]
            out_shape_10 += 2*[len(idx)]
    # Place grad shapes in dictionary:
    grad_kernel_shapes = {"1":out_shape_1, "10":out_shape_10}
    return (grad_kernel_1, grad_kernel_10, grad_kernel_shapes)

def differentiate_kernel_1(kernel, idx, x_dim, fwd_flag):
    # Unpack arguments:
    kernel = unpack_args(kernel, idx, x_dim)
    # Differentiate kernels:
    grad_kernel_1 = jax.jacfwd(kernel, argnums=1) if fwd_flag else jax.jacrev(kernel, argnums=1)
    # Repack arguments:
    grad_kernel_1 = repack_args(grad_kernel_1, idx, x_dim)
    return grad_kernel_1

def differentiate_kernel_10(kernel, idx, x_dim, fwd_flag_1, fwd_flag_2):
    # Unpack arguments:
    kernel = unpack_args(kernel, idx, x_dim)
    # Differentiate kernels:
    grad_kernel_1 = jax.jacfwd(kernel, argnums=1) if fwd_flag_1 else jax.jacrev(kernel, argnums=1)
    grad_kernel_10 = jax.jacfwd(grad_kernel_1, argnums=0) if fwd_flag_2 else jax.jacrev(grad_kernel_1, argnums=0)
    # Repack arguments:
    grad_kernel_10 = repack_args(grad_kernel_10, idx, x_dim)
    return grad_kernel_10

def create_grad_cov_diag(grad_kernel_10, grad_kernel_shapes):
    grad_idx = jnp.indices(grad_kernel_shapes["10"])
    mask = jnp.ones(grad_kernel_shapes["10"])
    for i in range(0,len(grad_kernel_shapes["10"]),2):
        mask = jnp.logical_and(mask, grad_idx[i,:] == grad_idx[i+1,:])
    def grad_cov_diag(x, params):
        grad_10 = grad_kernel_10(x, x, params)
        grad_10_diag = grad_10[mask].reshape(grad_kernel_shapes["1"])
        return grad_10_diag
    # Vectorise this function over multiple prediction points:
    grad_cov_diag_vmap = jax.vmap(grad_cov_diag, in_axes=(0,None), out_axes=0)
    return grad_cov_diag_vmap

# Helper function to change argument structure
def unpack_args(packed_kernel, diff_idx, x_dim):
    nondiff_idx = jnp.array([x for x in range(x_dim) if x not in diff_idx])
    idx_order = jnp.append(diff_idx, nondiff_idx).astype(jnp.int32)
    def kernel_unpacked(x_1_d, x_2_d, x_1_nd, x_2_nd, params):
        x_1, x_2 = jnp.append(x_1_d, x_1_nd), jnp.append(x_2_d, x_2_nd)
        x_1, x_2 = x_1[idx_order], x_2[idx_order]
        return packed_kernel(x_1, x_2, params)
    return kernel_unpacked

def repack_args(unpacked_kernel, diff_idx, x_dim):
    nondiff_idx = jnp.array([x for x in range(x_dim) if x not in diff_idx])
    empty_flag = nondiff_idx.size==0
    def kernel_repacked(x_1, x_2, params):
        x_1_d, x_2_d = x_1[diff_idx], x_2[diff_idx]
        if empty_flag:
          x_1_nd, x_2_nd = jnp.array([]), jnp.array([])
        else:
          x_1_nd, x_2_nd = x_1[nondiff_idx], x_2[nondiff_idx]
        return unpacked_kernel(x_1_d, x_2_d, x_1_nd, x_2_nd, params)
    return kernel_repacked