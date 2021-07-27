import jax
import jax.numpy as jnp
from scipy.linalg import solve_triangular

from gp_class import GP_Surrogate
from gp_utilities import create_K, create_cov_diag

# order = list of same size as feature vector; i'th entry of order specifies order of differentiation
def create_derivative_gp(GP, order=None):
    # TODO: Error check inputs
    if order is None:
        gp_grad = GP_Jac(GP)
    else:
        gp_grad = GP_Grad(GP, order)
    return gp_grad

# Jacobian of GP:
class GP_Jac(GP_Surrogate):
    def __init__(self, GP): 
        # Transfer relevant attributes from GP input to new GP Grad object:
        self.kernel_file, self.kernel_name, self.kernel = GP.kernel_file, GP.kernel_name, GP.kernel
        self.x_train, self.y_train = GP.x_train, GP.y_train
        self.constraints, self.params = GP.constraints, GP.params
        self.L, self.alpha, self.K = GP.L, GP.alpha, GP.K

        # Compute relevant Jacobians of kernel function:
        kernel_jac = jax.grad(self.kernel, argnums=1)
        kernel_hess = jax.jacfwd(jax.grad(self.kernel, argnums=0), argnums=1)

        # Create required function which extracts diagonal terms from Hessian gradient:
        kernel_hess_diag = lambda x_1, x_2, params : jnp.diag(kernel_hess(x_1, x_2, params))

        # Vectorise functions:
        self.kernel_jac = create_K(kernel_jac)
        self.kernel_hess_diag = create_cov_diag(kernel_hess_diag)

    def predict_jac(self, x_new, k=None):
        k_jac = self.compute_k_jac(x_new)
        mean_grad = self.predict_mean_grad(x_new, k=k_jac)
        var_grad = self.predict_var_grad(x_new, k=k_jac)
        return (mean_grad, var_grad)

    def predict_mean_jac(self, x_new, k_jac=None):
        x_new = jnp.atleast_2d(x_new)
        if k_jac is None:
            k_jac = self.compute_k_jac(x_new)
        print(k_jac.shape)
        print(x_new.shape)
        mean_grad = self.predict_mean(x_new, k=k_jac)
        return mean_grad.squeeze().T

    def predict_var_jac(self, x_new, k_jac=None, min_var=10**(-9)):
        x_new = jnp.atleast_2d(x_new)
        if k_jac is None:
            k_jac = self.compute_k_jac(x_new)
        v = solve_triangular(self.L, k_jac, lower=True)
        var_grad = self.kernel_hess_diag(x_new, x_new, self.params) -  jnp.sum(v*v, axis=0, keepdims=True)
        var_grad = jax.ops.index_update(var_grad, var_grad<min_var, min_var)
        return var_grad.squeeze()

    def compute_k_jac(self, x_new):
        return self.kernel_jac(self.x_train, x_new, self.params)

# Arbitrary derivative of GP (a SINGLE specified gradient):
class GP_Grad(GP_Surrogate):
    def __init__(self, GP, order):
        # Transfer relevant attributes from GP input to new GP Grad object:
        self.kernel_file, self.kernel_name, self.kernel = GP.kernel_file, GP.kernel_name, GP.kernel
        self.x_train, self.y_train = GP.x_train, GP.y_train
        self.constraints, self.params = GP.constraints, GP.params
        self.L, self.alpha, self.K = GP.L, GP.alpha, GP.K
        # Also store order attribute for documentation purposes:
        self.order = order

        # Compute gradient of kernel we want to predict:
        self.kernel_grad_both, self.kernel_grad_first = create_kernel_grad_fun(self.kernel, self.order)
        
        # Perform relevant vectorisation on these functions:
        self.K_grad = create_K(self.kernel_grad_both)
        self.cov_diag_grad = create_cov_diag(self.kernel_grad_first)

    def predict_grad(self, x_new, k=None):
        k_grad = self.compute_k_grad(x_new)
        mean_grad = self.predict_mean_grad(x_new, k=k_grad)
        var_grad = self.predict_var_grad(x_new, k=k_grad)
        return (mean_grad, var_grad)

    def predict_mean_grad(self, x_new, k=None):
        x_new = jnp.atleast_2d(x_new)
        if k is None:
            k_grad = self.compute_k_grad(x_new)
        mean_grad = self.predict_mean(x_new, k=k_grad)
        return mean_grad.squeeze()

    def predict_var_grad(self, x_new, k=None, min_var=10**(-9)):
        x_new = jnp.atleast_2d(x_new)
        if k is None:
            k = self.compute_k_grad(x_new)
        v = solve_triangular(self.L, k, lower=True)
        var = self.cov_diag_grad(x_new, x_new, self.params) -  jnp.sum(v*v, axis=0, keepdims=True)
        var = jax.ops.index_update(var, var<min_var, min_var)
        return var.squeeze()

    def compute_k_grad(self, x_new):
        return self.K_grad(self.x_train, x_new, self.params)

# Helper function for GP_Grad __init__ method:
def create_kernel_grad_fun(kernel, order):
    # Initialise gradient functions we're going to create:
    len_x = len(order)
    kernel_grad_both, kernel_grad_first = [unpack_x_inputs(kernel, len_x)], [unpack_x_inputs(kernel, len_x)]
    # Create list of gradient functions:
    for dim, diff_order in enumerate(order):
        for i in range(diff_order):
            # Diff wrt first argument:
            kernel_grad_first.append(diff_kernel(kernel_grad_first[-1], 0, dim, len_x)) 
            # Diff wrt both arguments:
            kernel_grad_both.append(diff_kernel(kernel_grad_both[-1], 0, dim, len_x))
            kernel_grad_both.append(diff_kernel(kernel_grad_both[-1], 1, dim, len_x))
    # Function closures - functions now have copy of all required derivatve functions:
    def kernel_grad_first_fun(x_1, x_2, params): 
        assert x_1.ndim==1 and x_2.ndim==1
        assert x_1.size==len_x and x_2.size==len_x
        x = jnp.append(x_1, x_2)
        assert x.ndim==1
        return kernel_grad_first[-1](params, *x)
    def kernel_grad_both_fun(x_1, x_2, params):
        assert x_1.ndim==1 and x_2.ndim==1
        assert x_1.size==len_x and x_2.size==len_x
        x = jnp.append(x_1, x_2)
        assert x.ndim==1
        return kernel_grad_both[-1](params, *x)
    # Return grad functions:
    return (kernel_grad_both_fun, kernel_grad_first_fun)

# Helper function required - or else kernel_grad_first[-1] leads to infinite recursion
def diff_kernel(kernel, diff_arg, dim, len_x):
    diff_idx = len_x*diff_arg + dim
    diff_fun = lambda params, *x: jax.grad(kernel, argnums=diff_idx)(params, *x)
    return diff_fun

# Function to unpack x input of kernel function:
# From: (x_1, x_2, params) To: (x_11, x_12, ... x_1N, x_21, x_22, ..., x_2N, params)
def unpack_x_inputs(kernel_func, len_x):
    def unpacked_func(params, *x):
        assert x.ndim==1
        x_1, x_2 = jnp.array(x[0:len_x]), jnp.array(x[len_x:])
        return kernel_func(x_1, x_2, params)
    return unpacked_func
# Function to repack x input of kernel function:
# From: (x_11, x_12, ... x_1N, x_21, x_22, ..., x_2N, params) To: (x_1, x_2, params)
def repack_x_inputs(kernel_func, len_x):
    def repacked_func(x_1, x_2, params):

        return kernel_func()
    return repacked_func
