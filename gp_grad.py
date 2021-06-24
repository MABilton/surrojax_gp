import jax
import jax.numpy as jnp
from scipy.linalg import solve_triangular

from gp_class import GP_Surrogate
from gp_utilities import create_kernel_matrix_func, create_cov_diag_func

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
        self.constraints, self.params = GP.contraints, GP.params
        self.L, self.alpha = GP.L, GP.alpha

        # Compute relevant Jacobians of kernel function:
        kernel_jac = jax.grad(self.kernel, argnums=1)
        kernel_hess = jax.jacfwd(jax.grad(self.kernel, argnums=0), argnums=1)

        # Create required function which extracts diagonal terms from Hessian gradient:
        kernel_hess_diag = lambda x_1, x_2, params : jnp.diag(kernel_hess(x_1, x_2, params))

        # Vectorise functions:
        self.kernel_jac = jax.vmap(kernel_jac, in_axes=[None,1,None])
        self.kernel_hess_diag = jax.vmap(kernel_hess_diag,in_axes=[1,1,None])

    def predict_jac(self, x_new, k=None):
        k_jac = self.compute_k_jac(x_new)
        mean_grad = self.predict_mean_grad(x_new, k=k_jac)
        var_grad = self.predict_var_grad(x_new, k=k_jac)
        return (mean_grad, var_grad)

    def predict_mean_jac(self, x_new, k_jac=None):
        x_new = jnp.atleast_2d(x_new)
        if k_jac is None:
            k_jac = self.compute_k_jac(x_new)
        mean_grad = self.predict_mean(x_new, k=k_jac)
        return mean_grad.squeeze()

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
        self.constraints, self.params = GP.contraints, GP.params
        self.L, self.alpha = GP.L, GP.alpha
        # Also store order attribute for documentation purposes:
        self.order = order

        # Compute gradient of kernel we want to predict:
        kernel_grad_both_input, kernel_grad_first_input  = create_kernel_grad_fun(self.kernel, self.order)

        # Perform relevant vectorisation on these functions:
        self.K_grad = create_kernel_matrix_func(kernel_grad_first_input)
        self.cov_diag_grad_func = create_cov_diag_func(kernel_grad_both_input)

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
        var = self.cov_diag_grad_func(x_new, x_new, self.params) -  jnp.sum(v*v, axis=0, keepdims=True)
        var = jax.ops.index_update(var, var<min_var, min_var)
        return var.squeeze()

    def compute_k_grad(self, x_new):
        return self.K(self.x_train, x_new, self.params)

# Helper function for GP_Grad __init__ method:
def create_kernel_grad_fun(kernel, order):
    # Initialise gradient functions we're going to create:
    kernel_grad_both, kernel_grad_first = kernel, kernel
    for dim, diff_order in enumerate(order):
        for i in range(diff_order):
            # Diff wrt first argument:
            kernel_grad_first = lambda x_1, x_2, params: jax.grad(kernel_grad_first, argnums=0)(x_1, x_2, params)[dim]
            # Diff wrt both arguments:
            kernel_grad_both = lambda x_1, x_2, params: jax.grad(kernel_grad_both, argnums=0)(x_1, x_2, params)[dim]
            kernel_grad_both = lambda x_1, x_2, params: jax.grad(kernel_grad_both, argnums=1)(x_1, x_2, params)[dim]
    # Function closures:
    def kernel_grad_both_input_func(x_1, x_2, params):
        return kernel_grad_both(x_1, x_2, params)
    def kernel_grad_first_input_func(x_1, x_2, params):
        return kernel_grad_first(x_1, x_2, params)
    # Return grad functions:
    return (kernel_grad_both_input_func, kernel_grad_first_input_func)