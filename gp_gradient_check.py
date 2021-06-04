import jax.numpy as jnp

from gp_utilities import create_kernel_grad_func, create_unpacked_kernel_func, create_matrix_func
from gp_optimise import loss_and_grad_func_template

from scipy.optimize import approx_fprime
import numpy as np
import time

epsilon = 10*(-1) #np.sqrt(np.finfo(float).eps)

def check_kernel_grad_matrix(matrix_fun, scalar_fun, x):
    params = [-1., -1.]
    for name, func in matrix_fun.items():
        print(name)
        matrix_grad = func(x,x,*params)
        scalar = scalar_fun[name]
        scalar_grad = np.ones(matrix_grad.shape)
        for i in range(matrix_grad.shape[0]):
            for j in range(matrix_grad.shape[1]):
                scalar_grad[i,j] = scalar(x[i],x[j],*params)
        print(abs(matrix_grad - scalar_grad))

def gp_gradient_check(fun, grad, num_args):
    # Epsilon for FD gradients:
    epsilon = np.sqrt(np.finfo(float).eps)

    # Gridding values for hyperparameters:
    max_arg = 10^5
    min_arg = -1*max_arg
    num_pts = 5000

    # Grid of hyperparameter values to check:
    x_0 = np.linspace(min_arg, max_arg, num_pts).reshape((num_pts, 1))
    x_0 = np.repeat(x_0, num_args, axis=1)
    x_0 = np.meshgrid(*x_0.T)

    for idx, val in enumerate(x_0):
        x_0[idx] = val.flatten()
    x_0 = np.array(x_0).T
    for i in range(x_0.shape[0]):
        approx_grad = approx_fprime(x_0[i,:], fun, epsilon)
        grad_val = []
        for name, grad_func in grad.items():
            grad_val.append(grad_func(x_0[i,:]))
        error = abs(approx_grad - grad_val)
        print(error)
            
def kernel(x_1, x_2, params):
    val = params["const"]*jnp.exp(-0.5*((x_2 - x_1)/params["length"])**2)
    return val

def f(x):
    """The function to predict."""
    return x * np.sin(x)

if __name__ == "__main__":
    # Define training data:
    x_train = jnp.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    y_train = f(x_train).ravel()

    # Decleare parameter names:
    param_names = ["length", "const"]

    # # Check kernel gradient: 
    # fun_grad = {}
    # num_components = 1
    # x_high = 1000
    # x_low = -1*x_high
    # rand_x = np.random.uniform(low=x_low, high=x_high, size=(num_components,2))
    # for idx, name in enumerate(param_names):
    #     fun_grad[name] = lambda params: create_kernel_grad_func(kernel, param_names, idx)(rand_x[:,0], rand_x[:,1], *params)
    # fun = lambda params : create_unpacked_kernel_func(kernel, param_names)(rand_x[:,0], rand_x[:,1], *params)
    # # Gradient check kernel:
    # gp_gradient_check(fun, fun_grad, len(param_names))

    # Check loss function gradient:
    K_fun = create_matrix_func(kernel, (0,None,None), (None,0,None))
    in_axes_inner = [0 if x == 0 else None for x in range(len(param_names)+2)]
    in_axes_outer = [0 if x == 1 else None for x in range(len(param_names)+2)]
    K_grad_fun = {}
    for i, name in enumerate(param_names):
        kernel_grad_func = create_kernel_grad_func(kernel, param_names, i)
        K_grad_fun[name] = create_matrix_func(kernel_grad_func, in_axes_inner, in_axes_outer)
    loss_and_grad = lambda params : loss_and_grad_func_template(params, x_train, y_train, K_fun, K_grad_fun, param_names)
    fun = lambda params : loss_and_grad(params)[0]
    fun_grad = {"all": lambda params : loss_and_grad(params)[1]}
    gp_gradient_check(fun, fun_grad, len(param_names))

    # # Check gradient matrix:
    # x = jnp.atleast_2d([1., 3., 5., 6., 7., 8.]).T
    # in_axes_inner = [0 if x == 0 else None for x in range(len(param_names)+2)]
    # in_axes_outer = [0 if x == 1 else None for x in range(len(param_names)+2)]
    # kernel_grad_func = {}
    # K_grad_fun = {}
    # for i, name in enumerate(param_names):
    #     kernel_grad_func[name] = create_kernel_grad_func(kernel, param_names, i)
    #     K_grad_fun[name] = create_matrix_func(kernel_grad_func[name], in_axes_inner, in_axes_outer)
    # check_kernel_grad_matrix(K_grad_fun, kernel_grad_func, x)
