
import numbers
from math import pi
import numpy as np
import jax.numpy as jnp
from arraytainers import Jaxtainer

class ScalarGP:

    _max_param = 1e1
    _min_param = -1*_max_param

    #
    #   Constructor Methods
    #

    def __init__(self, kernel, x, y, params, noise=None, selection=None, mean=None, lb=None, ub=None, preprocessing=None):
        self._data_dict = [{'x': x, 'y': y, 'operator': 'none'}]

        self._mean_dict = {'none': _create_mean_func(mean)}
        self._kernel
        self._K_dict = {'one_operator': {'none': _create_K_funcs(kernel)}, 
                        'two_operators': {}}
        self._params = Jaxtainer(params)

    @staticmethod
    def _create_mean_func(mean, operator=None):
        if mean is None:
            mean = lambda x: jnp.zeros((1,))
        if operator is None:
            mean = operator(mean)
        return jax.vmap(mean, in_axes=0)

    @staticmethod
    def _create_K_funcs(kernel, operator=None):
        if operator is not None:
            kernel_x1 = operator(kernel)
            kernel_x1_x2 = 1
        kernel_funcs = {: jax.vmap(kernel_x1, in_axes=(0,None,None)),
                        : jax.vmap(jax.vmap(kernel_x1_x2, in_axes=(0,None,None), in_axes=(None,0,None)))}
        return kernel_funcs

    def predict(self, x, params):
        mu = []
        K = []
        for idx, data in enumerate(self._data_dict):
            op_name = data['operator']
            mean = self._mean_dict[key]
            kernel = self._kernel_dict[key]
            mu.append


    @property
    def operators(self):
        return [data['operator'] for data in self._data_dict]    

    #
    #   Kernel Methods
    #

    @property
    def kernel(self):
        def kernel_func(x_1, x_2, params=None)
        if params are None:
            params = self._params
        return kernel

    @property
    def K_matrix(self, x_1, x_2, params):
        def K_matrix_func(x_1, x_2, params=None)
        if params are None:
            params = self._params
        return K_matrix_func

    #
    #   Sample methods
    #

    def sample(self, num_samples):
        pass

    #
    #   Function Construction Methods
    #

    def _update_kernel_dict(self, operator, operator_name):
        
        self._check_operator_name(operator_name)
        new_kernel_funcs = {}

        if operator is not None:
            kernel_x1, kernel_x1x2 = self._apply_operator(kernel, operator)
            new_kernel_funcs[operator_name] = {self._x1_key: kernel_x1, self._x1_x2_key: kernel_x1x2}

        for key, kernel in kernel_dict.items():
            kernel_dict[key] = self._vectorise_kernel(kernel)

        return kernel_dict



    #
    #   Negative Log-Likelihood Methods
    #

    def _get_x(self, x):
        if x is None:
            x = self.training_x
        return x

    def logpdf(self, batch_size=None):
        
        perform_minibatch = batch_size is not None
        
        def logpdf_func(params, prng=None):
            
            if (prng is None) and perform_minibatch:
                raise ValueError("Must specify Jax pseudo-random number generator using 'prng' keyword since minibatching is being performed.")
            
            if perform_minibatch:
                data = 
            else:
                data = 
            
            return -0.5* - 0.5*self.data_len*jnp.log(2*pi)

        return logpdf_func

    def neglogpdf(self, batch_size=None)

    #
    #   Data Methods
    #

    def add_data(self, x, y, operator=None, name=None, noise=None):
        pass

    @property
    def num_operators(self):
        return 

    @property
    def data(self):
        return self._data
    
    @property
    def data_len(self):
        return self.data.shape[0]

    #
    #   Python Operator Methods
    #
    
    def apply_operation(self, operator):
        pass self.__class__(operator )

    def __matmul__(self, operator):
        return self.apply_operation(operator)

    def __add__(self, val):
        if isinstance(val, numbers.Number):


    def __sub__(self, val):
        return self.__add__(-1*val)


    def __mult__(self):
        pass

    def __truediv__(self, val):
        return self.__mult__(1/val)
        